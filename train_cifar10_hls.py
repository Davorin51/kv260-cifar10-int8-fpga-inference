#!/usr/bin/env python3
"""Train the small CIFAR-10 CNN intended for a later manual HLS implementation.

Install: python -m pip install torch torchvision numpy
Run:     python train_cifar10_hls.py --epochs 150 --out runs/cifar10_01
GPU:     python train_cifar10_hls.py --device cuda --out runs/cifar10_gpu

Python >= 3.10, PyTorch >= 2.0 and a matching torchvision installation.
For NVIDIA GPU installation see https://pytorch.org/get-started/locally/ .
Default device is CUDA if available, otherwise CPU. Windows is supported;
the default --workers 0 avoids multiprocessing setup issues.

Architecture (NCHW):
  RGB 32x32 -> Conv(3,16,3,pad=1) -> ReLU -> MaxPool(2)
           -> Conv(16,32,3,pad=1) -> ReLU -> MaxPool(2)
           -> Conv(32,64,3,pad=1) -> ReLU -> MaxPool(2)
           -> AvgPool(4) -> Flatten -> Linear(64,10)
All convolutions have stride 1 and a bias. No BatchNorm or Softmax.
Input preprocessing: RGB uint8 -> float32 / 255 -> (x - 0.5) / 0.5.

Outputs:
  best_weights.pt        Best validation model's CPU state_dict (FP32).
  last_checkpoint.pt     Last epoch, model, optimizer, scheduler and settings.
  best_weights_fp32.npz  Best weights/biases as NumPy arrays, NOT quantized.
  config.json            Architecture, preprocessing, classes, versions, settings.
  split_indices.npz      Exact 45,000/5,000 train/validation split.
  history.csv            Per-epoch training/validation metrics.
  results.json           Best epoch, validation and final test metrics.
  golden_samples.npz     First 16 test images, normalized inputs, labels, logits.
  train_cifar10_hls.py   A copy of the training/model source for this run.

best_weights.pt is a state_dict, not an executable model. Load it with:
  model = SmallCifarCNN()
  model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
  model.eval()

An output directory must be new or empty. Checkpoints are saved atomically.
last_checkpoint.pt retains optimizer/scheduler state; this script does not
implement an automatic resume command. It trains a fresh model on each run.

API references:
https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
"""

import argparse
import csv
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class SmallCifarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=4)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def stratified_split(targets, seed):
    """Exactly 500 validation samples per class, from official train set only."""
    labels = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        if len(indices) != 5000:
            raise ValueError('Expected the official CIFAR-10 training set.')
        rng.shuffle(indices)
        val_indices.extend(indices[:500].tolist())
        train_indices.extend(indices[500:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cpu_weights(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def atomic_torch_save(value, path):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, count = 0.0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(1) == labels).sum().item()
            count += batch_size
    return {'loss': loss_sum / count, 'accuracy_pct': 100.0 * correct / count}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='./data')
    parser.add_argument('--out', default=None)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('epochs/batch-size must be positive; workers must be nonnegative.')
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.seed < 2**32:
        parser.error('lr must be positive, weight-decay nonnegative, seed in [0, 2**32).')

    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Install GPU-enabled PyTorch or use --device cpu.')
    device = torch.device(device_name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    out = Path(args.out or ('runs/cifar10_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f'Output must be a new or empty directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    if source != (out / source.name).resolve():
        shutil.copy2(source, out / source.name)

    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm,
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), norm])
    train_data = datasets.CIFAR10(args.data, train=True, download=True, transform=train_transform)
    val_data = datasets.CIFAR10(args.data, train=True, download=False, transform=eval_transform)
    train_indices, val_indices = stratified_split(train_data.targets, args.seed)
    np.savez(out / 'split_indices.npz', train=np.asarray(train_indices), val=np.asarray(val_indices))
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=device.type == 'cuda', worker_init_fn=seed_worker)
    train_loader = DataLoader(Subset(train_data, train_indices), shuffle=True,
                              generator=generator, **loader_options)
    val_loader = DataLoader(Subset(val_data, val_indices), shuffle=False, **loader_options)
    model = SmallCifarCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = {
        'architecture': 'SmallCifarCNN_v1', 'precision': 'FP32', 'arguments': vars(args),
        'device': str(device), 'torch_version': str(torch.__version__),
        'classes': train_data.classes, 'input_shape': [1, 3, 32, 32], 'input_layout': 'NCHW',
        'color_order': 'RGB', 'pixel_divisor': 255.0,
        'mean': [0.5]*3, 'std': [0.5]*3,
        'conv_channels': [3, 16, 32, 64], 'conv_kernel': [3, 3],
        'conv_stride': 1, 'conv_padding': 1, 'conv_bias': True,
        'activation': 'ReLU', 'max_pool_kernel_and_stride': 2,
        'final_average_pool_kernel_and_stride': 4, 'classifier': [64, 10],
        'conv_weight_layout': 'OIHW', 'fc_weight_layout': 'OI',
        'batch_norm': False, 'softmax': False,
        'train_count': len(train_indices), 'val_count': len(val_indices),
        'parameter_count': sum(p.numel() for p in model.parameters()),
    }
    write_json(out / 'config.json', config)
    print(f'Device: {device} | Parameters: {config["parameter_count"]:,}', flush=True)
    print(f'Output: {out.resolve()} | Train: 45000 | Validation: 5000', flush=True)
    print('Starting FP32 training. Test data is evaluated after model selection.', flush=True)

    best_acc, best_epoch = -1.0, 0
    fields = ['epoch', 'lr', 'train_loss', 'train_accuracy_pct', 'val_loss', 'val_accuracy_pct', 'seconds']
    with (out / 'history.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            lr = optimizer.param_groups[0]['lr']
            train_metrics = run_epoch(model, train_loader, device, optimizer)
            val_metrics = run_epoch(model, val_loader, device)
            if not np.isfinite(train_metrics['loss']) or not np.isfinite(val_metrics['loss']):
                raise RuntimeError('Non-finite loss encountered; inspect data and learning rate.')
            scheduler.step()
            weights = cpu_weights(model)
            improved = val_metrics['accuracy_pct'] > best_acc
            if improved:
                best_acc, best_epoch = val_metrics['accuracy_pct'], epoch
                atomic_torch_save(weights, out / 'best_weights.pt')
                np.savez(out / 'best_weights_fp32.npz',
                         **{name: value.numpy() for name, value in weights.items()})
            atomic_torch_save({
                'epoch': epoch, 'model_state_dict': weights,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy_pct': best_acc, 'best_epoch': best_epoch, 'config': config,
            }, out / 'last_checkpoint.pt')
            elapsed = time.perf_counter() - started
            writer.writerow(dict(epoch=epoch, lr=lr, train_loss=train_metrics['loss'],
                                 train_accuracy_pct=train_metrics['accuracy_pct'],
                                 val_loss=val_metrics['loss'], val_accuracy_pct=val_metrics['accuracy_pct'],
                                 seconds=elapsed))
            handle.flush()
            print(f'Epoch {epoch:3d}/{args.epochs} | train {train_metrics["accuracy_pct"]:6.2f}% '
                  f'| val {val_metrics["accuracy_pct"]:6.2f}% | loss {val_metrics["loss"]:.4f} '
                  f'| {elapsed:.1f}s' + (' | BEST saved' if improved else ''), flush=True)

    # Evaluate the selected best model, not the weights from the final epoch.
    best_weights = torch.load(out / 'best_weights.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best_weights)
    model.eval()
    test_data = datasets.CIFAR10(args.data, train=False, download=True, transform=eval_transform)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)
    test_metrics = run_epoch(model, test_loader, device)
    write_json(out / 'results.json', {
        'best_epoch': best_epoch, 'best_val_accuracy_pct': best_acc,
        'test_count': len(test_data), 'test_loss': test_metrics['loss'],
        'test_accuracy_pct': test_metrics['accuracy_pct'],
    })
    # CPU reference inputs/logits to debug the subsequent HLS/quantization stages.
    model.cpu().eval()
    reference_inputs = torch.stack([test_data[i][0] for i in range(16)])
    with torch.no_grad():
        reference_logits = model(reference_inputs).numpy()
    np.savez(out / 'golden_samples.npz',
             raw_rgb_uint8_nhwc=test_data.data[:16],
             input_fp32_nchw=reference_inputs.numpy(),
             labels=np.asarray(test_data.targets[:16], dtype=np.int64),
             logits_fp32=reference_logits,
             predictions=reference_logits.argmax(axis=1))
    print(f'Best epoch: {best_epoch} | Test accuracy: {test_metrics["accuracy_pct"]:.2f}%', flush=True)
    print(f'Saved model and FP32 export in: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
