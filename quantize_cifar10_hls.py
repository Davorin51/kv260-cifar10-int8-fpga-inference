#!/usr/bin/env python3
"""Step 2: manual power-of-two INT8 PTQ for train_cifar10_hls.py.

Run AFTER training:
  python quantize_cifar10_hls.py --run-dir runs/cifar10_01
  python quantize_cifar10_hls.py --run-dir runs/cifar10_01 --eval-size 5000 --out runs/cifar10_01/int8_full
Run numerical checks without PyTorch or CIFAR-10:
  python quantize_cifar10_hls.py --self-test

Requires numpy>=1.20; torchvision (and matching torch) only to read CIFAR-10.
Uses best_weights_fp32.npz, config.json and split_indices.npz from step 1.
All calibration/evaluation arithmetic runs on the CPU in NumPy. It is a
correctness reference, not a CPU performance benchmark or an FPGA compiler.

Scheme: real_value = integer_value * 2**(-fractional_bits), zero point 0.
Weights: symmetric per-layer INT8 [-127,127]. Activations after ReLU: [0,127].
Bias/accumulator: INT32. NumPy accumulates in INT64 to detect overflow.
Rounding: nearest, ties away from zero (ReLU requantization is nonnegative).
Conv requantization: shift = input_frac + weight_frac - output_frac.
Positive shift -> rounded right shift; negative -> left shift; saturate [0,127].
MaxPool follows requantization/ReLU. GAP over 4x4: (sum + 8) >> 4.
FC returns INT32 logits with ONE shared scale, so argmax needs no dequantization.
The 256-entry RGB input LUT avoids float preprocessing on the target.

This deliberately simple scheme uses power-of-two scales for shift-only HLS
rescaling. It is not PyTorch/Vitis AI automatic quantization. Accuracy loss is
measured, not guaranteed. No pruning or QAT is performed here.

Default calibration: 1000 original training images, no augmentation.
Default evaluation: 500 validation images, same subset for FP32 and INT8.
--eval-size 5000 evaluates the full validation set. Official test set is not
used for calibration or model selection in this step.

Outputs: model_int8.npz, quantization.json, model_params.h, comparison.json,
         comparison.csv, golden_int8.npz, and a copy of this script.
Output directory must be new/empty. Header contains actual learned constants
only AFTER you run this script on your trained weights. No fake trained model
is supplied. HLS kernels and a KV260 bitstream remain separate next steps.

References:
https://arxiv.org/abs/1712.05877 (integer-only inference principles)
https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

LAYERS = ('conv1', 'conv2', 'conv3', 'fc')
SHAPES = {'conv1': (16, 3, 3, 3), 'conv2': (32, 16, 3, 3),
          'conv3': (64, 32, 3, 3), 'fc': (10, 64)}
I32_MAX = 2**31 - 1


def round_away(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def fractional_bits(max_abs):
    if not np.isfinite(max_abs) or max_abs < 0:
        raise ValueError('Invalid range in weights/activations.')
    if max_abs == 0:
        return 0
    return max(-16, min(24, math.floor(math.log2(127.0 / max_abs))))


def conv_same(x, w, bias):
    """NCHW/OIHW cross-correlation, stride=1, padding=1, 3x3 kernel."""
    integer = np.issubdtype(x.dtype, np.integer)
    dtype = np.int64 if integer else np.float32
    n, _, h, width = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
    # N,C,H,W,KH,KW -> N,H,W,C,KH,KW; flatten in OIHW-compatible order.
    columns = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n*h*width, -1).astype(dtype)
    output = columns @ w.reshape(w.shape[0], -1).astype(dtype).T
    output += bias.astype(dtype)
    return output.reshape(n, h, width, w.shape[0]).transpose(0, 3, 1, 2)


def pool2(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h//2, 2, w//2, 2).max(axis=(3, 5))


def fp32_forward(raw_rgb, weights):
    x = raw_rgb.transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255.0)
    x = (x - np.float32(0.5)) / np.float32(0.5)
    activations = []
    for name in LAYERS[:3]:
        x = pool2(np.maximum(conv_same(x, weights[name+'.weight'], weights[name+'.bias']), 0))
        activations.append(x)
    x = x.mean(axis=(2, 3), dtype=np.float32)
    logits = x @ weights['fc.weight'].T + weights['fc.bias']
    return logits, activations


def check_int32(x):
    if np.max(x) > I32_MAX or np.min(x) < -2**31:
        raise OverflowError('Accumulator does not fit INT32.')


def requant_relu(acc, shift):
    if not -31 <= shift <= 62:
        raise ValueError('Unsupported shift; expected -31..62.')
    check_int32(acc)
    x = np.maximum(acc.astype(np.int64), 0)
    if shift > 0:
        x = (x + (1 << (shift-1))) >> shift
    elif shift < 0:
        x = x << (-shift)
    return np.clip(x, 0, 127).astype(np.int8)


def make_quantized(weights, activation_maxima):
    input_frac = 6  # Entire normalized RGB range [-1,1] fits without clipping.
    output_fracs = [fractional_bits(float(v)) for v in activation_maxima]
    # Exact integer-only LUT: round-away((2*pixel-255)*64/255).
    numerator = (2*np.arange(256, dtype=np.int64)-255) * (1 << input_frac)
    lut = (np.sign(numerator) * ((np.abs(numerator)+127)//255)).astype(np.int8)
    arrays = {'input_lut': lut}
    metadata = {'scheme': 'symmetric_power_of_two_per_layer_v1',
                'input_frac': input_frac, 'activation_fracs': output_fracs,
                'zero_point': 0, 'rounding': 'nearest_ties_away_from_zero',
                'weight_range': [-127, 127], 'relu_range': [0, 127],
                'conv_layout': 'OIHW_flat_C_order', 'fc_layout': 'OI_flat_C_order',
                'input_layout': 'NCHW', 'raw_layout': 'NHWC_RGB_uint8',
                'gap_rule': '(sum_of_16_nonnegative_int8_values + 8) >> 4',
                'layers': {}}
    input_fraction = input_frac
    for i, name in enumerate(LAYERS):
        w, b = weights[name+'.weight'], weights[name+'.bias']
        wf = fractional_bits(float(np.max(np.abs(w))))
        qw = np.clip(round_away(w.astype(np.float64)*2.0**wf), -127, 127).astype(np.int8)
        qb64 = round_away(b.astype(np.float64)*2.0**(input_fraction+wf))
        if not np.isfinite(qb64).all() or np.max(np.abs(qb64)) > I32_MAX:
            raise OverflowError(f'{name}: bias cannot be represented safely in INT32.')
        qb = qb64.astype(np.int32)
        # Conservative bound also protects every partial sum in an INT32 HLS MAC.
        bound = np.abs(qb.astype(np.int64)) + 127*np.abs(qw.astype(np.int64)).reshape(qw.shape[0], -1).sum(1)
        if np.max(bound) > I32_MAX:
            raise OverflowError(f'{name}: conservative INT32 accumulator bound exceeded.')
        arrays[name+'_weight'] = qw
        arrays[name+'_bias'] = qb
        entry = {'shape': list(w.shape), 'input_frac': input_fraction, 'weight_frac': wf,
                 'accumulator_frac': input_fraction+wf, 'accumulator_abs_bound': int(np.max(bound))}
        if name != 'fc':
            shift = input_fraction + wf - output_fracs[i]
            if not -31 <= shift <= 62:
                raise ValueError(f'{name}: unsupported requantization shift {shift}.')
            entry.update(output_frac=output_fracs[i], shift=shift)
            input_fraction = output_fracs[i]
        metadata['layers'][name] = entry
    metadata['logits_frac'] = metadata['layers']['fc']['accumulator_frac']
    return arrays, metadata


def int8_forward(raw_rgb, arrays, metadata):
    x = arrays['input_lut'][raw_rgb].transpose(0, 3, 1, 2)
    traces = {'input_int8': x.copy()}
    for name in LAYERS[:3]:
        acc = conv_same(x, arrays[name+'_weight'], arrays[name+'_bias'])
        x = pool2(requant_relu(acc, metadata['layers'][name]['shift']))
        traces[name+'_pooled_int8'] = x.copy()
    x = ((x.astype(np.int64).sum(axis=(2, 3)) + 8) >> 4).astype(np.int8)
    traces['gap_int8'] = x.copy()
    logits = x.astype(np.int64) @ arrays['fc_weight'].astype(np.int64).T + arrays['fc_bias']
    check_int32(logits)
    traces['logits_int32'] = logits.astype(np.int32)
    return traces['logits_int32'], traces


def write_header(path, arrays, metadata):
    with path.open('w', encoding='utf-8') as f:
        f.write('// Generated from trained weights. Layout and arithmetic: quantization.json\n')
        f.write('#ifndef CIFAR10_MODEL_PARAMS_H\n#define CIFAR10_MODEL_PARAMS_H\n#include <stdint.h>\n')
        f.write(f'static const int INPUT_FRAC = {metadata["input_frac"]};\n')
        f.write(f'static const int LOGITS_FRAC = {metadata["logits_frac"]};\n')
        for name in LAYERS:
            layer = metadata['layers'][name]
            for key in ('input_frac', 'weight_frac', 'output_frac', 'shift'):
                if key in layer:
                    f.write(f'static const int {name.upper()}_{key.upper()} = {layer[key]};\n')
        for name, values in arrays.items():
            dtype = 'int8_t' if values.dtype == np.int8 else 'int32_t'
            flat = values.reshape(-1)
            f.write(f'// Shape {list(values.shape)}, flattened in C order.\n')
            f.write(f'static const {dtype} {name.upper()}[{flat.size}] = {{\n')
            for offset in range(0, flat.size, 24):
                f.write('  '+', '.join(str(int(v)) for v in flat[offset:offset+24])+',\n')
            f.write('};\n')
        f.write('''
// Valid for this export: shift in [-31,62], acc in INT32, ReLU output.
static inline int8_t cnn_requant_relu(int32_t acc, int shift) {
    if (acc <= 0) return 0;
    int64_t x = (int64_t)acc;
    if (shift > 0) x = (x + ((int64_t)1 << (shift - 1))) >> shift;
    else if (shift < 0) x <<= -shift;
    return (int8_t)(x > 127 ? 127 : x);
}
#endif
''')


def self_test():
    rng = np.random.default_rng(12)
    x = rng.integers(-8, 9, (1, 2, 4, 4), dtype=np.int8)
    w = rng.integers(-8, 9, (3, 2, 3, 3), dtype=np.int8)
    b = np.array([1, -2, 3], dtype=np.int32)
    padded = np.pad(x, ((0,0),(0,0),(1,1),(1,1)))
    expected = np.zeros((1,3,4,4), dtype=np.int64)
    for oc in range(3):
        for y in range(4):
            for xx in range(4):
                total = int(b[oc])
                for ic in range(2):
                    for ky in range(3):
                        for kx in range(3):
                            total += int(padded[0,ic,y+ky,xx+kx])*int(w[oc,ic,ky,kx])
                expected[0,oc,y,xx] = total
    np.testing.assert_array_equal(conv_same(x,w,b), expected)
    np.testing.assert_allclose(conv_same(x.astype(np.float32),w.astype(np.float32),b), expected, atol=0, rtol=0)
    np.testing.assert_array_equal(round_away([-1.5,-0.5,0.5,1.5]), [-2,-1,1,2])
    np.testing.assert_array_equal(requant_relu(np.array([-4,0,1,3,255,1000]),1), [0,0,1,2,127,127])
    np.testing.assert_array_equal(requant_relu(np.array([0,1,I32_MAX]),-31), [0,127,127])
    np.testing.assert_array_equal(pool2(np.arange(16).reshape(1,1,4,4)), np.array([[[[5,7],[13,15]]]]))
    weights = {}
    for name, shape in SHAPES.items():
        weights[name+'.weight'] = rng.normal(0,0.05,shape).astype(np.float32)
        weights[name+'.bias'] = np.zeros(shape[0],dtype=np.float32)
    raw = rng.integers(0,256,(2,32,32,3),dtype=np.uint8)
    fp, acts = fp32_forward(raw,weights)
    arrays, meta = make_quantized(weights,[float(a.max()) for a in acts])
    lut_expected = round_away(((np.arange(256)*2.0/255.0)-1)*64).astype(np.int8)
    np.testing.assert_array_equal(arrays['input_lut'],lut_expected)
    iq, traces = int8_forward(raw,arrays,meta)
    assert fp.shape == iq.shape == (2,10)
    assert iq.dtype == np.int32 and traces['gap_int8'].shape == (2,64)
    assert (np.abs(iq.astype(np.int64)) <= meta['layers']['fc']['accumulator_abs_bound']).all()
    print('PASS: convolution vs scalar reference, FP32 convolution, pooling, rounding, shifts, input LUT, full INT8 graph.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', default='runs/cifar10_01')
    parser.add_argument('--data', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--calib-size', type=int, default=1000)
    parser.add_argument('--eval-size', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not 1 <= args.calib_size <= 45000 or not 1 <= args.eval_size <= 5000 or args.batch_size < 1:
        parser.error('calib-size: 1..45000; eval-size: 1..5000; batch-size must be positive.')
    if args.seed < 0:
        parser.error('seed must be nonnegative.')
    run = Path(args.run_dir)
    config = json.loads((run/'config.json').read_text(encoding='utf-8'))
    if (config['architecture'] != 'SmallCifarCNN_v1' or config['input_shape'] != [1,3,32,32]
            or config['mean'] != [0.5]*3 or config['std'] != [0.5]*3
            or config['pixel_divisor'] != 255.0 or config['color_order'] != 'RGB'):
        raise ValueError('This quantizer requires the exact architecture/preprocessing from step 1.')
    with np.load(run/'best_weights_fp32.npz', allow_pickle=False) as data:
        weights = {k: data[k].astype(np.float32) for k in data.files}
    for name, shape in SHAPES.items():
        assert weights[name+'.weight'].shape == shape
        assert weights[name+'.bias'].shape == (shape[0],)
        if not np.isfinite(weights[name+'.weight']).all() or not np.isfinite(weights[name+'.bias']).all():
            raise ValueError('Non-finite model parameters.')
    with np.load(run/'split_indices.npz', allow_pickle=False) as split:
        train_idx, val_idx = split['train'].copy(), split['val'].copy()
    if (len(train_idx) != 45000 or len(val_idx) != 5000
            or len(np.unique(np.concatenate([train_idx,val_idx]))) != 50000
            or min(train_idx.min(),val_idx.min()) != 0 or max(train_idx.max(),val_idx.max()) != 49999):
        raise ValueError('Invalid train/validation split.')
    rng = np.random.default_rng(args.seed)
    calib_idx = rng.choice(train_idx,args.calib_size,replace=False)
    eval_idx = rng.choice(val_idx,args.eval_size,replace=False)
    out = Path(args.out) if args.out else run/'int8'
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f'Output must be new/empty: {out}')

    # Only the training portion of CIFAR-10 is read; test remains untouched.
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(args.data or config['arguments']['data'], train=True, download=True)
    raw, labels = dataset.data, np.asarray(dataset.targets)
    golden_file = run/'golden_samples.npz'
    if golden_file.exists():
        with np.load(golden_file, allow_pickle=False) as golden:
            np_logits, _ = fp32_forward(golden['raw_rgb_uint8_nhwc'],weights)
            # Verifies NumPy graph and exported weights against step 1 PyTorch.
            np.testing.assert_allclose(np_logits,golden['logits_fp32'],rtol=1e-4,atol=1e-4,
                                       err_msg='NumPy/PyTorch golden logits disagree.')
        print('NumPy FP32 graph matches saved PyTorch golden logits.',flush=True)
    maxima = np.zeros(3,dtype=np.float64)
    for start in range(0,len(calib_idx),args.batch_size):
        _, acts = fp32_forward(raw[calib_idx[start:start+args.batch_size]],weights)
        maxima = np.maximum(maxima,[float(a.max()) for a in acts])
        if start == 0 or (start//args.batch_size)%20 == 0:
            print(f'Calibration {min(start+args.batch_size,len(calib_idx))}/{len(calib_idx)}',flush=True)
    arrays, meta = make_quantized(weights,maxima)
    meta.update(activation_calibration_maxima=maxima.tolist(),classes=config['classes'],
                source_weights_sha256=hashlib.sha256((run/'best_weights_fp32.npz').read_bytes()).hexdigest(),
                calibration_indices=calib_idx.tolist(),evaluation_indices=eval_idx.tolist(),
                calibration_split='train',evaluation_split='validation',seed=args.seed)
    out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'model_int8.npz',**arrays)
    (out/'quantization.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    write_header(out/'model_params.h',arrays,meta)
    source = Path(__file__).resolve()
    if source != (out/source.name).resolve():
        shutil.copy2(source,out/source.name)
    fp_predictions, iq_predictions = [], []
    for start in range(0,len(eval_idx),args.batch_size):
        ids = eval_idx[start:start+args.batch_size]
        fp_logits,_ = fp32_forward(raw[ids],weights)
        iq_logits,traces = int8_forward(raw[ids],arrays,meta)
        fp_predictions.extend(fp_logits.argmax(1).tolist())
        iq_predictions.extend(iq_logits.argmax(1).tolist())
        if start == 0:
            np.savez(out/'golden_int8.npz',raw_rgb_uint8_nhwc=raw[ids],indices=ids,
                     labels=labels[ids],logits_fp32=fp_logits,**traces)
        print(f'Evaluation {min(start+args.batch_size,len(eval_idx))}/{len(eval_idx)}',flush=True)
    fp_predictions,iq_predictions = np.asarray(fp_predictions),np.asarray(iq_predictions)
    fp_acc = float(100*np.mean(fp_predictions == labels[eval_idx]))
    iq_acc = float(100*np.mean(iq_predictions == labels[eval_idx]))
    result = {'split':'validation','samples':len(eval_idx),'fp32_accuracy_pct':fp_acc,
              'int8_accuracy_pct':iq_acc,'accuracy_drop_percentage_points':fp_acc-iq_acc,
              'prediction_agreement_pct':float(100*np.mean(fp_predictions==iq_predictions)),
              'scheme':meta['scheme'],'hardware_tested':False}
    (out/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['training_set_index','label','fp32_prediction','int8_prediction'])
        writer.writerows(zip(eval_idx,labels[eval_idx],fp_predictions,iq_predictions))
    print(json.dumps(result,indent=2))
    print(f'Saved INT8 model, C header and reference vectors: {out.resolve()}')


if __name__ == '__main__':
    main()
