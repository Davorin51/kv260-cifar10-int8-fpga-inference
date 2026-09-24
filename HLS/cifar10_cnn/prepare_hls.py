#!/usr/bin/env python3
"""Prepare HLS sources using YOUR real INT8 export (NumPy only).

Run from the component directory, after extracting this package there:
python prepare_hls.py --quant-dir ../../runs/cifar10_01/int8_full

Creates model_params.h, test_vectors.h and hls_config.ready.cfg beside cnn.cpp.
The existing hls_config.cfg is left for you to update in Vitis.
Default 2 samples plus a replay in the testbench = 3 accelerator calls.
--samples 16 uses up to 16 available golden samples (longer RTL simulation).
Prepared files can be regenerated; validation runs before files are written.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np


def validate_export(header, arrays, meta):
    shapes = {'input_lut': (256,), 'conv1_weight': (16,3,3,3), 'conv1_bias': (16,),
              'conv2_weight': (32,16,3,3), 'conv2_bias': (32,),
              'conv3_weight': (64,32,3,3), 'conv3_bias': (64,),
              'fc_weight': (10,64), 'fc_bias': (10,)}
    if (meta.get('scheme') != 'symmetric_power_of_two_per_layer_v1'
            or meta.get('rounding') != 'nearest_ties_away_from_zero'
            or meta.get('zero_point') != 0 or meta.get('input_frac') != 6):
        raise ValueError('Expected the exact power-of-two export from step 2.')
    if set(arrays) != set(shapes):
        raise ValueError('Unexpected quantized array names.')
    for name, shape in shapes.items():
        dtype = np.int32 if name.endswith('_bias') else np.int8
        a = arrays[name]
        if a.shape != shape or a.dtype != dtype:
            raise ValueError(f'{name}: expected {shape}, {dtype}; got {a.shape}, {a.dtype}')
        ctype = 'int32_t' if dtype == np.int32 else 'int8_t'
        match = re.search(r'static\s+const\s+'+ctype+r'\s+'+name.upper()
                          +r'\s*\[\s*(\d+)\s*\]\s*=\s*\{(.*?)\};',header,re.S)
        if not match or int(match.group(1)) != a.size:
            raise ValueError(f'Missing/wrong C array {name.upper()} in header.')
        values = np.asarray([int(v.strip()) for v in match.group(2).split(',') if v.strip()],dtype=np.int64)
        if not np.array_equal(values,a.reshape(-1)):
            raise ValueError(f'Header and model_int8.npz disagree: {name}')
    def constant(name, value):
        match = re.search(r'static\s+const\s+int\s+'+name+r'\s*=\s*(-?\d+)\s*;',header)
        if not match or int(match.group(1)) != value:
            raise ValueError(f'Header and quantization.json disagree: {name}')
    constant('INPUT_FRAC',meta['input_frac'])
    constant('LOGITS_FRAC',meta['logits_frac'])
    numerator=(2*np.arange(256,dtype=np.int64)-255)*64
    lut=(np.sign(numerator)*((np.abs(numerator)+127)//255)).astype(np.int8)
    if not np.array_equal(arrays['input_lut'],lut):
        raise ValueError('Unexpected RGB input LUT.')
    previous = meta['input_frac']
    for name in ('conv1','conv2','conv3','fc'):
        layer = meta['layers'][name]
        if layer['input_frac'] != previous or layer['accumulator_frac'] != previous+layer['weight_frac']:
            raise ValueError(f'Inconsistent fractional bits in {name}.')
        for key in ('input_frac','weight_frac','output_frac','shift'):
            if key in layer:
                constant(name.upper()+'_'+key.upper(),layer[key])
        if name != 'fc':
            shift=previous+layer['weight_frac']-layer['output_frac']
            if shift != layer['shift'] or not -31 <= shift <= 62:
                raise ValueError(f'Invalid shift in {name}.')
            previous=layer['output_frac']
        w=arrays[name+'_weight'].astype(np.int64)
        b=arrays[name+'_bias'].astype(np.int64)
        if np.min(w) < -127:
            raise ValueError('Expected symmetric weights in [-127,127].')
        bound=np.abs(b)+127*np.abs(w).reshape(w.shape[0],-1).sum(axis=1)
        if bound.max() > 2**31-1:
            raise OverflowError(f'{name} could overflow an INT32 accumulator.')
    if meta['logits_frac'] != meta['layers']['fc']['accumulator_frac']:
        raise ValueError('Inconsistent logits scale.')
    if 'cnn_requant_relu' not in header:
        raise ValueError('Missing integer requantization helper.')


def write_vectors(path, raw, logits, labels):
    n=len(raw)
    with path.open('w',encoding='utf-8') as f:
        f.write('// Real validation samples exported by prepare_hls.py; testbench only.\n')
        f.write('#ifndef CIFAR10_TEST_VECTORS_H\n#define CIFAR10_TEST_VECTORS_H\n#include <stdint.h>\n')
        f.write(f'static const int NUM_TEST_SAMPLES = {n};\n')
        for ctype,name,rows in [('uint8_t','TEST_IMAGES',raw.reshape(n,3072)),
                                ('int32_t','EXPECTED_LOGITS',logits)]:
            f.write(f'static const {ctype} {name}[{n}][{rows.shape[1]}] = {{\n')
            for row in rows:
                f.write(' {\n')
                for start in range(0,len(row),24):
                    f.write('  '+', '.join(str(int(v)) for v in row[start:start+24])+',\n')
                f.write(' },\n')
            f.write('};\n')
        f.write(f'static const uint8_t TEST_LABELS[{n}] = {{'+','.join(str(int(v)) for v in labels)+'};\n')
        f.write('#endif\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quant-dir',required=True,type=Path)
    parser.add_argument('--component-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--samples',type=int,default=2)
    args=parser.parse_args()
    if args.samples < 1:
        parser.error('samples must be positive.')
    target=args.component_dir.resolve()
    for name in ('cnn.cpp','cnn.h','tb_cnn.cpp'):
        if not (target/name).is_file():
            parser.error(f'Missing source {target/name}; extract the package into the component directory.')
    src=args.quant_dir.resolve()
    header=(src/'model_params.h').read_text(encoding='utf-8')
    meta=json.loads((src/'quantization.json').read_text(encoding='utf-8'))
    with np.load(src/'model_int8.npz',allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    validate_export(header,arrays,meta)
    with np.load(src/'golden_int8.npz',allow_pickle=False) as data:
        raw=data['raw_rgb_uint8_nhwc']
        logits=data['logits_int32']
        labels=data['labels']
        if (raw.ndim != 4 or raw.shape[1:] != (32,32,3) or raw.dtype != np.uint8
                or logits.shape != (len(raw),10) or logits.dtype != np.int32
                or labels.shape != (len(raw),) or not np.issubdtype(labels.dtype,np.integer)
                or len(raw) < 1 or np.any(labels < 0) or np.any(labels > 9)):
            raise ValueError('Invalid golden sample shapes/dtypes/labels.')
        n=min(args.samples,len(raw))
        raw,logits,labels=raw[:n].copy(),logits[:n].copy(),labels[:n].copy()
    (target/'model_params.h').write_text(header,encoding='utf-8')
    write_vectors(target/'test_vectors.h',raw,logits,labels)
    lines=['part=xck26-sfvc784-2LV-c','','[hls]','flow_target=vivado',
           'clock=10ns','syn.top=cifar10_cnn','syn.cflags=-std=c++14',
           'tb.cflags=-std=c++14']
    lines.extend('syn.file='+(target/name).as_posix() for name in ('cnn.cpp','cnn.h','model_params.h'))
    lines.extend('tb.file='+(target/name).as_posix() for name in ('tb_cnn.cpp','test_vectors.h'))
    lines.extend(['package.output.format=ip_catalog','package.output.syn=false',''])
    (target/'hls_config.ready.cfg').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Validated INT8 export: {src}')
    print(f'Prepared {n} real samples + replay; component: {target}')
    print('Generated model_params.h, test_vectors.h, hls_config.ready.cfg.')
    print('Copy hls_config.ready.cfg into your existing hls_config.cfg, then run C SIMULATION.')


if __name__ == '__main__':
    main()
