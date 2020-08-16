#!/usr/bin/env python3
import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++14']

nvcc_args = [
    # '-gencode', 'arch=compute_50,code=sm_50',
    # '-gencode', 'arch=compute_52,code=sm_52',
    # '-gencode', 'arch=compute_60,code=sm_60',
    # '-gencode', 'arch=compute_61,code=sm_61',
    # '-gencode', 'arch=compute_61,code=compute_61',
    # '-gencode', 'arch=compute_70,code=sm_70',
    # '-gencode', 'arch=compute_70,code=compute_70',
    '-gencode', 'arch=compute_75,code=sm_75',
    # '-gencode', 'arch=compute_75,code=compute_75'
]

#     '-gencode', 'arch=compute_75,code=sm_75',
#     '-gencode', 'arch=compute_75,code=compute_75'
# , extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}
setup(
    name='block_extractor_cuda',
    ext_modules=[
        CUDAExtension('block_extractor_cuda', [
            'block_extractor_cuda.cc',
            'block_extractor_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

