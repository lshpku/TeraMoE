import os
import subprocess
import setuptools
import importlib

from pathlib import Path
from paddle.utils.cpp_extension import BuildExtension, CUDAExtension
from paddle.utils.cpp_extension.extension_utils import add_compile_flag


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')


def _detect_local_gpu_arch():
    '''Auto-detect the compute capability of the first visible GPU via nvidia-smi.

    Returns a string like '10.3', or None if detection fails.
    '''
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL,
        )
        caps = {line.strip() for line in out.decode().splitlines() if line.strip()}
        return sorted(caps)[0] if caps else None
    except Exception:
        return None


def _resolve_arch_gencode_flags():
    '''Resolve the -gencode flags for this build.

    TeraMoE's compute kernels use `tcgen05.*` PTX, which requires the
    architecture-specific (`a`-suffixed) Blackwell targets. Paddle's
    `_get_cuda_arch_flags()` does not know about `10.3a`, so the flags are
    built here and passed to nvcc directly. As long as `PADDLE_CUDA_ARCH_LIST`
    is left unset, Paddle sees an `arch` flag in our nvcc args and refrains
    from appending its own (see `extension_utils._get_cuda_arch_flags`).

    Priority: TERAMOE_CUDA_ARCH env var > auto-detect > 10.3a
    '''
    import re

    raw = os.environ.get('TERAMOE_CUDA_ARCH', '').strip()
    if not raw:
        raw = _detect_local_gpu_arch() or ''

    archs = [a.strip() for a in re.split(r'[;,]', raw) if a.strip()]
    if not archs:
        archs = ['10.3']

    flags = []
    for arch in archs:
        major, minor = arch.rstrip('a').split('.')
        if int(major) < 10:
            raise ValueError(
                f'TeraMoE requires a Blackwell GPU (SM100+), but got arch {arch}'
            )
        num = f'{major}{minor}'
        # Always use the arch-specific target: tcgen05 is not available otherwise.
        flags.append(f'-gencode=arch=compute_{num}a,code=sm_{num}a')
    return sorted(set(flags))


if __name__ == '__main__':
    # Paddle appends its own -gencode flags from PADDLE_CUDA_ARCH_LIST; we supply
    # arch-specific (`a`-suffixed) flags ourselves, so the env var must stay unset.
    os.environ.pop('PADDLE_CUDA_ARCH_LIST', None)

    disable_nvshmem = False
    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    nvshmem_host_lib = 'libnvshmem_host.so'
    if nvshmem_dir is None:
        try:
            nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
            nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
            import nvidia.nvshmem as nvshmem  # noqa: F401
        except (ModuleNotFoundError, AttributeError, IndexError):
            print(
                'Warning: `NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. All internode features are disabled\n'
            )
            disable_nvshmem = True
    else:
        disable_nvshmem = False

    if not disable_nvshmem:
        assert os.path.exists(nvshmem_dir), f'The specified NVSHMEM directory does not exist: {nvshmem_dir}'

    _repo_root = os.path.dirname(os.path.abspath(__file__))
    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    gencode_flags = _resolve_arch_gencode_flags()
    nvcc_flags = ['-O3', '-Xcompiler', '-O3'] + gencode_flags
    sources = ['csrc/moe_extension.cpp', 'csrc/kernels/runtime.cu', 'csrc/kernels/layout.cu', 'csrc/kernels/intranode.cu', 'csrc/teramoe/teramoe_orchestrator.cu']
    include_dirs = [os.path.join(_repo_root, 'csrc')]
    _third_party_root = os.path.join(_repo_root, 'third-party')

    _cutlass_root = os.path.join(_third_party_root, 'cutlass')
    if os.path.isdir(os.path.join(_cutlass_root, 'include')):
        include_dirs.append(os.path.join(_cutlass_root, 'include'))
        include_dirs.append(os.path.join(_cutlass_root, 'tools', 'util', 'include'))

    _deepgemm_root = os.path.join(_third_party_root, 'DeepGEMM')
    _deepgemm_include = os.path.join(_deepgemm_root, 'deep_gemm', 'include')
    _deepgemm_cutlass_include = os.path.join(_deepgemm_root, 'third-party', 'cutlass', 'include')
    if os.path.isdir(_deepgemm_include):
        include_dirs.append(_deepgemm_include)
    if os.path.isdir(_deepgemm_cutlass_include):
        include_dirs.append(_deepgemm_cutlass_include)

    _quack_root = os.path.join(_third_party_root, 'quack')
    if os.path.isdir(os.path.join(_quack_root, 'quack')):
        include_dirs.append(_quack_root)
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = ['-lcuda']

    # NVSHMEM flags
    if disable_nvshmem:
        cxx_flags.append('-DDISABLE_NVSHMEM')
        nvcc_flags.append('-DDISABLE_NVSHMEM')
    else:
        sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu', 'csrc/teramoe/teramoe_notify.cu'])
        include_dirs.extend([f'{nvshmem_dir}/include'])
        # CCCL (libcudacxx) headers needed by nvshmem_tensor.h for cuda/std/tuple
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        cccl_include = os.path.join(cuda_home, 'include', 'cccl')
        if os.path.isdir(cccl_include):
            include_dirs.append(cccl_include)
        library_dirs.extend([f'{nvshmem_dir}/lib'])
        nvcc_dlink.extend(['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem_device'])
        extra_link_args.extend([f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_dir}/lib'])

    # CUDA 12 flags
    nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
    cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
    nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if "TOPK_IDX_BITS" in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
        nvcc_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    mk_compute_kernel = int(os.getenv('MK_COMPUTE_KERNEL', '1'))
    assert mk_compute_kernel in (1, 2), 'MK_COMPUTE_KERNEL must be 1 or 2 (WMMA path removed)'
    cxx_flags.append(f'-DMK_COMPUTE_KERNEL={mk_compute_kernel}')
    nvcc_flags.append(f'-DMK_COMPUTE_KERNEL={mk_compute_kernel}')

    if int(os.getenv('ENABLE_FAST_DEBUG', 0)):
        cxx_flags.append('-DENABLE_FAST_DEBUG')
        nvcc_flags.append('-DENABLE_FAST_DEBUG')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink + gencode_flags

    # Paddle-specific build macros (mirrors third_party/DeepEP/setup.py)
    add_compile_flag(extra_compile_args, ['-DPADDLE_WITH_CUDA'])
    add_compile_flag(extra_compile_args, ['-DWITH_DISTRIBUTE'])
    add_compile_flag(extra_compile_args, ['-DWITH_NVSHMEM'])
    add_compile_flag(extra_compile_args, ['-DWITH_GPU'])
    add_compile_flag(extra_compile_args, ['-DWITH_FLUID_ONLY'])

    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Gencode flags: {gencode_flags}')
    print(f' > NVSHMEM path: {nvshmem_dir}')
    print()

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(name='teramoe',
                     version='0.0.1' + revision,
                     packages=setuptools.find_packages(include=['teramoe']),
                     ext_modules=[
                         CUDAExtension(name='teramoe_cpp',
                                       include_dirs=include_dirs,
                                       library_dirs=library_dirs,
                                       sources=sources,
                                       extra_compile_args=extra_compile_args,
                                       extra_link_args=extra_link_args)
                     ],
                     cmdclass={'build_ext': BuildExtension})
