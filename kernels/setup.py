import io
import os
import re
import subprocess
from typing import List, Set
import warnings

from packaging.version import parse, Version
import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT_DIR = os.path.dirname(__file__)

# Supported NVIDIA GPU architectures.
# Blackwell FP4 tensor cores are SM100/SM110/SM120 (10.0, 11.0, 12.0);
# the NVFP4 experiment targets these in addition to the legacy QServe arches.
SUPPORTED_ARCHS = {"8.0", "8.6", "8.7", "8.9", "9.0", "10.0", "11.0", "12.0"}

# Compiler flags.
CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
# TODO(woosuk): Should we use -O3?
NVCC_FLAGS = [
    "-O2",
    "-std=c++17",
    "-DENABLE_BF16",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--threads=32",
]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_torch_arch_list() -> Set[str]:
    # TORCH_CUDA_ARCH_LIST can have one or more architectures,
    # e.g. "8.0" or "7.5,8.0,8.6+PTX". Here, the "8.6+PTX" option asks the
    # compiler to additionally include PTX code that can be runtime-compiled
    # and executed on the 8.6 or newer architectures. While the PTX code will
    # not give the best performance on the newer architectures, it provides
    # forward compatibility.
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if env_arch_list is None:
        return set()

    # List are separated by ; or space.
    torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
    if not torch_arch_list:
        return set()

    # Filter out the invalid architectures and print a warning.
    valid_archs = SUPPORTED_ARCHS.union({s + "+PTX" for s in SUPPORTED_ARCHS})
    arch_list = torch_arch_list.intersection(valid_archs)
    # If none of the specified architectures are valid, raise an error.
    if not arch_list:
        raise RuntimeError(
            "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
            f"variable ({env_arch_list}) is supported. "
            f"Supported CUDA architectures are: {valid_archs}.")
    invalid_arch_list = torch_arch_list - valid_archs
    if invalid_arch_list:
        warnings.warn(
            f"Unsupported CUDA architectures ({invalid_arch_list}) are "
            "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
            f"({env_arch_list}). Supported CUDA architectures are: "
            f"{valid_archs}.")
    return arch_list


# First, check the TORCH_CUDA_ARCH_LIST environment variable.
compute_capabilities = get_torch_arch_list()
if not compute_capabilities:
    # If TORCH_CUDA_ARCH_LIST is not defined or empty, target all available
    # GPUs on the current machine.
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 7:
            raise RuntimeError(
                "GPUs with compute capability below 7.0 are not supported.")
        compute_capabilities.add(f"{major}.{minor}")

nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if not compute_capabilities:
    # If no GPU is specified nor available, add all supported architectures
    # based on the NVCC CUDA version.
    compute_capabilities = SUPPORTED_ARCHS.copy()
    if nvcc_cuda_version < Version("11.1"):
        compute_capabilities.remove("8.6")
    if nvcc_cuda_version < Version("11.8"):
        compute_capabilities.remove("8.9")
        compute_capabilities.remove("9.0")

# Validate the NVCC CUDA version.
if nvcc_cuda_version < Version("11.0"):
    raise RuntimeError("CUDA 11.0 or higher is required to build the package.")
if nvcc_cuda_version < Version("11.1"):
    if any(cc.startswith("8.6") for cc in compute_capabilities):
        raise RuntimeError(
            "CUDA 11.1 or higher is required for compute capability 8.6.")
if nvcc_cuda_version < Version("11.8"):
    if any(cc.startswith("8.9") for cc in compute_capabilities):
        # CUDA 11.8 is required to generate the code targeting compute capability 8.9.
        # However, GPUs with compute capability 8.9 can also run the code generated by
        # the previous versions of CUDA 11 and targeting compute capability 8.0.
        # Therefore, if CUDA 11.8 is not available, we target compute capability 8.0
        # instead of 8.9.
        warnings.warn(
            "CUDA 11.8 or higher is required for compute capability 8.9. "
            "Targeting compute capability 8.0 instead.")
        compute_capabilities = set(cc for cc in compute_capabilities
                                   if not cc.startswith("8.9"))
        compute_capabilities.add("8.0+PTX")
    if any(cc.startswith("9.0") for cc in compute_capabilities):
        raise RuntimeError(
            "CUDA 11.8 or higher is required for compute capability 9.0.")
if any(cc.startswith(("10.", "12.")) for cc in compute_capabilities):
    if nvcc_cuda_version < Version("12.8"):
        raise RuntimeError(
            "CUDA 12.8 or higher is required for Blackwell compute "
            "capabilities 10.x/12.x.")
if any(cc.startswith("11.") for cc in compute_capabilities):
    if nvcc_cuda_version < Version("13.0"):
        raise RuntimeError(
            "CUDA 13.0 or higher is required for compute capability 11.x.")

# Add target compute capabilities to NVCC flags.
for capability in compute_capabilities:
    num = capability.split("+")[0].replace(".", "")
    if capability.endswith("+PTX"):
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]
    else:
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]

# Use NVCC threads to parallelize the build.
if nvcc_cuda_version >= Version("11.2"):
    num_threads = min(os.cpu_count(), 32)
    NVCC_FLAGS += ["--threads", str(num_threads)]

ext_modules = []

fused_attention_extension_fine_grained_dense = CUDAExtension(
    name="omniserve_backend.fused_attention_fine_grained_dense",
    sources=[
        "csrc/fused_attention/fused_attention_fine_grained/dense_attention/fused_attention.cpp",
        "csrc/fused_attention/fused_attention_fine_grained/dense_attention/decoderMaskedMultiheadAttention.cu",
        "csrc/fused_attention/fused_attention_fine_grained/fine_grained_common/update_kv_cache.cu",
        "csrc/fused_attention/common/input_metadata_helper.cu"
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_attention_extension_fine_grained_dense)

fused_attention_extension_fine_grained_sparse = CUDAExtension(
    name="omniserve_backend.fused_attention_fine_grained_sparse",
    sources=[
        "csrc/fused_attention/fused_attention_fine_grained/sparse_attention/fused_attention.cpp",
        "csrc/fused_attention/fused_attention_fine_grained/sparse_attention/decoderMaskedMultiheadAttention.cu",
        "csrc/fused_attention/fused_attention_fine_grained/fine_grained_common/update_kv_cache.cu",
        "csrc/fused_attention/common/input_metadata_helper.cu"
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_attention_extension_fine_grained_sparse)


fused_attention_extension_per_tensor_dense = CUDAExtension(
    name="omniserve_backend.fused_attention_per_tensor_dense",
    sources=[
        "csrc/fused_attention/fused_attention_per_tensor/dense_attention/fused_attention.cpp",
        "csrc/fused_attention/fused_attention_per_tensor/dense_attention/decoderMaskedMultiheadAttention.cu",
        "csrc/fused_attention/fused_attention_per_tensor/per_tensor_common/update_kv_cache.cu",
        "csrc/fused_attention/common/input_metadata_helper.cu"
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_attention_extension_per_tensor_dense)


fused_attention_extension_per_tensor_sparse = CUDAExtension(
    name="omniserve_backend.fused_attention_per_tensor_sparse",
    sources=[
        "csrc/fused_attention/fused_attention_per_tensor/sparse_attention/fused_attention.cpp",
        "csrc/fused_attention/fused_attention_per_tensor/sparse_attention/decoderMaskedMultiheadAttention.cu",
        "csrc/fused_attention/fused_attention_per_tensor/per_tensor_common/update_kv_cache.cu",
        "csrc/fused_attention/common/input_metadata_helper.cu"
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_attention_extension_per_tensor_sparse)

fused_attention_extension_pure_dense = CUDAExtension(
    name="omniserve_backend.fused_attention_pure_dense",
    sources=[
        "csrc/fused_attention/fused_attention_pure_dense/fused_attention.cpp",
        "csrc/fused_attention/fused_attention_pure_dense/decoderMaskedMultiheadAttention.cu",
        "csrc/fused_attention/fused_attention_pure_dense/update_kv_cache.cu",
        "csrc/fused_attention/fused_attention_pure_dense/input_metadata_helper.cu"
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_attention_extension_pure_dense)


fused_attention_ctx_pool_extension = CUDAExtension(
    name="omniserve_backend.fused_attention_ctx_pool",
    sources=[
        "csrc/fused_attention/sparse_utils/ContextPool/pybind.cpp",
        "csrc/fused_attention/sparse_utils/ContextPool/context_pool_kernel.cu",
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_attention_ctx_pool_extension)


fused_attention_selector_extension = CUDAExtension(
    name="omniserve_backend.fused_attention_selector",
    sources=[
        "csrc/fused_attention/sparse_utils/KVPageSelector/fused_kv_page_selector.cpp",
        "csrc/fused_attention/sparse_utils/KVPageSelector/KVPageSelector.cu",
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_attention_selector_extension)


qgemm_w8a8_extension = CUDAExtension(
    name="omniserve_backend.qgemm_w8a8",
    sources=[
        "csrc/qgemm/w8a8/pybind.cpp",
        "csrc/qgemm/w8a8/w8a8_gemm_cuda.cu",
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(qgemm_w8a8_extension)

qgemm_w4a8_per_chn_extension = CUDAExtension(
    name="omniserve_backend.qgemm_w4a8_per_chn",
    sources=[
        "csrc/qgemm/w4a8_per_chn/pybind.cpp",
        "csrc/qgemm/w4a8_per_chn/gemm_cuda.cu",
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(qgemm_w4a8_per_chn_extension)


qgemm_w4a8_per_group_extension = CUDAExtension(
    name="omniserve_backend.qgemm_w4a8_per_group",
    sources=[
        "csrc/qgemm/w4a8_per_group/pybind.cpp",
        "csrc/qgemm/w4a8_per_group/gemm_cuda.cu",
    ],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(qgemm_w4a8_per_group_extension)

# Fuse kernels.
fused_extension = CUDAExtension(
    name="omniserve_backend.fused_kernels",
    sources=["csrc/fused.cpp", "csrc/fused_kernels.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_extension)

# Layer normalization kernels.
layernorm_extension = CUDAExtension(
    name="omniserve_backend.layernorm_ops",
    sources=["csrc/layernorm.cpp", "csrc/layernorm_kernels.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(layernorm_extension)

# Activation kernels.
activation_extension = CUDAExtension(
    name="omniserve_backend.activation_ops",
    sources=["csrc/activation.cpp", "csrc/activation_kernels.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(activation_extension)



def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str):
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setuptools.setup(
    name="omniserve_backend",
    version="0.1.0",
    author="OmniServe Team",
    license="Apache 2.0",
    description=("A high-throughput and memory-efficient inference and "
                 "serving engine for LLMs"),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(exclude=("benchmarks", "csrc", "docs",
                                               "examples", "tests")),
    python_requires=">=3.8",
    #install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    #package_data={"tiychat_serve": ["py.typed"]},
)
