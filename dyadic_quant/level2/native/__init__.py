from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

_NATIVE_MODULE: Any = None
_BUILD_DIR = Path(__file__).resolve().parent / "_build"


class _LazyNative:
    def __init__(self, name: str) -> None:
        self._name = name

    def _resolve(self) -> Any:
        if _NATIVE_MODULE is None:
            raise RuntimeError(
                f"dyadic_quant.level2.native.{self._name} is not available. "
                "Call build_native_cpu() first to compile the native CPU extension."
            )
        return getattr(_NATIVE_MODULE, self._name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)


def build_native_cpu(*, force: bool = False) -> str:
    global _NATIVE_MODULE

    if _NATIVE_MODULE is not None and not force:
        return "already loaded"

    from torch.utils.cpp_extension import load

    extension_dir = Path(__file__).resolve().parent
    source = extension_dir / "cpu_extension.cpp"

    if not source.exists():
        raise RuntimeError(
            f"native CPU extension source not found: {source}\n"
            "Expected dyadic_quant/level2/native/cpu_extension.cpp"
        )

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    extra_cflags = ["-O3", "-ffast-math"]
    if platform.machine().lower() in {"x86_64", "amd64"}:
        extra_cflags.append("-march=native")

    try:
        module = load(
            name="dyadic_quant_level2_native",
            sources=[str(source)],
            build_directory=str(_BUILD_DIR),
            extra_cflags=extra_cflags,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to compile native CPU extension:\n{exc}\n\n"
            "This requires a C++17 compiler and Python development headers. "
            "On macOS: xcode-select --install\n"
            "On Ubuntu: apt install build-essential python3-dev"
        ) from exc

    _NATIVE_MODULE = module
    return "built"


def warm_native_cpu_workers() -> None:
    if _NATIVE_MODULE is None:
        build_native_cpu()
    _NATIVE_MODULE.warm_native_cpu_workers()


dyadic_linear_packed_native_cpu = _LazyNative("dyadic_linear_packed_native_cpu")
dyadic_embedding_packed_native_cpu = _LazyNative("dyadic_embedding_packed_native_cpu")
dyadic_qwen_mlp_packed_native_cpu = _LazyNative("dyadic_qwen_mlp_packed_native_cpu")
dyadic_qwen_mlp_stack_packed_native_cpu = _LazyNative("dyadic_qwen_mlp_stack_packed_native_cpu")
dyadic_qwen_mlp_stack_plan_native_cpu = _LazyNative("dyadic_qwen_mlp_stack_plan_native_cpu")
pack_qwen_mlp_stack_native_cpu = _LazyNative("pack_qwen_mlp_stack_native_cpu")
dyadic_conv2d_packed_native_cpu = _LazyNative("dyadic_conv2d_packed_native_cpu")
pack_native_cpu_weight = _LazyNative("pack_native_cpu_weight")
native_add_relu_cpu = _LazyNative("native_add_relu_cpu")
native_add_cpu = _LazyNative("native_add_cpu")
native_relu_cpu = _LazyNative("native_relu_cpu")
native_max_pool2d_cpu = _LazyNative("native_max_pool2d_cpu")
native_adaptive_avg_pool2d_cpu = _LazyNative("native_adaptive_avg_pool2d_cpu")
