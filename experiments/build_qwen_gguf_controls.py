from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from gguf import GGMLQuantizationType, GGUFReader


EXPECTED_SOURCE_SHA256 = (
    "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
)
SOURCE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
VARIANTS = {
    "q4_k_m": ("Q4_K_M", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"),
    "q6_k": ("Q6_K", "Qwen2.5-0.5B-Instruct-Q6_K.gguf"),
    "q8_0": ("Q8_0", "Qwen2.5-0.5B-Instruct-Q8_0.gguf"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> str:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout


def tensor_types(path: Path) -> dict[str, int]:
    reader = GGUFReader(str(path))
    counts = Counter(
        GGMLQuantizationType(int(tensor.tensor_type)).name
        for tensor in reader.tensors
    )
    return dict(sorted(counts.items()))


def artifact(path: Path) -> dict[str, object]:
    types = tensor_types(path)
    return {
        "file": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "tensor_count": sum(types.values()),
        "tensor_types": types,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument("--template-model", default="qwen2.5:0.5b")
    parser.add_argument("--model-prefix", default="qwen25-original")
    parser.add_argument("--output-dir", type=Path, default=Path("results/level1"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-ollama", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    source_file = source_dir / "model.safetensors"
    if sha256(source_file) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("the source Safetensors file does not match the audited hash")

    llama_cpp = args.llama_cpp_dir.resolve()
    converter = llama_cpp / "convert_hf_to_gguf.py"
    quantizer = llama_cpp / "build/bin/llama-quantize"
    if not converter.is_file() or not quantizer.is_file():
        raise RuntimeError("llama.cpp converter or quantizer is missing")
    llama_commit = run(["git", "-C", str(llama_cpp), "rev-parse", "HEAD"]).strip()

    checkpoint_dir = source_dir.parent
    bf16_gguf = checkpoint_dir / "Qwen2.5-0.5B-Instruct-BF16.gguf"
    conversion_command = [
        sys.executable,
        str(converter),
        str(source_dir),
        "--outfile",
        str(bf16_gguf),
        "--outtype",
        "bf16",
    ]
    if not bf16_gguf.is_file():
        run(conversion_command)

    lineage: dict[str, object] = {
        "source": {
            "repository": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": SOURCE_REVISION,
            "file": str(source_file.resolve()),
            "size_bytes": source_file.stat().st_size,
            "sha256": sha256(source_file),
            "tensor_dtype": "bfloat16",
        },
        "llama_cpp": {
            "directory": str(llama_cpp),
            "commit": llama_commit,
        },
        "bf16_gguf": artifact(bf16_gguf)
        | {
            "conversion_command": conversion_command,
            "lineage_note": (
                "Converted directly from the audited BF16 Safetensors checkpoint. "
                "BF16 values stored as F32 metadata tensors are exact value-preserving "
                "expansions, not a separately trained or quantized source."
            ),
        },
        "variants": {},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = ""
    if not args.skip_ollama:
        template = run(["ollama", "show", args.template_model, "--modelfile"])
        source_model_name = f"{args.model_prefix}-bf16"
        source_modelfile = re.sub(
            r"^FROM\s+.+$",
            f"FROM {bf16_gguf.resolve()}",
            template,
            count=1,
            flags=re.MULTILINE,
        )
        source_modelfile_path = args.output_dir / f"{source_model_name}.Modelfile"
        source_modelfile_path.write_text(source_modelfile)
        source_create_command = [
            "ollama",
            "create",
            source_model_name,
            "--file",
            str(source_modelfile_path),
        ]
        run(source_create_command)
        lineage["bf16_gguf"]["ollama_model"] = source_model_name
        lineage["bf16_gguf"]["ollama_modelfile"] = str(
            source_modelfile_path.resolve()
        )
        lineage["bf16_gguf"]["ollama_create_command"] = source_create_command
        lineage["bf16_gguf"]["ollama_show"] = run(
            ["ollama", "show", source_model_name]
        )

    for key, (quantization, filename) in VARIANTS.items():
        output = checkpoint_dir / filename
        command = [
            str(quantizer),
            str(bf16_gguf),
            str(output),
            quantization,
            str(args.threads),
        ]
        if not output.is_file():
            run(command)
        variant = artifact(output) | {
            "quantization_requested": quantization,
            "quantize_command": command,
            "immediate_parent_file": str(bf16_gguf.resolve()),
            "immediate_parent_sha256": sha256(bf16_gguf),
            "original_source_sha256": EXPECTED_SOURCE_SHA256,
        }
        if not args.skip_ollama:
            model_name = f"{args.model_prefix}-{key}"
            modelfile = re.sub(
                r"^FROM\s+.+$",
                f"FROM {output.resolve()}",
                template,
                count=1,
                flags=re.MULTILINE,
            )
            modelfile_path = args.output_dir / f"{model_name}.Modelfile"
            modelfile_path.write_text(modelfile)
            create_command = [
                "ollama",
                "create",
                model_name,
                "--file",
                str(modelfile_path),
            ]
            run(create_command)
            variant["ollama_model"] = model_name
            variant["ollama_modelfile"] = str(modelfile_path.resolve())
            variant["ollama_create_command"] = create_command
            variant["ollama_show"] = run(["ollama", "show", model_name])
        lineage["variants"][key] = variant

    lineage_path = args.output_dir / "qwen25_control_lineage.json"
    lineage_path.write_text(json.dumps(lineage, indent=2) + "\n")
    print(json.dumps(lineage, indent=2))
    print(f"Wrote {lineage_path}")


if __name__ == "__main__":
    main()
