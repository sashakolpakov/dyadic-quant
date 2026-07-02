from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
)
VARIANTS = {
    "source_import": None,
    "q4_k_m": "q4_K_M",
    "q6_k": "q6_K",
    "q8_0": "q8_0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args: str) -> str:
    completed = subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout


def parse_blob_path(modelfile: str) -> Path:
    match = re.search(r"^FROM\s+(.+)$", modelfile, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("could not find FROM blob in generated Modelfile")
    path = Path(match.group(1).strip())
    if not path.is_file():
        raise RuntimeError(f"Ollama blob does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--template-model", default="qwen2.5:0.5b")
    parser.add_argument("--model-prefix", default="qwen25-source")
    parser.add_argument("--output-dir", type=Path, default=Path("results/level1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    source_file = source_dir / "model.safetensors"
    source_hash = sha256(source_file)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"source hash mismatch: {source_hash} != {EXPECTED_SOURCE_SHA256}"
        )

    # Reuse only the prompt template/system/license from the registry model.
    # Every weight conversion below has FROM reset to the same Safetensors
    # directory, so no quantized model is derived from another quantized model.
    template = run("ollama", "show", args.template_model, "--modelfile")
    template = re.sub(
        r"^FROM\s+.+$",
        f"FROM {source_dir}",
        template,
        count=1,
        flags=re.MULTILINE,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = args.output_dir / "qwen25_source.Modelfile"
    modelfile_path.write_text(template)

    lineage: dict[str, object] = {
        "source": {
            "repository": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "file": str(source_file),
            "size_bytes": source_file.stat().st_size,
            "sha256": source_hash,
        },
        "template_model": args.template_model,
        "modelfile": str(modelfile_path),
        "variants": {},
    }
    for short_name, quantization in VARIANTS.items():
        model_name = f"{args.model_prefix}-{short_name}"
        command = [
            "ollama",
            "create",
            model_name,
            "--experimental",
        ]
        if quantization is not None:
            command.extend(["--quantize", quantization])
        command.extend(["--file", str(modelfile_path)])
        output = run(*command)
        generated = run("ollama", "show", model_name, "--modelfile")
        blob_path = parse_blob_path(generated)
        details = run("ollama", "show", model_name)
        lineage["variants"][short_name] = {
            "ollama_model": model_name,
            "quantization_requested": quantization,
            "create_command": command,
            "blob": str(blob_path),
            "size_bytes": blob_path.stat().st_size,
            "sha256": sha256(blob_path),
            "create_output_tail": output[-2000:],
            "show": details,
            "source_sha256": source_hash,
        }
        print(
            f"{model_name}: {quantization}, "
            f"{blob_path.stat().st_size / 1e6:.1f} MB"
        )

    lineage_path = args.output_dir / "qwen25_control_lineage.json"
    lineage_path.write_text(json.dumps(lineage, indent=2) + "\n")
    print(f"Wrote {lineage_path}")


if __name__ == "__main__":
    main()
