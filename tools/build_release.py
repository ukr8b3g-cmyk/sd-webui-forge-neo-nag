"""Build and verify a reproducible release ZIP from the committed Git tree.

Uses only Python's standard library. Does not import Forge, PyTorch or the model.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "sd-webui-forge-neo-nag/"


def git(*arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments])


def build(output_directory: Path) -> dict[str, object]:
    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    files = git("ls-tree", "-rz", "--name-only", "HEAD").decode("utf-8").split("\0")
    files = sorted(path for path in files if path and not path.startswith(".github/"))
    payloads: dict[str, bytes] = {}
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Unsafe archive path: {name}")
        payloads[name] = git("show", f"HEAD:{name}")
        if path.suffix == ".py":
            compile(payloads[name], name, "exec")

    version_tree = ast.parse(payloads["forge_neo_nag/__init__.py"].decode("utf-8"))
    version = None
    for node in version_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            version = ast.literal_eval(node.value)
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Release version must have the form X.Y.Z")

    notes = f"docs/releases/v{version}.md"
    required = {"README.md", "LICENSE", "NOTICE.md", "licenses/ComfyUI-Krea2-NAG-MIT.txt",
                "scripts/forge_neo_nag.py", "javascript/forge_neo_nag.js", notes}
    missing = required.difference(payloads)
    if missing:
        raise ValueError(f"Required release files are missing: {sorted(missing)}")
    if "BUILD_INFO.json" in payloads:
        raise ValueError("BUILD_INFO.json is reserved for release provenance")
    payloads["BUILD_INFO.json"] = (json.dumps({
        "version": version, "git_commit": commit, "source_file_count": len(files),
        "verification_scope": "CPU validation recorded; pretrained-model GPU validation pending",
    }, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    archive = output_directory / f"sd-webui-forge-neo-nag-v{version}.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(payloads.items()):
            item = ZipInfo(PREFIX + name, date_time=(1980, 1, 1, 0, 0, 0))
            item.create_system = 3
            item.external_attr = 0o100644 << 16
            item.compress_type = ZIP_DEFLATED
            z.writestr(item, data, compresslevel=9)

    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
        expected = {PREFIX + name for name in payloads}
        if set(z.namelist()) != expected or len(z.infolist()) != len(expected):
            raise ValueError("ZIP file list does not match the committed release content")
        for name, data in payloads.items():
            if z.read(PREFIX + name) != data:
                raise ValueError(f"ZIP content differs from the committed file: {name}")

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    result = {"tag": f"v{version}", "commit": commit, "zip": str(archive),
              "sha256_file": str(checksum), "notes": notes, "sha256": digest,
              "archive_file_count": len(payloads), "size_bytes": archive.stat().st_size}
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as stream:
            for key in ("tag", "zip", "sha256_file", "notes"):
                stream.write(f"{key}={result[key]}\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), ensure_ascii=False, indent=2))
