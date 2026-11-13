"""Download a pinned public HF model and verify every file against Hub metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=2_000_000_000)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision) or args.max_bytes <= 0:
        parser.error("require an exact 40-character revision and positive byte limit")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"status": "failed", "repo_id": args.repo_id, "revision": args.revision,
              "source": "Hugging Face directly on the executing host", "files": []}
    try:
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi(token=False).model_info(args.repo_id, revision=args.revision,
                                           files_metadata=True)
        if info.sha != args.revision or info.private or info.gated:
            raise ValueError("Expected the exact revision of a public, ungated model")
        files = [f for f in info.siblings if "/" not in f.rfilename and
                 (f.rfilename.endswith((".safetensors", ".json")) or
                  f.rfilename in {"LICENSE", "README.md", "merges.txt"})]
        names = {f.rfilename for f in files}
        if not {"config.json", "tokenizer_config.json", "tokenizer.json"} <= names:
            raise ValueError("Required configuration/tokenizer files are missing")
        if not any(name.endswith(".safetensors") for name in names):
            raise ValueError("No safetensors weights found")
        if any(f.size is None or f.size <= 0 for f in files):
            raise ValueError("Missing or invalid file size")
        total = sum(f.size for f in files)
        if total > args.max_bytes:
            raise ValueError(f"Payload {total} exceeds limit {args.max_bytes}")
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(args.cache_dir).free < 2 * total:
            raise ValueError("Insufficient free disk space for download and temporary files")
        plan = {"repo_id": args.repo_id, "revision": info.sha, "total_bytes": total,
                "files": [{"path": f.rfilename, "bytes": f.size,
                           "hub_lfs_sha256": f.lfs.sha256 if f.lfs else None,
                           "hub_git_blob_sha1": f.blob_id} for f in files]}
        (args.output_dir / "download-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(f"Downloading {len(files)} files, {total} bytes, directly from HF", flush=True)
        snapshot = Path(snapshot_download(args.repo_id, revision=args.revision,
                                         cache_dir=args.cache_dir, allow_patterns=sorted(names),
                                         token=False, max_workers=2))
        result["snapshot_path"] = str(snapshot)
        result["total_bytes"] = total
        for file in files:
            target = snapshot / file.rfilename
            size = target.stat().st_size
            sha256 = hashlib.sha256()
            git_sha1 = hashlib.sha1(f"blob {size}\0".encode())
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    sha256.update(block)
                    git_sha1.update(block)
            expected = file.lfs.sha256 if file.lfs else file.blob_id
            if not expected:
                raise ValueError(f"No upstream digest for {file.rfilename}")
            observed = sha256.hexdigest() if file.lfs else git_sha1.hexdigest()
            passed = size == file.size and observed == expected
            result["files"].append({"path": file.rfilename, "bytes": size,
                                    "sha256": sha256.hexdigest(),
                                    "hub_digest_kind": "sha256" if file.lfs else "git-blob-sha1",
                                    "hub_digest": expected, "verified": passed})
            if not passed:
                raise ValueError(f"Upstream digest or size mismatch: {file.rfilename}")
        result["status"] = "verified"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "files": len(result["files"]),
                      "output_dir": str(args.output_dir), "error": result.get("error")}), flush=True)
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
