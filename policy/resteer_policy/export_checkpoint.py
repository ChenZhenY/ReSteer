"""Prepares a trained checkpoint for release and for serving with stock openpi.

Copies ``params/`` and the normalization statistics into ``<out>``, laid out the way openpi's
``pi05_libero`` config expects (``assets/physical-intelligence/libero/norm_stats.json``); the
optimizer state (``train_state/``) is dropped. Checkpoints from the paper's training runs stored the
statistics under ``assets/libero``; both layouts are accepted.

    uv run --project third_party/openpi python -m resteer_policy.export_checkpoint \\
        checkpoints/pi05_libero_resteer_srbc/srbc/9999 exported/resteer_srbc

Then serve it with ``scripts/serve_policy.sh --checkpoint exported/resteer_srbc``.
"""

import dataclasses
import hashlib
import pathlib
import shutil

from openpi.shared import download
import tyro

ASSET_ID = "physical-intelligence/libero"
REFERENCE_NORM_STATS = f"gs://openpi-assets/checkpoints/pi05_libero/assets/{ASSET_ID}/norm_stats.json"


@dataclasses.dataclass
class Args:
    checkpoint: tyro.conf.Positional[pathlib.Path]
    """Checkpoint step directory (contains params/ and assets/)."""
    out: tyro.conf.Positional[pathlib.Path]
    check_norm_stats: bool = True
    """Warn if the statistics differ from the released pi05_libero ones (the ReSteer configs reuse them)."""


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args: Args) -> None:
    candidates = [args.checkpoint / "assets" / ASSET_ID / "norm_stats.json",
                  args.checkpoint / "assets" / "libero" / "norm_stats.json"]  # fmt: skip
    norm_stats = next((p for p in candidates if p.exists()), None)
    if norm_stats is None or not (args.checkpoint / "params").is_dir():
        raise SystemExit(f"{args.checkpoint} is not an openpi checkpoint step dir with params/ and norm stats")
    if args.out.exists():
        raise SystemExit(f"{args.out} already exists")
    shutil.copytree(args.checkpoint / "params", args.out / "params")
    (args.out / "assets" / ASSET_ID).mkdir(parents=True)
    shutil.copy2(norm_stats, args.out / "assets" / ASSET_ID / "norm_stats.json")
    if args.check_norm_stats:
        reference = pathlib.Path(download.maybe_download(REFERENCE_NORM_STATS))
        if _sha256(reference) != _sha256(norm_stats):
            print(f"WARNING: {norm_stats} differs from the released pi05_libero statistics")
    print(f"Exported {args.checkpoint} -> {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
