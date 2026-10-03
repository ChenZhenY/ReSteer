"""Aggregates steerability results into the steerability score.

A *cell* is (source task, switch step k, target task); its success rate is the fraction of its
repeats that succeeded. The steerability score is the mean cell success rate over all cells with
target != source (paper definition: 10 x 9 task pairs x 20 switch steps). The mean over all cells
including target == source, which the paper-era aggregation scripts reported, is printed as well.

    python -m resteer.eval.score results/pi05_libero
    python -m resteer.eval.score old_results_dir --legacy     # paper-era task_*/single_state_*/ layout
"""

import collections
import dataclasses
import pathlib
from typing import Literal

import numpy as np
import tyro

from resteer import tasks as _tasks
from resteer import utils

Cells = dict[tuple[str, int, str], list[bool]]


def load_cells(results_dir: pathlib.Path) -> Cells:
    cells: Cells = {}
    for path in sorted(results_dir.glob("task_*/step_*.json")):
        data = utils.read_json(path)
        source, k = data["source_task"], int(data["switch_step"])
        for r in data["rollouts"]:
            cells.setdefault((source, k, r["target"]), []).append(bool(r["success"]))
    return cells


def load_legacy_cells(results_dir: pathlib.Path, dedupe: Literal["max_sr", "all"] = "max_sr") -> Cells:
    """Reads paper-era ``task_*/single_state_*/<task>_<demo>_step<k>.json`` files.

    A (task, k) could be evaluated more than once (relaunched jobs). ``max_sr`` keeps the file with
    the highest mean success rate, as the paper-era combiner did; ``all`` pools every file.
    """
    per_step: dict[tuple[str, int], list[Cells]] = collections.defaultdict(list)
    for path in sorted(results_dir.glob("task_*/single_state_*/*.json")):
        data = utils.read_json(path)
        source = _tasks.get_task(data["metadata"]["task_name"]).name
        k = int(data["state_experiment_result"]["step_idx"])
        cells: Cells = {}
        for block in data["state_experiment_result"]["prompt_results"].values():
            target = _tasks.get_task(block["prompt"]).name
            cells.setdefault((source, k, target), []).extend(bool(e["success"]) for e in block["experiments"])
        per_step[(source, k)].append(cells)
    merged: Cells = {}
    for candidates in per_step.values():
        if dedupe == "max_sr":
            best = max(candidates, key=lambda c: np.mean([s for v in c.values() for s in v]))
            merged.update(best)
        else:
            for c in candidates:
                for key, values in c.items():
                    merged.setdefault(key, []).extend(values)
    return merged


def summarize(cells: Cells) -> dict:
    rates = {key: float(np.mean(v)) for key, v in cells.items()}
    cross = {key: r for key, r in rates.items() if key[0] != key[2]}
    sources = sorted({key[0] for key in rates}, key=lambda n: _tasks.get_task(n).id)
    steps = sorted({key[1] for key in rates})
    targets = sorted({key[2] for key in rates}, key=lambda n: _tasks.get_task(n).id)

    def mean(values) -> float | None:
        values = list(values)
        return float(np.mean(values)) if values else None

    per_task = {
        s: {
            "score": mean(r for key, r in cross.items() if key[0] == s),
            "score_incl_same_task": mean(r for key, r in rates.items() if key[0] == s),
            "success_rate": {  # [target][k]
                t: {k: rates.get((s, k, t)) for k in steps} for t in targets
            },
        }
        for s in sources
    }
    return {
        "steerability_score": mean(cross.values()),
        "score_incl_same_task": mean(rates.values()),
        "num_cells": len(rates),
        "num_cross_task_cells": len(cross),
        "num_rollouts": int(sum(len(v) for v in cells.values())),
        "score_by_switch_step": {k: mean(r for key, r in cross.items() if key[1] == k) for k in steps},
        "per_task": per_task,
    }


def plot(summary: dict, out_path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_task = summary["per_task"]
    fig, axes = plt.subplots(2, 5, figsize=(26, 10), squeeze=False)
    for ax, (source, info) in zip(axes.flat, per_task.items()):
        targets = list(info["success_rate"])
        steps = list(next(iter(info["success_rate"].values())))
        matrix = np.array([[np.nan if info["success_rate"][t][k] is None else info["success_rate"][t][k] for k in steps]
                           for t in targets])  # fmt: skip
        ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(f"{source}\nscore {info['score']:.3f}", fontsize=9)
        ax.set_xticks(range(len(steps)), steps, fontsize=6)
        ax.set_yticks(range(len(targets)), [t[:28] for t in targets], fontsize=6)
        ax.set_xlabel("switch step")
    fig.suptitle(f"steerability score {summary['steerability_score']:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


@dataclasses.dataclass
class Args:
    results_dir: tyro.conf.Positional[pathlib.Path]
    legacy: bool = False
    """Read a paper-era result directory (task_*/single_state_*/*.json)."""
    legacy_dedupe: Literal["max_sr", "all"] = "max_sr"
    plot: bool = False
    """Also write summary.png (needs the `viz` extra)."""


def main(args: Args) -> None:
    cells = load_legacy_cells(args.results_dir, args.legacy_dedupe) if args.legacy else load_cells(args.results_dir)
    if not cells:
        raise SystemExit(f"No results found under {args.results_dir}")
    summary = summarize(cells)
    utils.write_json(args.results_dir / "summary.json", summary)
    print(f"{'task':<46}{'score':>8}{'incl.same':>11}")
    for source, info in summary["per_task"].items():
        score = "n/a" if info["score"] is None else f"{info['score']:.3f}"
        print(f"{source:<46}{score:>8}{info['score_incl_same_task']:>11.3f}")
    print(
        f"\nsteerability score (target != source): {summary['steerability_score']}"
        f"\nincluding target == source:            {summary['score_incl_same_task']:.4f}"
        f"\ncells: {summary['num_cells']}  rollouts: {summary['num_rollouts']}"
        f"\nwrote {args.results_dir / 'summary.json'}"
    )
    if args.plot:
        plot(summary, args.results_dir / "summary.png")


if __name__ == "__main__":
    main(tyro.cli(Args))
