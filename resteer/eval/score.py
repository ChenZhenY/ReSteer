"""Aggregates steerability results into the steerability score.

A *cell* is (source task, switch step k, target task); its success rate is the fraction of its
repeats that succeeded. The steerability score is the mean cell success rate over all cells with
target != source (paper definition: 10 x 9 task pairs x 20 switch steps). The mean over all cells
including target == source, which the paper-era aggregation scripts reported, is printed as well.

    python -m resteer.eval.score results/pi05_libero
"""

import dataclasses
import pathlib

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


@dataclasses.dataclass
class Args:
    results_dir: tyro.conf.Positional[pathlib.Path]


def main(args: Args) -> None:
    cells = load_cells(args.results_dir)
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


if __name__ == "__main__":
    main(tyro.cli(Args))
