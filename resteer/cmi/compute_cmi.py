"""Turns sampled entropies into CMI, and selects low-CMI (poorly steerable) states.

For a state ``s`` of source task ``i`` and instruction ``l``::

    CMI(s, l)        = max(0, H(A|s) - H(A|s,l))
    normalized(s, l) = CMI(s, l) / H(A|s)        (fraction of action uncertainty explained by l; 0 if H(A|s) <= 1e-9)

A task's CMI is the mean over all its (state, instruction) cells, including the task's own
instruction (the number reported in the paper, which uses the normalized form); the overall CMI is
the unweighted mean over tasks.

Low-CMI tuples, the switching configurations that CMI-guided data generation targets, are the
``low_cmi_percent`` % of a task's cells with the lowest CMI (stable ascending sort), each giving
(source task, target instruction, switch step). They are written per task and for all tasks in the
format read by resteer.srbc and resteer.steergen.select_by_cmi.

    python -m resteer.cmi.compute_cmi results/cmi/pi05_libero
"""

import dataclasses
import pathlib

import numpy as np
import tyro

from resteer import tasks as _tasks
from resteer import utils


@dataclasses.dataclass
class Args:
    results_dir: tyro.conf.Positional[pathlib.Path]
    """Output directory of resteer.cmi.sample_actions."""
    normalize: bool = True
    """Rank and report the normalized CMI (the paper's default)."""
    low_cmi_percent: float = 50.0
    max_policy_step: int | None = None
    """Only select switch steps <= this policy step."""
    include_same_task: bool = False
    """Allow target == source in the low-CMI tuples (a same-task 'switch' generates no switching data)."""
    paper_compat: bool = False
    """Reproduce the paper-era selection: raw bank index <= 100 and same-task tuples included."""
    plot: bool = False
    """Write cmi_heatmaps.png (needs the `viz` extra)."""


def cmi(state_entropy: float, prompt_entropy: float, normalize: bool) -> float:
    value = max(0.0, state_entropy - prompt_entropy)
    if normalize:
        value = value / state_entropy if state_entropy > 1e-9 else 0.0
    return value


def load_records(results_dir: pathlib.Path, normalize: bool) -> dict[str, list[dict]]:
    """Per source task: one record per (state, target) in state order, then target-id order."""
    records: dict[str, list[dict]] = {}
    for path in sorted(results_dir.glob("task_*/state_*.json")):
        d = utils.read_json(path)
        source = _tasks.get_task(d["source_task"])
        for target_name, h in sorted(d["prompt_entropy"].items(), key=lambda kv: _tasks.get_task(kv[0]).id):
            target = _tasks.get_task(target_name)
            records.setdefault(source.name, []).append(
                {
                    "source_task": source.name,
                    "target_task": target.name,
                    "policy_step": d["policy_step"],
                    "state_index": d["state_index"],
                    "demo": d["demo"],
                    "cmi": cmi(d["state_entropy"], h, normalize),
                    # Paper-era field names, read by the paper's SRBC/SteerGen selection scripts.
                    "old_prompt": source.title,
                    "new_prompt": target.title,
                    "new_prompt_step": d["state_index"],
                    "demo_name": d["demo"],
                }
            )
    for task_records in records.values():
        for r in task_records:
            r["mutual_info"] = r["cmi"]
    return records


def select_low_cmi(records: list[dict], args: Args) -> list[dict]:
    if args.paper_compat:
        pool = [r for r in records if r["state_index"] <= 100]
    else:
        pool = [
            r
            for r in records
            if (args.max_policy_step is None or r["policy_step"] <= args.max_policy_step)
            and (args.include_same_task or r["source_task"] != r["target_task"])
        ]
    if not pool:
        return []
    ranked = sorted(pool, key=lambda r: r["cmi"])  # stable: ties keep state/target order
    return ranked[: max(1, int(len(ranked) * args.low_cmi_percent / 100.0))]


def summarize(records: dict[str, list[dict]]) -> dict:
    per_task = {}
    for name, rs in sorted(records.items(), key=lambda kv: _tasks.get_task(kv[0]).id):
        steps = sorted({r["policy_step"] for r in rs})
        per_task[name] = {
            "cmi": float(np.mean([r["cmi"] for r in rs])),
            "cmi_cross_task": float(np.mean([r["cmi"] for r in rs if r["target_task"] != name])),
            "cmi_by_step": {k: float(np.mean([r["cmi"] for r in rs if r["policy_step"] == k])) for k in steps},
            "cmi_by_target": {
                t.name: float(np.mean([r["cmi"] for r in rs if r["target_task"] == t.name]))
                for t in _tasks.TASKS
                if any(r["target_task"] == t.name for r in rs)
            },
        }
    return {"cmi": float(np.mean([v["cmi"] for v in per_task.values()])), "per_task": per_task}


def plot(records: dict[str, list[dict]], out_path: pathlib.Path, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(26, 10), squeeze=False)
    for ax, (name, rs) in zip(axes.flat, sorted(records.items(), key=lambda kv: _tasks.get_task(kv[0]).id)):
        steps = sorted({r["policy_step"] for r in rs})
        matrix = np.full((len(_tasks.TASKS), len(steps)), np.nan)
        for r in rs:
            matrix[_tasks.get_task(r["target_task"]).id, steps.index(r["policy_step"])] = r["cmi"]
        ax.imshow(matrix, cmap="magma", aspect="auto")
        ax.set_title(f"{name}\nmean {np.nanmean(matrix):.3f}", fontsize=9)
        ax.set_xticks(range(len(steps)), steps, fontsize=6)
        ax.set_yticks(range(len(_tasks.TASKS)), [t.name[:28] for t in _tasks.TASKS], fontsize=6)
        ax.set_xlabel("policy step")
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def count_undefined(results_dir: pathlib.Path) -> int:
    """(state, instruction) cells whose entropies are NaN (identical action samples); their CMI counts as 0."""
    count = 0
    for path in results_dir.glob("task_*/state_*.json"):
        d = utils.read_json(path)
        count += sum(not np.isfinite(d["state_entropy"] - h) for h in d["prompt_entropy"].values())
    return count


def main(args: Args) -> None:
    records = load_records(args.results_dir, args.normalize)
    if not records:
        raise SystemExit(f"No state_*.json files under {args.results_dir}")
    if undefined := count_undefined(args.results_dir):
        print(f"WARNING: {undefined} cells have undefined entropy (identical action samples); their CMI is 0")
    summary = summarize(records)
    summary["normalized"] = args.normalize
    utils.write_json(args.results_dir / "cmi_summary.json", summary)

    selection = {
        "low_cmi_percent": args.low_cmi_percent,
        "max_policy_step": args.max_policy_step,
        "include_same_task": args.include_same_task,
        "paper_compat": args.paper_compat,
        "normalized": args.normalize,
    }
    all_tuples = []
    for name, rs in records.items():
        tuples = select_low_cmi(rs, args)
        all_tuples.extend(tuples)
        task = _tasks.get_task(name)
        utils.write_json(
            args.results_dir / f"task_{task.id}_{task.name}" / "low_cmi_tuples.json",
            {"task_name": name, "parameters": selection, "low_mutual_info_tuples": tuples},
        )
    utils.write_json(
        args.results_dir / "low_cmi_tuples.json",
        {"task_name": "all", "parameters": selection, "low_mutual_info_tuples": all_tuples},
    )

    kind = "normalized CMI" if args.normalize else "CMI"
    for name, v in summary["per_task"].items():
        print(f"{name:<46}{v['cmi']:.4f}")
    print(f"\nmean {kind} over tasks: {summary['cmi']:.4f}\nlow-CMI tuples: {len(all_tuples)}")
    if args.plot:
        plot(records, args.results_dir / "cmi_heatmaps.png", f"{kind} (mean {summary['cmi']:.3f})")


if __name__ == "__main__":
    main(tyro.cli(Args))
