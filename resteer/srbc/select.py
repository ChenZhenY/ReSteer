"""Which prompt switches SRBC collects rollouts for.

A *switch config* is (source task, target task, switch step). Two sources, as in the paper:

- ``success_rate``: cells of a steerability evaluation (``resteer.eval.steerability`` output) whose
  success rate lies in ``[min, max]`` (default 0.2-0.9) and whose target differs from the source:
  switches the policy can make, but not reliably. Ordered by source task, switch step, target.
- ``cmi``: the low-CMI tuples written by ``resteer.cmi.compute_cmi``: states where the instruction
  barely changes the policy's actions. Tuples index states of a rollout state bank; ``policy_step``
  is the number of policy steps taken before that state (bank index minus the warm-up states).
  Tuples whose target equals the source (present in paper-era files) are kept unless
  ``include_same_task=False``.

    python -m resteer.srbc.select --source success_rate --results results/steergen_policy
    python -m resteer.srbc.select --source cmi --results results/cmi/steergen_policy
"""

import dataclasses
import pathlib
from typing import Literal

import tyro

from resteer import tasks as _tasks
from resteer import utils

Source = Literal["success_rate", "cmi"]


@dataclasses.dataclass(frozen=True)
class SwitchConfig:
    source: str  # task names (resteer.tasks)
    target: str
    switch_step: int  # step of the rollout at which the instruction switches
    success_rate: float | None = None
    mutual_info: float | None = None
    state_index: int | None = None  # raw state-bank index of a CMI tuple
    demo_name: str | None = None

    @property
    def key(self) -> str:
        return f"{self.source}->{self.target}@{self.switch_step}"


def from_eval_results(
    results_dir: pathlib.Path,
    min_success_rate: float = 0.2,
    max_success_rate: float = 0.9,
    source_tasks: tuple[int, ...] | None = None,
) -> list[SwitchConfig]:
    configs = []
    for task in _tasks.TASKS:
        if source_tasks is not None and task.id not in source_tasks:
            continue
        for path in sorted(results_dir.glob(f"task_{task.id}_*/step_*.json")):
            data = utils.read_json(path)
            k = int(data["switch_step"])
            rates = data["success_rate"]
            for target in sorted(rates, key=lambda name: _tasks.get_task(name).id):
                rate = float(rates[target])
                if target != task.name and min_success_rate <= rate <= max_success_rate:
                    configs.append(SwitchConfig(task.name, target, k, success_rate=rate))
    return configs


def _tuple_files(path: pathlib.Path) -> list[pathlib.Path]:
    """A tuples file, the combined file of a compute_cmi output directory, or a paper-era directory."""
    if path.is_file():
        return [path]
    if (path / "low_cmi_tuples.json").is_file():
        return [path / "low_cmi_tuples.json"]
    files = sorted(path.glob("task_*/step_threshold_100/*_low_mi_tuples.json"))
    if not files:
        raise FileNotFoundError(f"No low_cmi_tuples.json (or paper-era task_*/step_threshold_100/*) under {path}")
    return files


def from_low_cmi_tuples(
    path: pathlib.Path,
    include_same_task: bool = True,
    source_tasks: tuple[int, ...] | None = None,
    state_offset: int = 10,
    switch_at_state_index: bool = False,
) -> list[SwitchConfig]:
    """Reads a tuples JSON, a ``compute_cmi`` output directory, or a paper-era results directory
    (``task_<N>/step_threshold_100/*_low_mi_tuples.json``).

    ``switch_at_state_index`` switches at the raw state-bank index instead of the policy step,
    as the paper-era collector did. ``state_offset`` converts indices of files that predate the
    ``policy_step`` field.
    """
    configs = []
    for file in _tuple_files(path):
        for t in utils.read_json(file)["low_mutual_info_tuples"]:
            source = _tasks.get_task(t.get("source_task") or t["old_prompt"])
            target = _tasks.get_task(t.get("target_task") or t["new_prompt"])
            if source_tasks is not None and source.id not in source_tasks:
                continue
            if not include_same_task and source == target:
                continue
            index = int(t["new_prompt_step"])
            policy_step = int(t.get("policy_step", index - state_offset))
            configs.append(
                SwitchConfig(
                    source.name,
                    target.name,
                    index if switch_at_state_index else policy_step,
                    mutual_info=float(t["mutual_info"]),
                    state_index=index,
                    demo_name=t.get("demo_name"),
                )
            )
    return configs


def select(
    source: Source,
    results: pathlib.Path,
    *,
    min_success_rate: float = 0.2,
    max_success_rate: float = 0.9,
    include_same_task: bool = True,
    source_tasks: tuple[int, ...] | None = None,
    paper_compat: bool = False,
) -> list[SwitchConfig]:
    if source == "success_rate":
        return from_eval_results(results, min_success_rate, max_success_rate, source_tasks)
    return from_low_cmi_tuples(
        results, include_same_task, source_tasks=source_tasks, switch_at_state_index=paper_compat
    )


@dataclasses.dataclass
class Args:
    source: Source
    results: pathlib.Path
    """success_rate: a steerability results directory. cmi: a compute_cmi output directory or tuples JSON."""
    min_success_rate: float = 0.2
    max_success_rate: float = 0.9
    include_same_task: bool = True
    """cmi source: keep tuples whose target is the source task (paper-era tuple files contain them;
    compute_cmi excludes them unless run with --include-same-task or --paper-compat)."""
    tasks: tuple[int, ...] | None = None
    """Only these source task ids."""
    paper_compat: bool = False


def main(args: Args) -> None:
    configs = select(
        args.source,
        args.results,
        min_success_rate=args.min_success_rate,
        max_success_rate=args.max_success_rate,
        include_same_task=args.include_same_task,
        source_tasks=args.tasks,
        paper_compat=args.paper_compat,
    )
    for c in configs:
        score = f"sr={c.success_rate:.2f}" if c.success_rate is not None else f"cmi={c.mutual_info:.4f}"
        print(f"{c.key}  {score}")
    print(f"{len(configs)} switch configs")


if __name__ == "__main__":
    main(tyro.cli(Args))
