"""The 10 LIBERO-Goal tasks studied by ReSteer.

All tasks live in one scene (``libero_goal_steerability.bddl``), whose success dictionary reports
every task's goal predicate. Task ids follow the alphabetical order of the task names; this is
the ``task_0 ... task_9`` order of the paper's evaluation results.

Prompts: the policy receives the lower-case instruction (``Task.prompt``). Paper-era scripts
also used a Title Case variant (``Task.title``) for CMI queries and for the prompts stored in
generated datasets; see :func:`format_prompt`.
"""

import dataclasses
from typing import Literal

PromptStyle = Literal["lower", "title"]


@dataclasses.dataclass(frozen=True)
class Task:
    id: int
    name: str
    # Keys of ``env.step(action, return_success_dict=True)[2]`` that mean this task succeeded.
    goal_keys: tuple[str, ...]

    @property
    def prompt(self) -> str:
        return self.name.replace("_", " ")

    @property
    def title(self) -> str:
        return self.prompt.title()


TASKS: tuple[Task, ...] = (
    Task(0, "open_the_middle_drawer_of_the_cabinet", ("open_wooden_cabinet_1_middle_region",)),
    Task(1, "open_the_top_drawer_and_put_the_bowl_inside", ("in_akita_black_bowl_1_wooden_cabinet_1_top_region",)),
    Task(2, "push_the_plate_to_the_front_of_the_stove", ("on_plate_1_main_table_stove_front_region",)),
    Task(3, "put_the_bowl_on_the_plate", ("on_akita_black_bowl_1_plate_1",)),
    Task(4, "put_the_bowl_on_the_stove", ("on_akita_black_bowl_1_flat_stove_1_cook_region",)),
    Task(5, "put_the_bowl_on_top_of_the_cabinet", ("on_akita_black_bowl_1_wooden_cabinet_1_top_side",)),
    Task(6, "put_the_cream_cheese_in_the_bowl", ("on_cream_cheese_1_akita_black_bowl_1",)),
    Task(7, "put_the_wine_bottle_on_the_rack", ("on_wine_bottle_1_wine_rack_1_top_region",)),
    Task(8, "put_the_wine_bottle_on_top_of_the_cabinet", ("on_wine_bottle_1_wooden_cabinet_1_top_side",)),
    Task(9, "turn_on_the_stove", ("turnon_flat_stove_1",)),
)

# Paper-era SteerGen files identify tasks by their index in this order (the task table of the
# original generator; it is neither alphabetical nor LIBERO's benchmark order).
LEGACY_STEERGEN_ORDER: tuple[str, ...] = (
    "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "turn_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
    "put_the_cream_cheese_in_the_bowl",
    "open_the_middle_drawer_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_stove",
)


def _normalize(text: str) -> str:
    return " ".join(text.replace("_", " ").strip().rstrip(".").lower().split())


_BY_KEY = {_normalize(t.name): t for t in TASKS}


def get_task(key: "int | str | Task") -> Task:
    """Looks a task up by id, snake_case name, or prompt text in any capitalization."""
    if isinstance(key, Task):
        return key
    if isinstance(key, int):
        return TASKS[key]
    try:
        return _BY_KEY[_normalize(key)]
    except KeyError:
        raise KeyError(f"Unknown LIBERO-Goal task: {key!r}") from None


def format_prompt(task: "int | str | Task", style: PromptStyle = "lower") -> str:
    task = get_task(task)
    if style == "lower":
        return task.prompt
    if style == "title":
        return task.title
    raise ValueError(f"Unknown prompt style: {style!r}")


def is_success(task: "int | str | Task", success_dict: dict) -> bool:
    """Whether ``task``'s goal holds in a success dictionary returned by the steerability scene."""
    return any(success_dict.get(key, False) for key in get_task(task).goal_keys)
