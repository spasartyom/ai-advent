from dataclasses import dataclass

TASK_STAGES = ("idle", "planning", "execution", "validation", "done")
TASK_TRANSITIONS = {
    "idle": ("planning",),
    "planning": ("execution",),
    "execution": ("validation",),
    "validation": ("execution", "done"),
    "done": ("planning",),
}


@dataclass(frozen=True)
class TaskState:
    stage: str = "idle"
    title: str = ""
    current_step: str = ""
    expected_action: str = ""
    paused: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "title": self.title,
            "current_step": self.current_step,
            "expected_action": self.expected_action,
            "paused": self.paused,
        }


def create_task_state(
    *,
    stage: str = "idle",
    title: str = "",
    current_step: str = "",
    expected_action: str = "",
    paused: bool = False,
) -> TaskState:
    if stage not in TASK_STAGES:
        raise ValueError(f"Task stage must be one of: {', '.join(TASK_STAGES)}.")
    return TaskState(
        stage=stage,
        title=title,
        current_step=current_step,
        expected_action=expected_action,
        paused=paused,
    )


def validate_task_transition(
    current_stage: str,
    next_stage: str,
    *,
    approved: bool = False,
) -> None:
    if current_stage not in TASK_STAGES:
        raise ValueError(f"Current task stage must be one of: {', '.join(TASK_STAGES)}.")
    if next_stage not in TASK_STAGES:
        raise ValueError(f"Next task stage must be one of: {', '.join(TASK_STAGES)}.")
    if current_stage == next_stage:
        return
    if next_stage not in TASK_TRANSITIONS[current_stage]:
        allowed = ", ".join(TASK_TRANSITIONS[current_stage]) or "(none)"
        raise ValueError(
            f"Cannot transition task from {current_stage} to {next_stage}. "
            f"Allowed next stages: {allowed}."
        )
    if current_stage == "planning" and next_stage == "execution" and not approved:
        raise ValueError(
            "Cannot transition task from planning to execution before the plan is approved. "
            "Use /task approve."
        )


def task_state_from_dict(value: dict[object, object]) -> TaskState:
    stage = value.get("stage", "idle")
    title = value.get("title", "")
    current_step = value.get("current_step", "")
    expected_action = value.get("expected_action", "")
    paused = value.get("paused", False)
    if not isinstance(stage, str):
        raise ValueError("Task state stage must be a string.")
    if not isinstance(title, str):
        raise ValueError("Task state title must be a string.")
    if not isinstance(current_step, str):
        raise ValueError("Task state current_step must be a string.")
    if not isinstance(expected_action, str):
        raise ValueError("Task state expected_action must be a string.")
    if not isinstance(paused, bool):
        raise ValueError("Task state paused must be a boolean.")
    return create_task_state(
        stage=stage,
        title=title,
        current_step=current_step,
        expected_action=expected_action,
        paused=paused,
    )
