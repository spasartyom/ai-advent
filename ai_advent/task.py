from dataclasses import dataclass

TASK_STAGES = ("idle", "planning", "execution", "validation", "done")


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
