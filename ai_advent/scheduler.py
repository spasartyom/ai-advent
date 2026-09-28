import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_SCHEDULER_FILE = ".ai-advent/study-scheduler.json"


def create_reminder(
    storage_path: Path,
    *,
    title: str,
    due_in_seconds: int = 0,
    note: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    if not title.strip():
        raise ValueError("Reminder title cannot be empty.")
    if due_in_seconds < 0:
        raise ValueError("due_in_seconds must be zero or greater.")

    current_time = _utc_now(now)
    reminder = {
        "id": uuid.uuid4().hex,
        "title": title.strip(),
        "note": note.strip(),
        "status": "pending",
        "created_at": _to_iso(current_time),
        "due_at": _to_iso(current_time + timedelta(seconds=due_in_seconds)),
        "completed_at": None,
    }
    data = load_scheduler_data(storage_path)
    data["reminders"].append(reminder)
    save_scheduler_data(storage_path, data)
    return reminder


def list_reminders(storage_path: Path) -> dict[str, Any]:
    data = load_scheduler_data(storage_path)
    reminders = data["reminders"]
    pending = [reminder for reminder in reminders if reminder["status"] == "pending"]
    completed = [
        reminder for reminder in reminders if reminder["status"] == "completed"
    ]
    return {
        "total": len(reminders),
        "pending_count": len(pending),
        "completed_count": len(completed),
        "pending": pending,
        "completed": completed,
    }


def run_due_tasks(
    storage_path: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = _utc_now(now)
    data = load_scheduler_data(storage_path)
    due_reminders: list[dict[str, Any]] = []

    for reminder in data["reminders"]:
        if reminder["status"] != "pending":
            continue
        if _from_iso(reminder["due_at"]) <= current_time:
            reminder["status"] = "completed"
            reminder["completed_at"] = _to_iso(current_time)
            due_reminders.append(reminder)

    pending_count = sum(
        1 for reminder in data["reminders"] if reminder["status"] == "pending"
    )
    completed_count = sum(
        1 for reminder in data["reminders"] if reminder["status"] == "completed"
    )
    run_summary = {
        "ran_at": _to_iso(current_time),
        "due_count": len(due_reminders),
        "pending_count": pending_count,
        "completed_count": completed_count,
        "completed_ids": [reminder["id"] for reminder in due_reminders],
        "summary": _build_summary(due_reminders, pending_count),
    }
    data["runs"].append(run_summary)
    save_scheduler_data(storage_path, data)
    return {
        **run_summary,
        "completed": due_reminders,
    }


def load_scheduler_data(storage_path: Path) -> dict[str, Any]:
    if not storage_path.exists():
        return _empty_data()

    with storage_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if not isinstance(raw_data, dict):
        raise ValueError("Scheduler file must contain a JSON object.")

    reminders = raw_data.get("reminders", [])
    runs = raw_data.get("runs", [])
    if not isinstance(reminders, list) or not isinstance(runs, list):
        raise ValueError("Scheduler file has invalid reminders or runs.")
    return {
        "reminders": reminders,
        "runs": runs,
    }


def save_scheduler_data(storage_path: Path, data: dict[str, Any]) -> None:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    with storage_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _empty_data() -> dict[str, Any]:
    return {
        "reminders": [],
        "runs": [],
    }


def _build_summary(due_reminders: list[dict[str, Any]], pending_count: int) -> str:
    if not due_reminders:
        return f"No due study reminders. Pending reminders: {pending_count}."

    titles = ", ".join(reminder["title"] for reminder in due_reminders)
    return (
        f"Completed {len(due_reminders)} due study reminder(s): {titles}. "
        f"Pending reminders: {pending_count}."
    )


def _utc_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _to_iso(value: datetime) -> str:
    return _utc_now(value).isoformat()


def _from_iso(value: str) -> datetime:
    return _utc_now(datetime.fromisoformat(value))
