import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ai_advent.scheduler import (
    create_reminder,
    list_reminders,
    load_scheduler_data,
    run_due_tasks,
)


class SchedulerTests(unittest.TestCase):
    def test_create_reminder_persists_pending_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scheduler.json"
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)

            reminder = create_reminder(
                path,
                title="Review MCP",
                due_in_seconds=60,
                note="Day 18",
                now=now,
            )

            data = load_scheduler_data(path)
            self.assertEqual(reminder["title"], "Review MCP")
            self.assertEqual(reminder["status"], "pending")
            self.assertEqual(data["reminders"], [reminder])

    def test_run_due_tasks_completes_due_reminders_and_returns_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scheduler.json"
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)
            create_reminder(path, title="Due", due_in_seconds=0, now=now)
            create_reminder(path, title="Later", due_in_seconds=120, now=now)

            result = run_due_tasks(path, now=now)

            self.assertEqual(result["due_count"], 1)
            self.assertEqual(result["pending_count"], 1)
            self.assertEqual(result["completed_count"], 1)
            self.assertEqual(result["completed"][0]["title"], "Due")
            self.assertIn("Completed 1 due study reminder", result["summary"])

    def test_list_reminders_groups_by_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "scheduler.json"
            now = datetime(2026, 1, 1, tzinfo=timezone.utc)
            create_reminder(path, title="Due", now=now)
            run_due_tasks(path, now=now)

            result = list_reminders(path)

            self.assertEqual(result["total"], 1)
            self.assertEqual(result["pending_count"], 0)
            self.assertEqual(result["completed_count"], 1)


if __name__ == "__main__":
    unittest.main()
