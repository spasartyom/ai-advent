import unittest

from ai_advent.task import validate_task_transition


class TaskStateMachineTests(unittest.TestCase):
    def test_rejects_planning_to_execution_without_approval(self) -> None:
        with self.assertRaises(ValueError):
            validate_task_transition("planning", "execution")

    def test_allows_planning_to_execution_with_approval(self) -> None:
        validate_task_transition("planning", "execution", approved=True)

    def test_rejects_execution_to_done(self) -> None:
        with self.assertRaises(ValueError):
            validate_task_transition("execution", "done")

    def test_allows_validation_to_done(self) -> None:
        validate_task_transition("validation", "done")


if __name__ == "__main__":
    unittest.main()
