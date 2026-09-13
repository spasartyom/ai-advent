import unittest

from ai_advent.tokens import TokenReport


class TokenReportTests(unittest.TestCase):
    def test_token_report_stores_api_usage(self) -> None:
        report = TokenReport(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )

        self.assertEqual(report.prompt_tokens, 10)
        self.assertEqual(report.completion_tokens, 20)
        self.assertEqual(report.total_tokens, 30)


if __name__ == "__main__":
    unittest.main()
