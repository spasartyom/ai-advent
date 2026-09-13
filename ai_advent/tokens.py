from dataclasses import dataclass


@dataclass(frozen=True)
class TokenReport:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
