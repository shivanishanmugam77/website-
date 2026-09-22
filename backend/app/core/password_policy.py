"""Password strength rules. Pure functions (no I/O) so they are trivially testable.

Length matters far more than composition rules (NIST SP 800-63B), so we require a
generous minimum length and forbid only the obviously bad cases. This is not a
breached-password check; that needs an external service and is out of scope.
"""

from __future__ import annotations

MIN_LENGTH = 12
MAX_LENGTH = 128  # also bounds the work an attacker can force us to spend hashing
MIN_DISTINCT_CHARACTERS = 5
_MIN_EMAIL_NAME_LENGTH = 4


class WeakPasswordError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def password_problems(password: str, email: str | None = None) -> list[str]:
    problems: list[str] = []
    if len(password) < MIN_LENGTH:
        problems.append(f"must be at least {MIN_LENGTH} characters long")
    if len(password) > MAX_LENGTH:
        problems.append(f"must be at most {MAX_LENGTH} characters long")
    if password and len(set(password)) < MIN_DISTINCT_CHARACTERS:
        problems.append(f"must contain at least {MIN_DISTINCT_CHARACTERS} different characters")
    if email:
        name = email.split("@", 1)[0].lower()
        if len(name) >= _MIN_EMAIL_NAME_LENGTH and name in password.lower():
            problems.append("must not contain the name part of your email address")
    return problems


def ensure_strong_password(password: str, email: str | None = None) -> None:
    problems = password_problems(password, email)
    if problems:
        raise WeakPasswordError(problems)
