"""Built-in doctor accounts.

Accounts are hard-coded on purpose: there is no in-app registration, and only
the developer can add accounts by editing BUILTIN_USERS below and shipping a
new build. Keys are doctor display names, values are login passwords.
"""

from __future__ import annotations

# Developer-maintained account list. Add a doctor by adding a new line here.
BUILTIN_USERS: dict[str, str] = {
    "张医生": "123456",
    "李医生": "123456",
    "王医生": "123456",
}


def verify_login(doctor_name: str, password: str) -> bool:
    """Return True only when the name/password pair matches a built-in account."""
    name = doctor_name.strip()
    return name in BUILTIN_USERS and BUILTIN_USERS[name] == password


def account_names() -> list[str]:
    return list(BUILTIN_USERS)
