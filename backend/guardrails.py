import re

BLOCK_PATTERNS = [
    r"ignore previous instructions",
    r"system prompt",
    r"reveal.*prompt",
    r"act as system",
    r"jailbreak"
]

def guardrail_check(query: str) -> bool:
    q = query.lower()

    for pattern in BLOCK_PATTERNS:
        if re.search(pattern, q):
            return False

    return True