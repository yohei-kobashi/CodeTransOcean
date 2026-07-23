"""Shared helpers for normalizing generated code before evaluation."""

import re


def strip_code_block_wrappers(source_code: str) -> str:
    """Remove a Markdown code fence and prose surrounding the fenced code."""
    if not source_code:
        return ""

    source_code = re.sub(
        r"```\s*[#\*]* *(?:Explanation|Note).*",
        "",
        source_code,
        flags=re.DOTALL | re.IGNORECASE,
    )
    stripped = re.sub(r"^.*?```[^\n]*\n", "", source_code, flags=re.DOTALL)
    stripped = re.sub(r"```\s*.*$", "", stripped, flags=re.DOTALL)
    stripped = stripped.replace("```", "")
    return stripped.strip()


def preprocess_source_code(raw_code: str | None) -> str:
    """Normalize generated source text without damaging literal escapes."""
    if not raw_code:
        return ""

    code = raw_code
    if code.startswith("\ufeff"):
        code = code.lstrip("\ufeff")

    # Decode dataset-level escaped newlines only when the entire value has no
    # real line feeds. This preserves literal "\\n" inside normal source code.
    if "\n" not in code and "\\n" in code:
        code = code.replace("\\r\\n", "\n")
        code = code.replace("\\n", "\n")

    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return strip_code_block_wrappers(code)
