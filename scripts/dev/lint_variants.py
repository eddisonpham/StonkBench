#!/usr/bin/env python
"""Lint gan_tests/variants.json — schema enforcement + ROADMAP parity check.

Enforces:
1. variants.json top-level must be a list of variant items.
2. Required fields per item: variant, base_model, comment, metadata, training_hp,
   risk_class, compute_estimate_min, safety_warnings.
3. risk_class \u2208 {low, medium, high}.
4. compute_estimate_min is a positive int.
5. safety_warnings is a list of strings (may be empty).
6. Variant name referenced in ROADMAP.md is defined in some variants.json
   (catches silent rename / forgotten registration).
7. Each variant's `aggressive` tag (e.g. FIX_F1, FIX_F2+F3, FIX_F1_PAPER) refers
   only to F-digit IDs that have a corresponding `### F\d+` heading in the
   matching GAN's ADAPTER_VS_PAPER.md (catches stray FIX_F99 typos).
   Sub-tags like FIX_F1_PAPER pass-through; only the F-digit prefix is checked.

Usage:
    python scripts/dev/lint_variants.py [gan_tests_dir]

Default gan_tests_dir = ./gan_tests
Exit code = 0 if all checks pass, 1 if any failure.

This script is intentionally pure-stdlib (no project imports) so it can be
invoked from any working directory and won't depend on the rest of the
codebase being importable.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


REQUIRED_FIELDS = (
    "variant",
    "base_model",
    "comment",
    "metadata",
    "training_hp",
    "risk_class",
    "compute_estimate_min",
    "safety_warnings",
)
VALID_RISK_CLASSES = {"low", "medium", "high"}
# Map GAN folder -> base_model regex prefix pattern used in variant names.
# (Used in #6/ROADMAP extraction.)
_BASE_PREFIX = {
    "quantgan": "quantgan",
    "cond_sig_wgan": "cond_sig_wgan",
}
# (2026-07-30 cleanup note: GAN_FOLDERS used to also include "pcf_gan".
# PCF-GAN was removed entirely on 2026-07-30 (12+ variant attempts never
# un-collapsed mean std_ratio=0.45). list(_BASE_PREFIX.keys()) is the
# canonical source of truth — keep them in sync.)
GAN_FOLDERS = list(_BASE_PREFIX.keys())


def _load_variants(gan_dir: Path) -> list[dict] | None:
    p = gan_dir / "variants.json"
    if not p.exists():
        print(f"[ERR] missing {p}")
        return None
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        print(f"[ERR] {p} parse failure: {exc}")
        return None
    if not isinstance(data, list):
        print(f"[ERR] {p} top-level must be a list, got {type(data).__name__}")
        return None
    return data


def _lint_item(
    item: dict,
    source: Path,
    idx: int,
    errors: list[str],
    f_digits_for_gan: set[int] | None,
) -> str | None:
    """Append errors. Return the variant name (or None if missing)."""
    prefix = f"{source}[{idx}]"
    name = item.get("variant")
    for required in REQUIRED_FIELDS:
        if required not in item:
            errors.append(f"{prefix} missing required field '{required}'")
    if not isinstance(name, str) or not name:
        errors.append(f"{prefix} variant must be non-empty string")
    rc = item.get("risk_class")
    if rc not in VALID_RISK_CLASSES:
        errors.append(
            f"{prefix} risk_class={rc!r} invalid (must be in {sorted(VALID_RISK_CLASSES)})"
        )
    est = item.get("compute_estimate_min")
    if not isinstance(est, int) or isinstance(est, bool) or est <= 0:
        errors.append(f"{prefix} compute_estimate_min={est!r} must be positive int")
    warn = item.get("safety_warnings")
    if not isinstance(warn, list) or any(
        not isinstance(x, str) for x in warn
    ):
        errors.append(f"{prefix} safety_warnings must be list of strings")
    md = item.get("metadata")
    if not isinstance(md, dict):
        errors.append(f"{prefix} metadata must be dict")
    hp = item.get("training_hp")
    if not isinstance(hp, dict):
        errors.append(f"{prefix} training_hp must be dict")
    # Check aggressive tag references valid F-digits in ADAPTER_VS_PAPER.
    # Aggressive tag tokens are split on '+' / '_' and matched against the
    # strict form `F<digits>` to avoid spurious matches inside substrings
    # (e.g. "FIX_FOO_F1_BAR" would otherwise falsely extract `1`).
    agg = item.get("aggressive", "")
    if isinstance(agg, str) and f_digits_for_gan is not None and agg.startswith("FIX_"):
        tokens = re.split(r"[+_]", agg)
        digits = [int(t[1:]) for t in tokens if re.fullmatch(r"F\d+", t)]
        for d in digits:
            if d not in f_digits_for_gan:
                errors.append(
                    f"{prefix} aggressive={agg!r} references FIX_F{d} "
                    f"without matching '### F{d}' heading in ADAPTER_VS_PAPER.md"
                )
    return name if isinstance(name, str) and name else None


def _f_digits_from_adapters(adapter_md: Path) -> set[int]:
    """Extract F-digit set from `### F\d+` headings in ADAPTER_VS_PAPER.md."""
    if not adapter_md.exists():
        return set()
    text = adapter_md.read_text()
    return {int(d) for d in re.findall(r"^### F(\d+)\b", text, flags=re.MULTILINE)}


def _roadmap_names(roadmap: Path) -> set[str]:
    if not roadmap.exists():
        return set()
    text = roadmap.read_text()
    # 2026-07-30 cleanup: rebuild backtick-wrapped variant-name pattern
    # dynamically from GAN_FOLDERS (was previously a hard-coded alternation
    # including pcf_gan_*, removed by the PCF-GAN cleanup). GAN_FOLDERS is
    # derived from _BASE_PREFIX.keys() above, so adding a new canonical GAN
    # only requires editing _BASE_PREFIX — the lint pattern auto-follows.
    pattern = r"`((?:" + "|".join(re.escape(g) for g in GAN_FOLDERS) + r")[\w_]*)`"
    return set(re.findall(pattern, text, flags=re.MULTILINE))


def main(argv: list[str]) -> int:
    base = Path(argv[1] if len(argv) > 1 else "./gan_tests").resolve()
    if not base.is_dir():
        print(f"[ERR] {base} is not a directory")
        return 1

    errors: list[str] = []
    defined: dict[str, str] = {}  # variant -> gan folder
    total = 0

    for gan in GAN_FOLDERS:
        f_digits = _f_digits_from_adapters(base / gan / "ADAPTER_VS_PAPER.md")
        data = _load_variants(base / gan)
        if data is None:
            continue
        for i, item in enumerate(data):
            total += 1
            name = _lint_item(
                item, base / gan / "variants.json", i, errors, f_digits
            )
            if name:
                if name in defined:
                    errors.append(
                        f"[ERR] duplicate variant '{name}' in "
                        f"{defined[name]}/variants.json and {gan}/variants.json"
                    )
                else:
                    defined[name] = gan

    # Cross-doc: ROADMAP names must be defined.
    missing = sorted(_roadmap_names(base / "ROADMAP.md") - set(defined.keys()))
    for n in missing:
        errors.append(f"[ERR] ROADMAP.md references '{n}' not defined in any variants.json")

    print(f"Total variants across {len(GAN_FOLDERS)} GANs: {total}")
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(e)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
