#!/usr/bin/env python3
"""Author the v3 certification scale-up worlds and the plan reservations file.

Deterministic, pure stdlib: same bytes on every run. One world per NEW family
(1-row families keep the panel cost linear and the audit families complete by
construction). Targets close every audit cell at 120 families/split, which
tolerates one critical-error family per split cell (wilson(1,120) = 4.6% < 5%).
On top of the existing census (train 31 / development 8 / calibration 2 /
test 9) this adds 89 + 112 + 118 + 111 = 430 families.

Writes, relative to the jev root:
  runs/worlds/worlds-v3-train.jsonl    89 worlds
  runs/worlds/worlds-v3-dev.jsonl     112 worlds
  runs/worlds/worlds-v3-cal.jsonl     118 worlds
  runs/worlds/worlds-v3-test.jsonl    111 worlds
  runs/reports/reservations-v3.json    row_id -> split, pins plan() to the target

With --append N the tool instead writes a same-unit train top-up
(runs/worlds/worlds-v3-topup.jsonl + runs/reports/reservations-topup.json,
prefix v3tx-): replacements for train rows the verifier quarantined. Cross-unit
sides are excluded by design because the verify triage showed every t=0.0 flip
(12/12) sat on a cross-unit row whose own evidence was correct.

World schema (jev_data.oracle.validate_world): exactly schema_version, kind,
family_id, template_id, entity, quantity|null, threshold, operator. Numerals are
Arabic-Indic with U+066B decimals (numeral_to_ascii accepts them). Row ids match
tools/oracle_batch.py: <prefix><index:06d> with prefixes v3tr- v3dv- v3cl- v3ts-.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = "scaleup-v3-2026-10"
# (split, file stem, row-id prefix, count)
SPLITS = (("train", "train", "v3tr-", 89), ("development", "dev", "v3dv-", 112),
          ("calibration", "cal", "v3cl-", 118), ("test", "test", "v3ts-", 111))
OPERATORS = ("<", "<=", ">", ">=", "==", "!=")

# dim -> (base unit, small unit, factor, kind); kind selects the render
# money/length/mass: base = S/factor with decimals, small = S
# ratio: base = percent points S, small = S/100 as a two-decimal ratio
DIMS = (
    ("SAR", "halala", 100, "dec"),
    ("m", "cm", 100, "dec"),
    ("kg", "g", 1000, "dec"),
    ("percent", "ratio", 100, "ratio"),
)
ENTITIES = (
    "مقدم الطلب", "الشركة المتقدمة", "الحساب المفوتر", "طلب التمويل",
    "عقد الإيجار", "محفظة الأوراق", "المشروع المقترح", "فاتورة الخدمة",
    "رصيد الاشتراك", "الكمية المشحونة", "الوزن المعلن", "المسافة المقطوعة",
    "نسبة الفائدة", "هامش الربح", "مساحة المستودع", "حمولة الشاحنة",
    "دفعة القسط", "أجرة النقل", "الاستهلاك الشهري", "الحد الائتماني",
    "كلفة الصيانة", "مخزون القطع", "مدة التوريد", "كمية الوقود",
)
_DIGITS = str.maketrans("0123456789.", "٠١٢٣٤٥٦٧٨٩٫")


def arabic(text: str) -> str:
    return text.translate(_DIGITS)


def dec(S: int, factor: int) -> str:
    """Render S/factor as a minimal decimal string in ASCII digits."""
    whole, rem = divmod(S, factor)
    if rem == 0:
        return str(whole)
    width = len(str(factor)) - 1
    return (f"{whole}.{rem:0{width}d}").rstrip("0")


def side(S: int, dim: tuple, which: str) -> dict:
    base, small, factor, kind = dim
    if which == "base":
        if kind == "ratio":
            return {"value": arabic(str(S)), "unit": base}
        return {"value": arabic(dec(S, factor)), "unit": base}
    if kind == "ratio":
        return {"value": arabic(dec(S, factor)), "unit": small}
    return {"value": arabic(str(S)), "unit": small}


def _target(S: int, operator: str, index: int) -> int:
    """Threshold in smallest units: boundary rows sit exactly on S, the rest one
    unit off in the direction the seeded eligible flag demands."""
    boundary = operator in {"<", "<=", ">", ">="} and index % 7 == 6
    if boundary:
        return S  # threshold exactly equal to the stated quantity
    eligible = (index // len(OPERATORS)) % 2 == 0
    if operator in {"<", "<="}:
        T = S + 1 if eligible else S - 1
    elif operator in {">", ">="}:
        T = S - 1 if eligible else S + 1
    elif operator == "==":
        T = S if eligible else S + 1
    else:  # "!="
        T = S + 1 if eligible else S
    if T < 1:
        T = S + 2  # keep small-unit integers positive after the delta
    return T


def world_for(index: int, prefix: str, rng: random.Random, offset: int) -> dict:
    dim = DIMS[index % len(DIMS)]
    operator = OPERATORS[index % len(OPERATORS)]
    entity = ENTITIES[(index + offset) % len(ENTITIES)]
    common = {"schema_version": 1, "kind": "comparison",
              "family_id": prefix + f"fam{index:04d}", "template_id": prefix + f"tpl{index:04d}",
              "entity": entity}
    if index % 13 == 12:  # abstention rows: the required value is not stated
        S = rng.randint(700, 90000)
        return {**common, "quantity": None, "threshold": side(S, dim, "base"),
                "operator": operator}
    S = rng.randint(700, 90000)  # S is the smallest-unit integer; delta = 1
    if dim[3] == "ratio":
        S = rng.randint(5, 95)  # percent points
    T = _target(S, operator, index)
    if index % 3 == 2:  # cross-unit sides: one base, one smallest unit
        left, right = ("base", "small") if (index // 3) % 2 == 0 else ("small", "base")
    else:
        left = right = "small" if (index // 3) % 2 == 1 else "base"
    return {**common, "quantity": side(S, dim, left), "threshold": side(T, dim, right),
            "operator": operator}


def world_topup(index: int, rng: random.Random) -> dict:
    """Replacement profile for quarantined train rows: same-unit sides only.

    The verify triage showed every t=0.0 disagreement (12/12) sat on a cross-unit
    row whose written evidence converted and compared correctly, so replacements
    keep both sides in one unit while operators, boundary rows and abstentions
    keep their usual seeded mix."""
    prefix = "v3tx-"
    dim = DIMS[index % len(DIMS)]
    operator = OPERATORS[index % len(OPERATORS)]
    entity = ENTITIES[(index + 3) % len(ENTITIES)]
    common = {"schema_version": 1, "kind": "comparison",
              "family_id": prefix + f"fam{index:04d}", "template_id": prefix + f"tpl{index:04d}",
              "entity": entity}
    which = "base" if (index // 3) % 2 == 0 else "small"
    if index % 13 == 12:  # abstention rows: the required value is not stated
        S = rng.randint(700, 90000)
        return {**common, "quantity": None, "threshold": side(S, dim, which),
                "operator": operator}
    S = rng.randint(700, 90000)
    if dim[3] == "ratio":
        S = rng.randint(5, 95)  # percent points
    T = _target(S, operator, index)
    return {**common, "quantity": side(S, dim, which), "threshold": side(T, dim, which),
            "operator": operator}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="jev root containing runs/")
    parser.add_argument("--append", type=int, default=0,
                        help="write only an N-row same-unit train top-up (v3tx- prefix)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    (root / "runs/worlds").mkdir(parents=True, exist_ok=True)
    (root / "runs/reports").mkdir(parents=True, exist_ok=True)
    if args.append > 0:
        rng = random.Random(f"{SEED}:topup")
        worlds = [world_topup(index, rng) for index in range(args.append)]
        path = root / "runs/worlds/worlds-v3-topup.jsonl"
        path.write_text("\n".join(json.dumps(world, ensure_ascii=False) for world in worlds)
                        + "\n", encoding="utf-8")
        reservations = {f"v3tx-{index:06d}": "train" for index in range(args.append)}
        out = root / "runs/reports/reservations-topup.json"
        out.write_text(json.dumps(reservations, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
        print(f"wrote {path} ({args.append} same-unit worlds)")
        print(f"wrote {out} ({len(reservations)} reservations)")
        return 0
    reservations: dict[str, str] = {}
    for position, (split, stem, prefix, count) in enumerate(SPLITS):
        rng = random.Random(f"{SEED}:{split}")
        worlds = [world_for(index, prefix, rng, offset=position * 5)
                  for index in range(count)]
        path = root / f"runs/worlds/worlds-v3-{stem}.jsonl"
        path.write_text("\n".join(json.dumps(world, ensure_ascii=False) for world in worlds)
                        + "\n", encoding="utf-8")
        for index in range(count):
            reservations[prefix + f"{index:06d}"] = split
        print(f"wrote {path} ({count} worlds, families {prefix}fam0000..fam{count - 1:04d})")
    out = root / "runs/reports/reservations-v3.json"
    out.write_text(json.dumps(reservations, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"wrote {out} ({len(reservations)} reservations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
