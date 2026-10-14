#!/usr/bin/env python3
"""Author the voice-coverage v1 worlds for every executable oracle rule.

Deterministic, same bytes on every run. Nine rule kinds (comparison,
temporal_window, policy_logic, budget_check, intent_routing, safety_gate,
triage_rubric, readback_match, slot_assertion) cross six rendering styles
(MSA, Egyptian, Levantine, Gulf, code-switch, English) over twelve scenarios
each, one world per NEW family like worlds_v3. Splits step through overlapping
scenario slices (offsets 0/3/6/9) so the union covers every scenario and every
style in every kind.

Per split: train 12 steps -> 648 worlds; development 4 -> 216; calibration
2 -> 108; test 4 -> 216; 1188 total.

Writes, relative to the jev root:
  runs/worlds/worlds-voice-v1-train.jsonl   ids vv1tr-000000..  (648)
  runs/worlds/worlds-voice-v1-dev.jsonl     ids vv1dv-000000..  (216)
  runs/worlds/worlds-voice-v1-cal.jsonl     ids vv1cl-000000..  (108)
  runs/worlds/worlds-voice-v1-test.jsonl    ids vv1ts-000000..  (216)
  runs/reports/reservations-voice-v1.json   row_id -> split

Every generated world is self-checked against its rule at generation time
(validate, primary solve, independent solve, declared expected decision); a
single mismatch aborts the run. Requires the jev_data package importable.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = "voice-v1-2026-10"
STYLES = ("msa", "egyptian", "levantine", "gulf", "code_switch", "english")
SPLITS = (("train", "train", "vv1tr-", 12), ("development", "dev", "vv1dv-", 4),
          ("calibration", "cal", "vv1cl-", 2), ("test", "test", "vv1ts-", 4))
OFFSETS = {"train": 0, "development": 3, "calibration": 6, "test": 9}
KINDS = ("comparison", "temporal_window", "policy_logic", "budget_check",
         "intent_routing", "safety_gate", "triage_rubric", "readback_match",
         "slot_assertion")


def en_style(style: str) -> bool:
    return style == "english"


def pick(style: str, ar: str, en: str) -> str:
    return en if en_style(style) else ar


# ---------------------------------------------------------------- comparison

DIMS = (("SAR", "halala", 100, "dec"), ("m", "cm", 100, "dec"),
        ("kg", "g", 1000, "dec"), ("percent", "ratio", 100, "ratio"))
# (dim, op, mode, eligible, quantity side, threshold side)
COMPARISON_SCENARIOS = (
    (0, "<", "same", True, "base", "base"),
    (0, ">=", "same", False, "small", "small"),
    (1, "<=", "cross", True, "small", "base"),
    (2, "==", "cross", False, "base", "small"),
    (3, ">", "same", True, "base", "base"),
    (0, "<=", "boundary", True, "base", "base"),
    (0, "<", "boundary", False, "small", "small"),
    (0, "<", "abstain", False, "base", "base"),
    (1, "!=", "same", True, "small", "small"),
    (2, ">", "same", False, "base", "base"),
    (3, "<=", "boundary", True, "base", "base"),
    (3, ">=", "abstain", False, "base", "base"),
)
COMPARISON_ENTITIES = (
    ("طلب التمويل", "the financing request"), ("عقد الإيجار", "the lease contract"),
    ("الفاتورة الشهرية", "the monthly bill"), ("محفظة الأوراق", "the securities portfolio"),
    ("دفعة القسط", "the installment payment"), ("الحد الائتماني", "the credit limit"),
    ("الكمية المشحونة", "the shipped quantity"), ("الوزن المعلن", "the declared weight"),
    ("نسبة الفائدة", "the interest rate"), ("هامش الربح", "the profit margin"),
    ("رصيد الاشتراك", "the subscription balance"), ("كلفة الصيانة", "the maintenance cost"),
)


def dec(S: int, factor: int) -> str:
    whole, rem = divmod(S, factor)
    if rem == 0:
        return str(whole)
    return (f"{whole}.{rem:0{len(str(factor)) - 1}d}").rstrip("0")


def comparison_side(S: int, dim: tuple, which: str) -> dict:
    base, small, factor, kind = dim
    if which == "base":
        return {"value": str(S) if kind == "ratio" else dec(S, factor), "unit": base}
    return {"value": dec(S, factor) if kind == "ratio" else str(S), "unit": small}


def comparison_builder(si: int, style: str, rng: random.Random):
    dim_index, op, mode, eligible, left, right = COMPARISON_SCENARIOS[si]
    dim = DIMS[dim_index]
    entity = pick(style, *COMPARISON_ENTITIES[si])
    S = rng.randint(5, 95) if dim[3] == "ratio" else rng.randint(700, 90000)
    if mode == "abstain":
        T = rng.randint(5, 95) if dim[3] == "ratio" else rng.randint(700, 90000)
        return ({"rendering": style, "entity": entity, "quantity": None,
                 "threshold": comparison_side(T, dim, right), "operator": op},
                "insufficient")
    if mode == "boundary":
        T = S
        expected = "eligible" if op in ("<=", ">=", "==") else "ineligible"
    else:
        if op in ("<", "<="):
            T = S + 1 if eligible else max(S - 1, 1)
        elif op in (">", ">="):
            T = max(S - 1, 1) if eligible else S + 1
        elif op == "==":
            T = S if eligible else S + 1
        else:
            T = S + 1 if eligible else S
        expected = "eligible" if eligible else "ineligible"
    return ({"rendering": style, "entity": entity,
             "quantity": comparison_side(S, dim, left),
             "threshold": comparison_side(T, dim, right), "operator": op}, expected)


# ------------------------------------------------------------ temporal_window

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def dated(si: int, weekend: list[str], want_in: bool) -> str:
    start = date(2026, 11, 1) + timedelta(days=si % 7)
    indices = {WEEKDAYS.index(name) for name in weekend}
    for offset in range(60):
        candidate = start + timedelta(days=offset)
        if (candidate.weekday() in indices) == want_in:
            return candidate.isoformat()
    raise AssertionError("unreachable weekday search")


def temp_scen(si: int, style: str, rng: random.Random):
    """Return (body-without-common, expected)."""
    mon_sat = ["friday", "saturday"]
    if si == 0:
        return ({"scenario": "business_hours", "date": dated(si, mon_sat, False),
                 "opens": "09:00", "closes": "17:00", "event": "11:30",
                 "duration_minutes": 90, "weekend": mon_sat}, "eligible")
    if si == 1:
        return ({"scenario": "business_hours", "date": dated(si, mon_sat, True),
                 "opens": "09:00", "closes": "17:00", "event": "11:30",
                 "duration_minutes": 90, "weekend": mon_sat}, "ineligible")
    if si == 2:
        return ({"scenario": "business_hours", "date": dated(si, mon_sat, False),
                 "opens": "22:00", "closes": "02:00", "event": "23:30",
                 "duration_minutes": 60, "weekend": mon_sat}, "eligible")
    if si == 3:
        return ({"scenario": "business_hours", "date": dated(si, mon_sat, False),
                 "opens": "09:00", "closes": "17:00", "event": "16:30",
                 "duration_minutes": 60, "weekend": mon_sat}, "ineligible")
    if si == 4:
        return ({"scenario": "sla", "opened": "2026-11-02T09:15",
                 "replied": "2026-11-02T13:45", "sla_minutes": 480}, "true")
    if si == 5:
        return ({"scenario": "sla", "opened": "2026-11-02T09:15",
                 "replied": "2026-11-03T10:00", "sla_minutes": 480}, "false")
    if si == 6:
        return ({"scenario": "business_hours", "date": dated(si, mon_sat, False),
                 "opens": "09:00", "closes": "17:00", "event": None,
                 "duration_minutes": 45, "weekend": mon_sat}, "insufficient")
    if si == 7:
        return ({"scenario": "sla", "opened": "2026-11-02T20:00",
                 "replied": "2026-11-04T19:59", "sla_minutes": 2880}, "true")
    if si == 8:
        return ({"scenario": "business_hours", "date": dated(si, mon_sat, False),
                 "opens": "09:00", "closes": "17:00", "event": "08:15",
                 "duration_minutes": 30, "weekend": mon_sat}, "ineligible")
    if si == 9:
        sunday_only = ["sunday"]
        return ({"scenario": "business_hours", "date": dated(si, sunday_only, False),
                 "opens": "10:00", "closes": "18:00", "event": "12:00",
                 "duration_minutes": 120, "weekend": sunday_only}, "eligible")
    if si == 10:
        return ({"scenario": "sla", "opened": "2026-11-02T09:15",
                 "replied": "2026-11-02T13:46", "sla_minutes": 270}, "false")
    return ({"scenario": "business_hours", "date": dated(si, mon_sat, False),
             "opens": "09:00", "closes": "17:00", "event": "09:00",
             "duration_minutes": 480, "weekend": mon_sat}, "eligible")


def temporal_builder(si: int, style: str, rng: random.Random):
    body, expected = temp_scen(si, style, rng)
    return {"rendering": style, **body}, expected


# --------------------------------------------------------------- policy_logic

POLICY_SCENARIOS = (
    ("طلب استرداد مشترٍ", "a buyer refund request", "all",
     [("purchase_days", "أيام الشراء", "purchase days", "<=", 14, "rand14"),
      ("opened", "عبوة مفتوحة", "package opened", "==", False, False),
      ("receipt", "إيصال الشراء", "purchase receipt", "==", True, True)], "eligible"),
    ("طلب استرداد مشترٍ", "a buyer refund request", "all",
     [("purchase_days", "أيام الشراء", "purchase days", "<=", 14, "rand25"),
      ("opened", "عبوة مفتوحة", "package opened", "==", False, True),
      ("receipt", "إيصال الشراء", "purchase receipt", "==", True, True)], "ineligible"),
    ("طلب استرداد مشترٍ", "a buyer refund request", "all",
     [("purchase_days", "أيام الشراء", "purchase days", "<=", 14, 7),
      ("opened", "عبوة مفتوحة", "package opened", "==", False, False),
      ("receipt", "إيصال الشراء", "purchase receipt", "==", True, None)], "insufficient"),
    ("طلب استرداد مشترٍ", "a buyer refund request", "all",
     [("purchase_days", "أيام الشراء", "purchase days", "<=", 14, 6),
      ("opened", "عبوة مفتوحة", "package opened", "==", False, True),
      ("receipt", "إيصال الشراء", "purchase receipt", "==", True, None)], "ineligible"),
    ("طلب تسجيل طالب", "a student enrollment request", "any",
     [("age", "العمر", "age", ">=", 18, 21),
      ("guardian_consent", "موافقة ولي الأمر", "guardian consent", "==", True, False)],
     "eligible"),
    ("طلب تسجيل طالب", "a student enrollment request", "any",
     [("age", "العمر", "age", ">=", 18, 16),
      ("guardian_consent", "موافقة ولي الأمر", "guardian consent", "==", True, False)],
     "ineligible"),
    ("طلب تسجيل طالب", "a student enrollment request", "any",
     [("age", "العمر", "age", ">=", 18, None),
      ("guardian_consent", "موافقة ولي الأمر", "guardian consent", "==", True, False)],
     "insufficient"),
    ("طلب تغطية تأمينية", "an insurance coverage request", "all",
     [("property_age", "عمر الممتلكات", "property age", "<=", 30, 12),
      ("no_claims", "سجل بلا مطالبات", "claims-free record", "==", True, True),
      ("policy_active", "وثيقة سارية", "active policy", "==", True, True)], "eligible"),
    ("طلب تغطية تأمينية", "an insurance coverage request", "all",
     [("property_age", "عمر الممتلكات", "property age", "<=", 30, 45),
      ("no_claims", "سجل بلا مطالبات", "claims-free record", "==", True, None),
      ("policy_active", "وثيقة سارية", "active policy", "==", True, False)], "ineligible"),
    ("طلب ضمان قرض", "a loan guarantee request", "any",
     [("monthly_income", "الدخل الشهري", "monthly income", ">=", 5000, None),
      ("guarantor", "وجود كفيل", "a guarantor", "==", True, True)], "eligible"),
    ("طلب ضمان قرض", "a loan guarantee request", "any",
     [("monthly_income", "الدخل الشهري", "monthly income", ">=", 5000, None),
      ("guarantor", "وجود كفيل", "a guarantor", "==", True, False)], "insufficient"),
    ("طلب استرداد مشترٍ", "a buyer refund request", "all",
     [("purchase_days", "أيام الشراء", "purchase days", "<=", 14, 14),
      ("opened", "عبوة مفتوحة", "package opened", "==", False, False),
      ("receipt", "إيصال الشراء", "purchase receipt", "==", True, True)], "eligible"),
)


def policy_builder(si: int, style: str, rng: random.Random):
    entity_ar, entity_en, combine, spec, expected = POLICY_SCENARIOS[si]
    clauses, labels, facts = [], {}, {}
    for name, label_ar, label_en, op, value, provided in spec:
        clauses.append({"fact": name, "op": op, "value": value})
        labels[name] = {"ar": label_ar, "en": label_en}
        if provided == "rand14":
            provided = rng.randint(3, 14)
        elif provided == "rand25":
            provided = rng.randint(15, 25)
        if provided is not None:
            facts[name] = provided
    return {"rendering": style, "entity": pick(style, entity_ar, entity_en),
            "combine": combine, "clauses": clauses, "labels": labels, "facts": facts}, expected


# ---------------------------------------------------------------- budget_check

BUDGET_ENTITIES = (
    ("طلب صرف إضافي", "an additional spend request"),
    ("أمر الشراء الجديد", "the new purchase order"),
    ("خطة التقسيط المقترحة", "the proposed installment plan"),
    ("فاتورة الصيانة", "the maintenance invoice"),
)


def budget_builder(si: int, style: str, rng: random.Random):
    unit = ({**{"ar": "ريال", "en": "riyals"}})
    entity = pick(style, *BUDGET_ENTITIES[si % len(BUDGET_ENTITIES)])
    base = {"rendering": style, "entity": entity, "unit": unit}
    total = rng.randint(5000, 50000)
    spent = rng.randint(500, total // 2)
    if si == 0:
        return {**base, "scenario": "single", "total": total, "spent": spent,
                "requested": total - spent}, "eligible"
    if si == 1:
        return {**base, "scenario": "single", "total": total, "spent": spent,
                "requested": total - spent + 1}, "ineligible"
    if si == 2:
        return {**base, "scenario": "single", "total": total, "spent": None,
                "requested": 250}, "insufficient"
    if si == 3:
        return {**base, "scenario": "single", "total": total, "spent": spent,
                "requested": None}, "insufficient"
    months = rng.randint(3, 24)
    installment = rng.randint(50, 900)
    if si == 4:
        return {**base, "scenario": "installments", "price": installment * months,
                "installment": installment, "months": months}, "eligible"
    if si == 5:
        return {**base, "scenario": "installments", "price": installment * months + 1,
                "installment": installment, "months": months}, "ineligible"
    if si == 6:
        return {**base, "scenario": "installments", "price": installment * months,
                "installment": installment, "months": None}, "insufficient"
    if si == 7:
        return {**base, "scenario": "installments",
                "price": max(installment * months - rng.randint(1, max(installment - 1, 1)), 1),
                "installment": installment, "months": months}, "eligible"
    if si == 8:
        return {**base, "scenario": "single", "total": total, "spent": 0,
                "requested": rng.randint(1, 100)}, "eligible"
    if si == 9:
        return {**base, "scenario": "single", "total": total, "spent": total,
                "requested": 1}, "ineligible"
    if si == 10:
        return {**base, "scenario": "installments", "price": installment * months,
                "installment": None, "months": months}, "insufficient"
    return {**base, "scenario": "single", "total": total, "spent": rng.randint(0, total),
            "requested": 0}, "eligible"


# --------------------------------------------------------------- intent_routing

# domain -> (label_ar, label_en, [(intent_id, ar_label, en_label,
#                                 [(ar_word, en_word) x3]) x4], [(ar,en) fillers x3])
DOMAINS = {
    "banking": (
        "الخدمات المصرفية", "banking services",
        [("bk_transfer", "تحويل الأموال", "money transfer",
          [("تحويل", "transfer"), ("حوالة", "remittance"), ("بنك", "bank")]),
         ("bk_card", "إيقاف البطاقة", "blocking a card",
          [("إيقاف", "block"), ("بطاقة", "card"), ("ضائعة", "lost")]),
         ("bk_statement", "كشف الحساب", "account statement",
          [("كشف", "statement"), ("حساب", "account"), ("مصاريف", "charges")]),
         ("bk_loan", "طلب قرض", "a loan request",
          [("قرض", "loan"), ("تمويل", "financing"), ("أقساط", "installments")])],
        [("استفسار", "inquiry"), ("عاجل", "urgent"), ("المساعدة", "help")]),
    "telecom": (
        "خدمات الاتصالات", "telecom services",
        [("tc_plan", "ترقية الباقة", "plan upgrade",
          [("باقة", "plan"), ("ترقية", "upgrade"), ("إنترنت", "internet")]),
         ("tc_roaming", "التجوال الدولي", "international roaming",
          [("تجوال", "roaming"), ("سفر", "travel"), ("خارجي", "international")]),
         ("tc_bill", "اعتراض على فاتورة", "a bill dispute",
          [("فاتورة", "bill"), ("اعتراض", "dispute"), ("مبلغ", "amount")]),
         ("tc_sim", "بدل شريحة", "replacing a SIM",
          [("شريحة", "sim"), ("بدل", "replace"), ("فقدت", "lost")])],
        [("خدمة", "service"), ("سؤال", "question"), ("دعم", "support")]),
    "travel": (
        "خدمات السفر", "travel services",
        [("tv_booking", "تغيير الحجز", "changing a booking",
          [("حجز", "booking"), ("تغيير", "change"), ("رحلة", "flight")]),
         ("tv_baggage", "الأمتعة المفقودة", "lost baggage",
          [("أمتعة", "baggage"), ("مفقودة", "lost"), ("وزن", "weight")]),
         ("tv_refund", "استرداد التذكرة", "a ticket refund",
          [("استرداد", "refund"), ("إلغاء", "cancellation"), ("تذكرة", "ticket")]),
         ("tv_checkin", "تسجيل الوصول", "check-in",
          [("تسجيل", "check-in"), ("مقعد", "seat"), ("صعود", "boarding")])],
        [("مساعدة", "assistance"), ("معلومات", "information"), ("طارئ", "emergency")]),
    "healthcare": (
        "الخدمات الصحية", "healthcare services",
        [("hc_appointment", "حجز موعد", "scheduling an appointment",
          [("موعد", "appointment"), ("عيادة", "clinic"), ("حجز", "schedule")]),
         ("hc_refill", "تجديد وصفة", "a prescription refill",
          [("وصفة", "prescription"), ("صيدلية", "pharmacy"), ("تجديد", "refill")]),
         ("hc_claim", "مطالبة تأمين", "an insurance claim",
          [("مطالبة", "claim"), ("تأمين", "insurance"), ("تعويض", "reimbursement")]),
         ("hc_results", "نتائج التحاليل", "test results",
          [("تحاليل", "results"), ("تقرير", "report"), ("مختبر", "laboratory")])],
        [("دواء", "medication"), ("استفسار", "inquiry"), ("متابعة", "follow-up")]),
    "retail": (
        "خدمات المتاجر", "retail services",
        [("rt_track", "تتبع الشحنة", "shipment tracking",
          [("تتبع", "track"), ("شحنة", "shipment"), ("طلب", "order")]),
         ("rt_return", "إرجاع منتج", "returning a product",
          [("إرجاع", "return"), ("منتج", "product"), ("استبدال", "replacement")]),
         ("rt_warranty", "مطالبة ضمان", "a warranty claim",
          [("ضمان", "warranty"), ("إصلاح", "repair"), ("عيب", "defect")]),
         ("rt_hours", "أوقات الفروع", "branch hours",
          [("أوقات", "hours"), ("فرع", "branch"), ("متجر", "store")])],
        [("خصم", "discount"), ("عرض", "offer"), ("شكراً", "thanks")]),
    "government": (
        "الخدمات الحكومية", "government services",
        [("gv_id", "تجديد الهوية", "an ID renewal",
          [("هوية", "ID"), ("تجديد", "renewal"), ("وطنية", "national")]),
         ("gv_fines", "سداد المخالفات", "paying fines",
          [("مخالفة", "fine"), ("سداد", "payment"), ("مرور", "traffic")]),
         ("gv_permit", "إصدار تصريح", "issuing a permit",
          [("تصريح", "permit"), ("نشاط", "business"), ("بلدية", "municipality")]),
         ("gv_appointment", "حجز موعد إداري", "an administrative appointment",
          [("موعد", "appointment"), ("إدارة", "authority"), ("خدمة", "service")])],
        [("استعلام", "inquiry"), ("إجراء", "procedure"), ("مساعدة", "assistance")]),
}
ROUTING_SCENARIOS = (
    ("banking", "winner"), ("telecom", "tie"), ("travel", "zero"),
    ("healthcare", "winner"), ("retail", "partial"), ("government", "tie"),
    ("banking", "partial"), ("telecom", "zero"), ("travel", "winner"),
    ("healthcare", "tie"), ("retail", "winner"), ("government", "partial"),
)


def domain_intents(domain: str):
    """Return (intents, keyword lexicon, filler lexicon). Fillers are only
    merged into the world lexicon for tokens actually used in the utterance."""
    _, _, intents, fillers = DOMAINS[domain]
    built, lexicon = [], {}
    for intent_index, (intent_id, ar_label, en_label, words) in enumerate(intents):
        keywords = []
        for word_index, (ar_word, en_word) in enumerate(words):
            token = f"t{intent_index}{word_index}"
            keywords.append(token)
            lexicon[token] = {"ar": ar_word, "en": en_word}
        built.append({"id": intent_id, "label": {"ar": ar_label, "en": en_label},
                      "keywords": keywords})
    filler_map = {f"f{index}": {"ar": ar_word, "en": en_word}
                  for index, (ar_word, en_word) in enumerate(fillers)}
    return built, lexicon, filler_map


def routing_builder(si: int, style: str, rng: random.Random):
    domain, case = ROUTING_SCENARIOS[si]
    label_ar, label_en, _, _ = DOMAINS[domain]
    intents, lexicon, fillers = domain_intents(domain)
    if case == "winner":
        utterance = intents[0]["keywords"] + [intents[1]["keywords"][0]]
        expected = intents[0]["id"]
    elif case == "tie":
        utterance = intents[0]["keywords"][:2] + intents[1]["keywords"][:2]
        expected = intents[0]["id"]
    elif case == "zero":
        utterance = ["f0", "f1", "f2"]
        expected = "other"
    else:
        utterance = [intents[0]["keywords"][0]] + intents[2]["keywords"][:2] + ["f0"]
        expected = intents[2]["id"]
    for token in utterance:
        if token in fillers:
            lexicon[token] = fillers[token]
    return {"rendering": style, "domain": {"ar": label_ar, "en": label_en},
            "intents": intents, "utterance": utterance, "lexicon": lexicon}, expected


# ------------------------------------------------------------------ safety_gate

# (request_ar, request_en, facts, [(outcome, [(fact, op, value) xN]) xR], default, expected)
SAFETY_SCENARIOS = (
    ("سؤال عن جرعة دواء", "a question about a medication dose",
     {"consent": False},
     [("block", [("consent", "==", False)]),
      ("allow", [("consent", "==", True), ("professional", "==", True)])],
     "escalate", "block"),
    ("سؤال عن جرعة دواء", "a question about a medication dose",
     {"consent": True, "professional": True},
     [("block", [("consent", "==", False)]),
      ("allow", [("consent", "==", True), ("professional", "==", True)])],
     "escalate", "allow"),
    ("تحويل مبلغ من الحساب", "a transfer of funds",
     {"amount": 350},
     [("allow", [("amount", "<=", 1000)]),
      ("escalate", [("amount", ">", 1000)])],
     "block", "allow"),
    ("تحويل مبلغ من الحساب", "a transfer of funds",
     {"amount": 1500},
     [("allow", [("amount", "<=", 1000)]),
      ("escalate", [("amount", ">", 1000)])],
     "block", "escalate"),
    ("استعادة حساب مسروق", "recovering a hijacked account",
     {"verified": False},
     [("block", [("verified", "==", False)]),
      ("allow", [("verified", "==", True), ("second_factor", "==", True)])],
     "escalate", "block"),
    ("رسالة تهديد", "a threatening message",
     {"risk_level": 3},
     [("escalate", [("risk_level", ">=", 1)])],
     "allow", "escalate"),
    ("طلب خدمة لقاصر", "a service request for a minor",
     {"age": 15, "third_party": True},
     [("block", [("age", "<", 18), ("third_party", "==", True)]),
      ("allow", [("age", ">=", 18)])],
     "escalate", "block"),
    ("طلب تجديد وصفة طبية", "a prescription refill request",
     {"prescriber": True},
     [("allow", [("prescriber", "==", True), ("dose_known", "==", True)]),
      ("escalate", [("dose_known", "==", False)])],
     "block", "escalate"),
    ("استفسار عام عن الخدمة", "a general service inquiry",
     {"age": 25},
     [("block", [("age", "<", 18)])],
     "escalate", "escalate"),
    ("طلب استرداد كبير", "a large refund request",
     {"amount": 100},
     [("block", [("amount", ">", 5000), ("verified", "==", True)]),
      ("allow", [("amount", "<=", 5000)])],
     "escalate", "allow"),
    ("تعديل بيانات الحساب", "editing account details",
     {"balance": 70},
     [("allow", [("balance", ">=", 100)]),
      ("block", [("balance", ">=", 50)])],
     "escalate", "block"),
    ("طلب خدمة بدون بيانات", "a service request without details",
     {},
     [("block", [("age", "<", 18)]),
      ("allow", [("age", ">=", 18)])],
     "escalate", "escalate"),
)
SAFETY_FACT_LABELS = {
    "consent": ("موافقة صريحة", "explicit consent"),
    "professional": ("مراجعة مختص", "review by a professional"),
    "amount": ("المبلغ", "the amount"),
    "verified": ("توثيق الهوية", "identity verification"),
    "second_factor": ("عامل تحقق ثانٍ", "a second factor"),
    "risk_level": ("مستوى الخطورة", "the risk level"),
    "age": ("العمر", "the age"),
    "third_party": ("طرف آخر", "a third party"),
    "prescriber": ("وصفة من طبيب", "a prescriber's order"),
    "dose_known": ("جرعة محددة", "a stated dose"),
    "balance": ("الرصيد", "the balance"),
}


def safety_builder(si: int, style: str, rng: random.Random):
    request_ar, request_en, facts, rules, default, expected = SAFETY_SCENARIOS[si]
    labels = {}
    for rule in rules:
        for fact, _, _ in rule[1]:
            labels[fact] = {"ar": SAFETY_FACT_LABELS[fact][0], "en": SAFETY_FACT_LABELS[fact][1]}
    return {"rendering": style, "request": {"ar": request_ar, "en": request_en},
            "facts": facts, "fact_labels": labels,
            "rules": [{"when": [{"fact": fact, "op": op, "value": value}
                                for fact, op, value in clauses], "outcome": outcome}
                      for outcome, clauses in rules],
            "default": default}, expected


# ----------------------------------------------------------------- triage_rubric

TRIAGE_SCENARIOS = (
    (3, 0, 0, [0, 1], "0"),
    (3, 3, 9, [0, 9], "1"),
    (4, 2, 6, [0, 6, 7], "1"),
    (4, 2, 6, [0, 7, 8], "0"),
    (5, 3, 10, [0, 5, 10], "2"),
    (5, 4, 12, [0, 6, 12, 13], "2"),
    (6, 3, 8, [0, 9, 10, 11], "0"),
    (6, 5, 15, [0, 3, 6, 12, 15], "4"),
    (6, 6, 24, [0, 4, 8, 12, 18, 24], "5"),
    (7, 4, 11, [0, 5, 11, 12], "2"),
    (8, 5, 14, [0, 3, 8, 15, 22], "2"),
    (10, 7, 20, [0, 2, 4, 6, 8, 10, 12, 14, 16, 20], "9"),
)
TRIAGE_ITEMS = (
    ("impact", "أثر على العمل", "business impact"),
    ("users", "عدد المتأثرين", "users affected"),
    ("workaround", "توفر حل بديل", "workaround available"),
    ("sensitive", "بيانات حساسة", "sensitive data"),
    ("deadline", "موعد نهائي قريب", "an upcoming deadline"),
    ("recurrence", "تكرار المشكلة", "the problem recurring"),
    ("revenue", "خسارة إيرادات", "lost revenue"),
    ("compliance", "أثر تنظيمي", "a compliance impact"),
    ("escalated", "تصعيد سابق", "a prior escalation"),
    ("vip", "عميل رئيسي", "a key account"),
    ("data_loss", "فقدان بيانات", "data loss"),
    ("outage", "انقطاع كامل", "a full outage"),
)
ARABIC_LEVELS = ("ضعيف", "مقبول", "جيد", "جيد جداً", "ممتاز",
                 "حرج", "حرج جداً", "طارئ", "طارئ جداً", "قصوى")
ENGLISH_LEVELS = ("very low", "low", "moderate", "elevated", "high",
                  "very high", "severe", "critical", "very critical", "maximum")
TRIAGE_ENTITIES = (
    ("تذكرة دعم عاجلة", "an urgent support ticket"),
    ("بلاغ عطل تقني", "a technical fault report"),
    ("شكوى خدمة", "a service complaint"),
    ("طلب صيانة", "a maintenance request"),
)


def triage_builder(si: int, style: str, rng: random.Random):
    item_count, met_count, target, thresholds, expected = TRIAGE_SCENARIOS[si]
    items, extras = [], target - met_count
    for index in range(item_count):
        _, label_ar, label_en = TRIAGE_ITEMS[index]
        if index < met_count:
            take = min(4, extras)
            points = 1 + take
            extras -= take
            met = True
        else:
            points = rng.randint(1, 5)
            met = False
        items.append({"id": f"it{index}", "label": {"ar": label_ar, "en": label_en},
                      "points": points, "met": met})
    assert extras == 0, "met points must sum exactly to the target"
    levels = [{"label": {"ar": ARABIC_LEVELS[i], "en": ENGLISH_LEVELS[i]}}
              for i in range(len(thresholds))]
    entity = pick(style, *TRIAGE_ENTITIES[si % len(TRIAGE_ENTITIES)])
    return {"rendering": style, "entity": entity, "items": items,
            "levels": levels, "thresholds": thresholds}, expected


# --------------------------------------------------------------- readback_match

READBACK_SCENARIOS = (
    ({"value": "120", "unit": "SAR"}, {"value": "120", "unit": "SAR"}, "true"),
    ({"value": "120", "unit": "SAR"}, {"value": "130", "unit": "SAR"}, "false"),
    ({"value": "1.5", "unit": "kg"}, {"value": "1500", "unit": "g"}, "true"),
    ({"value": "120", "unit": "SAR"}, None, "false"),
    ({"value": "3", "unit": "m"}, {"value": "3", "unit": "kg"}, "false"),
    ({"value": "0.5", "unit": "SAR"}, {"value": "50", "unit": "halala"}, "true"),
    ({"value": "2", "unit": "hour"}, {"value": "120", "unit": "minute"}, "true"),
    ({"value": "90", "unit": "minute"}, {"value": "2", "unit": "hour"}, "false"),
    ({"value": "0.5", "unit": "minute"}, {"value": "30", "unit": "second"}, "true"),
    ({"value": "10500", "unit": "g"}, {"value": "10.5", "unit": "kg"}, "true"),
    ({"value": "10.25", "unit": "SAR"}, {"value": "10.2", "unit": "SAR"}, "false"),
    ({"value": "100", "unit": "halala"}, {"value": "2", "unit": "SAR"}, "false"),
)


def readback_builder(si: int, style: str, rng: random.Random):
    expected_q, readback_q, expected = READBACK_SCENARIOS[si]
    return {"rendering": style, "expected": dict(expected_q),
            "readback": dict(readback_q) if readback_q else None}, expected


# ---------------------------------------------------------------- slot_assertion

SLOT_LABELS = {
    "name": ("الاسم", "name"),
    "phone": ("الجوال", "phone"),
    "address": ("العنوان", "address"),
    "city": ("المدينة", "city"),
    "email": ("البريد الإلكتروني", "email"),
    "amount": ("المبلغ", "amount"),
    "date": ("التاريخ", "date"),
    "reference": ("المرجع", "reference"),
}
# (form [(field, value)], claim (field, value|None), expected)
SLOT_SCENARIOS = (
    ([("name", "سارة العتيبي")], ("name", "سارة العتيبي"), "true"),
    ([("name", "سارة العتيبي")], ("name", "نورة العتيبي"), "false"),
    ([("phone", "0501234567")], ("address", "الرياض"), "false"),
    ([("phone", "0501234567")], ("address", None), "true"),
    ([("phone", "0501234567")], ("phone", None), "false"),
    ([("name", "خالد المطيري"), ("city", "جدة")], ("city", "جدة"), "true"),
    ([("name", "أحمد")], ("name", "احمد"), "false"),
    ([("email", "Ali@Example.com")], ("email", "ali@example.com"), "false"),
    ([("amount", "250.00")], ("amount", "250.00"), "true"),
    ([("date", "2026-11-15")], ("date", "2026-11-16"), "false"),
    ([("name", "منى"), ("phone", "0555111222"), ("city", "الدمام")],
     ("phone", "0555111222"), "true"),
    ([("city", "أبها")], ("city", None), "false"),
)


def slots_builder(si: int, style: str, rng: random.Random):
    form, claim, expected = SLOT_SCENARIOS[si]
    references = {field for field, _ in form} | {claim[0]}
    labels = {field: {"ar": SLOT_LABELS[field][0], "en": SLOT_LABELS[field][1]}
              for field in sorted(references)}
    return {"rendering": style,
            "form": [{"field": field, "value": value} for field, value in form],
            "claim": {"field": claim[0], "value": claim[1]},
            "field_labels": labels}, expected


BUILDERS = {
    "comparison": comparison_builder,
    "temporal_window": temporal_builder,
    "policy_logic": policy_builder,
    "budget_check": budget_builder,
    "intent_routing": routing_builder,
    "safety_gate": safety_builder,
    "triage_rubric": triage_builder,
    "readback_match": readback_builder,
    "slot_assertion": slots_builder,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="jev root containing runs/")
    args = parser.parse_args(argv)
    root = Path(args.root)
    (root / "runs/worlds").mkdir(parents=True, exist_ok=True)
    (root / "runs/reports").mkdir(parents=True, exist_ok=True)

    from jev_data.rules import rule_for

    reservations: dict[str, str] = {}
    total = 0
    for split, stem, prefix, steps in SPLITS:
        worlds, failures, coverage = [], [], {}
        for kind in KINDS:
            builder = BUILDERS[kind]
            rng = random.Random(f"{SEED}:{split}:{kind}")
            for step in range(steps):
                scenario = (step + OFFSETS[split]) % 12
                for style in STYLES:
                    index = len(worlds)
                    body, expected = builder(scenario, style, rng)
                    world = {"schema_version": 1, "kind": kind,
                             "family_id": f"{prefix}fam{index:04d}",
                             "template_id": f"{prefix}{kind}-tpl{scenario:02d}",
                             **body}
                    rule = rule_for(world)
                    rule.validate(world)
                    answer = rule.solve(world)
                    independent = rule.independent_solve(world)
                    rule.surface(world)
                    if answer != expected or independent != expected:
                        failures.append({"kind": kind, "scenario": scenario, "style": style,
                                         "expected": expected, "solve": answer,
                                         "independent": independent})
                    coverage[(kind, style)] = coverage.get((kind, style), 0) + 1
                    worlds.append(world)
        if failures:
            for failure in failures[:10]:
                print(f"SELF-CHECK FAILED {failure}")
            print(f"ABORT: {len(failures)} worlds disagree with their declared decision")
            return 2
        path = root / f"runs/worlds/worlds-voice-v1-{stem}.jsonl"
        path.write_text("\n".join(json.dumps(world, ensure_ascii=False) for world in worlds)
                        + "\n", encoding="utf-8")
        for index in range(len(worlds)):
            reservations[prefix + f"{index:06d}"] = split
        total += len(worlds)
        styles_seen = len({style for (_, style) in coverage})
        print(f"wrote {path} ({len(worlds)} worlds; "
              f"{len(KINDS)} kinds x {steps} steps x {styles_seen} styles, self-checked)")
    out = root / "runs/reports/reservations-voice-v1.json"
    out.write_text(json.dumps(reservations, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"wrote {out} ({len(reservations)} reservations; {total} worlds total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
