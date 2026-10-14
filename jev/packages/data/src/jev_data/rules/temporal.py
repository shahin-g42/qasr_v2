"""Temporal window predicates for scheduling, support hours and SLA checks.

business_hours: a daily window (possibly crossing midnight) plus a weekend day
set; the appointment must start inside the window and finish before it closes.
sla: reply deadline arithmetic in exact minutes. The primary solver uses pure
minute integers and a civil-calendar weekday derivation; the independent solver
uses datetime/timedelta arithmetic so the two share no code path.
"""

from __future__ import annotations

import re
from datetime import date as Date
from datetime import datetime, timedelta
from datetime import time as Time

from ..io import fields, integer, require, strings, text, version
from ._shared import (
    attributes,
    decision_options,
    digits,
    join_ar,
    join_en,
    missing_word,
    noised,
    noul_options,
    style_of,
    surface_words,
    variant_of,
)

KIND = "temporal_window"
RULE_VERSION = "temporal-window-v1"
TASK = "temporal_policy"

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
ARABIC_WEEKDAYS = {"monday": "الاثنين", "tuesday": "الثلاثاء", "wednesday": "الأربعاء",
                   "thursday": "الخميس", "friday": "الجمعة", "saturday": "السبت",
                   "sunday": "الأحد"}
ENGLISH_WEEKDAYS = {"monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday",
                    "thursday": "Thursday", "friday": "Friday", "saturday": "Saturday",
                    "sunday": "Sunday"}
ARABIC_MONTHS = ("يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس",
                 "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر")
ENGLISH_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
                  "August", "September", "October", "November", "December")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_TIME = re.compile(r"([01][0-9]|2[0-3]):[0-5][0-9]")
_STAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T([01][0-9]|2[0-3]):[0-5][0-9]")


def _calendar_date(value: object) -> Date:
    text(value)
    require(_DATE.fullmatch(value) is not None, "date must be YYYY-MM-DD")
    try:
        return Date.fromisoformat(value)
    except ValueError as exc:
        raise require(False, "invalid calendar date") from exc


def _clock(value: object) -> Time:
    text(value)
    require(_TIME.fullmatch(value) is not None, "time must be HH:MM")
    return Time.fromisoformat(value)


def _stamp(value: object) -> datetime:
    text(value)
    require(_STAMP.fullmatch(value) is not None, "timestamp must be YYYY-MM-DDTHH:MM")
    return datetime.fromisoformat(value)


def _minutes(value: str) -> int:
    hours, _, minutes = value.partition(":")
    return int(hours) * 60 + int(minutes)


def _days_from_civil(year: int, month: int, day: int) -> int:
    """Howard Hinnant's civil calendar; days since 1970-01-01 without datetime."""
    year -= month <= 2
    era = (year if year >= 0 else year - 399) // 400
    year_of_era = year - era * 400
    day_of_year = (153 * (month + (-3 if month > 2 else 9)) + 2) // 5 + day - 1
    day_of_era = year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year
    return era * 146097 + day_of_era - 719468


def _weekday_index(value: Date) -> int:
    """Monday-based weekday derived by pure integer arithmetic, not Date.weekday."""
    days = _days_from_civil(value.year, value.month, value.day)
    return (days + 3) % 7


def validate(world: dict) -> None:
    require(type(world) is dict, "expected an object")
    scenario = world.get("scenario")
    require(scenario in ("business_hours", "sla"), "unknown temporal scenario")
    common = {"schema_version", "kind", "family_id", "template_id", "scenario", "rendering"}
    if scenario == "business_hours":
        fields(world, common | {"date", "opens", "closes", "event", "duration_minutes",
                                "weekend"})
    else:
        fields(world, common | {"opened", "replied", "sla_minutes"})
    version(world["schema_version"])
    require(world["kind"] == KIND, "unsupported symbolic world")
    for key in ("family_id", "template_id"):
        text(world[key])
    variant_of(world, KIND, optional=False)
    if scenario == "business_hours":
        _calendar_date(world["date"])
        _clock(world["opens"])
        _clock(world["closes"])
        require(world["event"] is None or _TIME.fullmatch(world["event"] or "") is not None,
                "event must be HH:MM or null")
        integer(world["duration_minutes"], 1, 1440)
        weekend = strings(world["weekend"])
        require(set(weekend) <= set(WEEKDAYS), "unknown weekday")
        require(bool(weekend), "weekend day list cannot be empty")
    else:
        _stamp(world["opened"])
        _stamp(world["replied"])
        integer(world["sla_minutes"], 1, 20160)
        require(datetime.fromisoformat(world["replied"]) >= datetime.fromisoformat(
            world["opened"]), "reply cannot precede the request")


def solve(world: dict) -> str:
    validate(world)
    if world["scenario"] == "sla":
        waited = (_minutes(world["replied"][11:]) - _minutes(world["opened"][11:])
                  + 1440 * (_days_from_civil(*map(int, world["replied"][:10].split("-")))
                            - _days_from_civil(*map(int, world["opened"][:10].split("-")))))
        return "true" if waited <= world["sla_minutes"] else "false"
    if world["event"] is None:
        return "insufficient"
    if _weekday_index(_calendar_date(world["date"])) in {
            WEEKDAYS.index(day) for day in world["weekend"]}:
        return "ineligible"
    opening = _minutes(world["opens"])
    closing = _minutes(world["closes"])
    if closing <= opening:
        closing += 1440
    start = _minutes(world["event"])
    for candidate in (start, start + 1440):
        if opening <= candidate and candidate + world["duration_minutes"] <= closing:
            return "eligible"
    return "ineligible"


def independent_solve(world: dict) -> str:
    validate(world)
    if world["scenario"] == "sla":
        waited = (datetime.fromisoformat(world["replied"])
                  - datetime.fromisoformat(world["opened"]))
        return "true" if waited <= timedelta(minutes=world["sla_minutes"]) else "false"
    if world["event"] is None:
        return "insufficient"
    day = _calendar_date(world["date"])
    if day.weekday() in {WEEKDAYS.index(name) for name in world["weekend"]}:
        return "ineligible"
    opening = datetime.combine(day, _clock(world["opens"]))
    closing = datetime.combine(day, _clock(world["closes"]))
    if closing <= opening:
        closing += timedelta(days=1)
    for offset in (timedelta(), timedelta(days=1)):
        start = datetime.combine(day, _clock(world["event"])) + offset
        if start >= opening and start + timedelta(minutes=world["duration_minutes"]) <= closing:
            return "eligible"
    return "ineligible"


def _date_words(value: str, style: str) -> str:
    day = _calendar_date(value)
    if style == "english":
        return f"{ENGLISH_WEEKDAYS[WEEKDAYS[day.weekday()]]}, {digits(str(day.day), style)} " \
               f"{ENGLISH_MONTHS[day.month - 1]} {digits(str(day.year), style)}"
    return (f"{ARABIC_WEEKDAYS[WEEKDAYS[day.weekday()]]} {digits(str(day.day), style)} "
            f"{ARABIC_MONTHS[day.month - 1]} {digits(str(day.year), style)}")


def surface(world: dict) -> dict:
    validate(world)
    style, phrase, noise = variant_of(world, KIND, optional=False)
    words = surface_words(KIND, style, phrase)
    arabic = style != "english"
    join = join_ar if arabic else join_en
    if world["scenario"] == "sla":
        opened = digits(world["opened"].replace("T", " "), style)
        replied = digits(world["replied"].replace("T", " "), style)
        sla = digits(str(world["sla_minutes"]), style)
        separator = "، " if arabic else " "
        state = (f"{words['ticket_label']}: {words['opened_intro']} {opened}{separator}"
                 f"{words['replied_intro']} {replied}. {words['sla_label']}: "
                 f"{words['sla_deadline']} {sla} {words['minutes_word']}.")
        return {"state": noised(state, noise), "question": {
            "type": "noul", "instructions": words["question_sla"],
            "options": noul_options(style)}}
    weekend = join([ARABIC_WEEKDAYS[day] if arabic else ENGLISH_WEEKDAYS[day]
                    for day in world["weekend"]])
    event = (missing_word(style) if world["event"] is None
             else digits(world["event"], style))
    duration = digits(str(world["duration_minutes"]), style)
    date_words = _date_words(world["date"], style)
    window = (f"{words['from_word']} {digits(world['opens'], style)} "
              f"{words['to_word']} {digits(world['closes'], style)}" if arabic
              else f"{digits(world['opens'], style)} {words['to_word']} "
                   f"{digits(world['closes'], style)}")
    state = (f"{words['policy_label']}: {words['closed_days_label']}: {weekend}. "
             f"{words['hours_label']}: {window}. "
             f"{words['appointment_label']}: {event} {words['on_word']} {date_words} "
             f"{words['for_word']} {duration} {words['minutes_word']}.")
    return {"state": noised(state, noise), "question": {
        "type": "choice", "instructions": words["question_business"],
        "options": decision_options(style)}}


def attributes_for(world: dict) -> dict:
    validate(world)
    return attributes(style_of(world, optional=False), TASK)
