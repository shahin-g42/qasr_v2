"""Surface-phrasing vocabulary banks and label-preserving ASR-noise profiles.

A phrasing is a deterministic surface variant of one symbolic world: the same
frozen semantics, the same values, a different wording. Each rule kind defines
named slots (SLOTS[kind]); for every (kind, rendering style) the VOCAB bank
lists alternative wordings per slot. A phrase index is decomposed over the
slots by mixed-radix positional arithmetic, so every index selects exactly one
combination and index 0 selects the first choice of every slot — which is the
byte-exact v1 surface the released pilot rows pin by hash.

Stability contract: choices may be appended ONLY to the final slot of a kind
("question" or the last question slot). Appending to any earlier slot would
remap existing phrase indices onto different combinations. The generator
records phrase indices in symbolic worlds, so this rule keeps frozen releases
readable while the bank grows.

Generated extensions: when the sibling module phrasings_llm.py exists (written
by tools/bank_author.py apply from LLM-drafted, human-approved phrasings), its
QUESTION_EXTENSIONS are merged into the final slots at import time; malformed
or duplicate extensions fail loudly. Seed-drafted banks simply omit the
module and keep the hand-written bank unchanged.

The noise profiles model common transcript artifacts of a voice-agent front
end. They are total, deterministic, character-level functions of the rendered
state; they never touch the world, the solver inputs, or the question, so the
gold decision cannot move. None of them emits format-control characters or
replacement characters, keeping the curate fidelity flags at soft level.
"""

from __future__ import annotations

import importlib.util
import re

NOISE_PROFILES = ("none", "hamza", "ta_marbuta", "western_digits", "punct")

# Slot order per kind; the final slot is append-safe (see module docstring).
SLOTS = {
    "comparison": ("state_label", "condition_label", "connector", "value_missing",
                   "value_prefix", "question"),
    "temporal_window": ("ticket_label", "opened_intro", "replied_intro", "sla_label",
                        "sla_deadline", "policy_label", "closed_days_label", "hours_label",
                        "from_word", "to_word", "appointment_label", "on_word", "for_word",
                        "minutes_word", "question_business", "question_sla"),
    "policy_logic": ("state_label", "head_all", "head_any", "provided_label", "missing_label",
                     "question"),
    "budget_check": ("state_label", "total_label", "spent_label", "requested_label",
                     "price_label", "installment_label", "months_label", "question_single",
                     "question_installments"),
    "intent_routing": ("domain_label", "request_label", "intents_label", "question"),
    "safety_gate": ("request_label", "facts_label", "missing_label", "rules_label",
                    "default_label", "question"),
    "triage_rubric": ("state_label", "items_label", "levels_label", "question"),
    "readback_match": ("expected_label", "readback_label", "missing_readback", "question"),
    "slot_assertion": ("form_label", "claim_label", "claim_intro", "value_intro",
                       "missing_value", "question"),
}

VOCAB: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "comparison": {
        "msa": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "condition_label": ("الشرط", "القاعدة", "الشرط المطلوب"),
            "connector": ("أن تكون القيمة", "يجب أن تكون القيمة", "مطلوب أن تكون القيمة"),
            "value_missing": ("القيمة غير مذكورة", "لم تُذكر القيمة", "القيمة غير محددة"),
            "value_prefix": ("القيمة المذكورة: ", "القيمة المعطاة: ", "القيمة المعلنة: "),
            "question": ("هل يتحقق الشرط وفق المعلومات المذكورة فقط؟",
                         "هل الشرط متحقق بناءً على المعلومات المذكورة فقط؟",
                         "اعتمادًا على المعلومات المذكورة فقط، هل يتحقق الشرط؟"),
        },
        "egyptian": {
            "state_label": ("الحالة", "الوضع", "الموقف"),
            "condition_label": ("الشرط", "القاعدة", "الشرط المطلوب"),
            "connector": ("القيمة لازم تكون", "لازم القيمة تكون", "القيمة لازم تبقى"),
            "value_missing": ("القيمة غير مذكورة", "القيمة مش مذكورة", "القيمة متقالتش"),
            "value_prefix": ("القيمة المذكورة: ", "القيمة المتقالة: ", "القيمة اللي اتقالت: "),
            "question": ("هل الشرط متحقق حسب المعلومات المذكورة بس؟",
                         "الشرط متحقق ولا لأ حسب اللي مذكور بس؟",
                         "بالمعلومات المذكورة بس، هل الشرط متحقق؟"),
        },
        "levantine": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "condition_label": ("الشرط", "القاعدة", "الشرط المطلوب"),
            "connector": ("لازم القيمة تكون", "القيمة لازم تكون", "لازم القيمة تبقى"),
            "value_missing": ("القيمة غير مذكورة", "القيمة ما ذكرت", "ما ذكروا القيمة"),
            "value_prefix": ("القيمة المذكورة: ", "القيمة المعطاة: ", "القيمة اللي انذكرت: "),
            "question": ("هل الشرط متحقق حسب المعلومات المذكورة بس؟",
                         "الشرط متحقق ولا لأ حسب المعلومات المذكورة بس؟",
                         "حسب المعلومات المذكورة بس، هل الشرط متحقق؟"),
        },
        "gulf": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "condition_label": ("الشرط", "القاعدة", "الشرط المطلوب"),
            "connector": ("لازم القيمة تكون", "القيمة لا بد تكون", "لازم إن القيمة تكون"),
            "value_missing": ("القيمة غير مذكورة", "القيمة ما ذُكرت", "ما ذكروا القيمة"),
            "value_prefix": ("القيمة المذكورة: ", "القيمة المعطاة: ", "القيمة اللي ذكروها: "),
            "question": ("هل الشرط متحقق حسب المعلومات المذكورة بس؟",
                         "الشرط متحقق ولا لا حسب اللي مذكور بس؟",
                         "حسب المعلومات المذكورة بس، الشرط متحقق؟"),
        },
        "code_switch": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "condition_label": ("الشرط", "القاعدة", "الشرط المطلوب"),
            "connector": ("أن تكون القيمة", "لازم القيمة تكون", "يجب أن تكون القيمة"),
            "value_missing": ("القيمة غير مذكورة", "القيمة مش مذكورة", "لم تُذكر القيمة"),
            "value_prefix": ("القيمة المذكورة: ", "القيمة المعطاة: ", "القيمة المعلنة: "),
            "question": ("هل يتحقق الشرط وفق المعلومات المذكورة فقط؟",
                         "هل الشرط متحقق حسب المعلومات المذكورة بس؟",
                         "بناءً على المعلومات المذكورة فقط، هل يتحقق الشرط؟"),
        },
        "english": {
            "state_label": ("State", "Situation", "Case"),
            "condition_label": ("Policy", "Rule", "Requirement"),
            "connector": ("the value must be", "the value is required to be",
                          "the value needs to be"),
            "value_missing": ("Value not stated.", "No value was given.",
                              "The value is not stated."),
            "value_prefix": ("Stated value: ", "Given value: ", "Declared value: "),
            "question": ("Does the stated value satisfy the policy on the information given only?",
                         "On the stated information only, does the value satisfy the policy?",
                         "Given only the stated information, is the policy satisfied?"),
        },
    },
    "temporal_window": {
        "msa": {
            "ticket_label": ("طلب الدعم", "طلب المساعدة", "تذكرة الدعم"),
            "opened_intro": ("فُتحت التذكرة في", "فُتح الطلب في", "بدأت التذكرة في"),
            "replied_intro": ("وتم الرد في", "ورد فريق الدعم في", "وكان الرد في"),
            "sla_label": ("اتفاقية مستوى الخدمة", "اتفاقية الخدمة", "المدة المتفق عليها للرد"),
            "sla_deadline": ("يجب الرد خلال", "الرد مطلوب خلال", "مهلة الرد هي"),
            "policy_label": ("السياسة", "قواعد العمل", "سياسة المواعيد"),
            "closed_days_label": ("أيام الإغلاق", "أيام العطلة", "الأيام المغلقة"),
            "hours_label": ("ساعات العمل", "أوقات العمل", "ساعات الدوام"),
            "from_word": ("من",),
            "to_word": ("إلى", "حتى"),
            "appointment_label": ("الموعد المطلوب", "الموعد المطلوب حجزه", "الطلب"),
            "on_word": ("بتاريخ", "في يوم", "يوم"),
            "for_word": ("لمدة", "بمدة", "لمدة قدرها"),
            "minutes_word": ("دقيقة",),
            "question_business": ("هل يقع الموعد داخل أيام وساعات العمل وفق السياسة المذكورة فقط؟",
                                  "هل الموعد ضمن أيام وساعات العمل حسب السياسة المذكورة فقط؟",
                                  "وفق السياسة المذكورة فقط، هل الموعد داخل أيام وساعات العمل؟"),
            "question_sla": ("هل تم الرد خلال المدة المحددة في الاتفاقية؟",
                             "هل جاء الرد ضمن المدة المحددة في الاتفاقية؟",
                             "هل التزم الرد بالمدة المحددة في الاتفاقية؟"),
        },
        "egyptian": {
            "ticket_label": ("طلب الدعم", "طلب المساعدة", "تذكرة الدعم"),
            "opened_intro": ("فُتحت التذكرة في", "التذكرة اتفتحت في", "الطلب بدأ في"),
            "replied_intro": ("وتم الرد في", "ورد فريق الدعم في", "والرد جه في"),
            "sla_label": ("اتفاقية مستوى الخدمة", "اتفاقية الخدمة", "المدة المتفق عليها للرد"),
            "sla_deadline": ("يجب الرد خلال", "لازم الرد يتم خلال", "الرد لازم يكون خلال"),
            "policy_label": ("السياسة", "قواعد الشغل", "سياسة المواعيد"),
            "closed_days_label": ("أيام الإغلاق", "أيام العطلة", "الأيام المقفولة"),
            "hours_label": ("ساعات العمل", "أوقات الشغل", "مواعيد الدوام"),
            "from_word": ("من",),
            "to_word": ("إلى", "لحد"),
            "appointment_label": ("الموعد المطلوب", "الموعد المطلوب حجزه", "الطلب"),
            "on_word": ("بتاريخ", "في يوم", "يوم"),
            "for_word": ("لمدة", "بمدة", "لمدة قد إيه"),
            "minutes_word": ("دقيقة",),
            "question_business": ("هل يقع الموعد داخل أيام وساعات العمل وفق السياسة المذكورة فقط؟",
                                  "هل الموعد جوه أيام وساعات الشغل حسب السياسة المذكورة بس؟",
                                  "الموعد واقع في أيام وساعات الشغل ولا لأ حسب اللي مذكور بس؟"),
            "question_sla": ("هل تم الرد خلال المدة المحددة في الاتفاقية؟",
                             "هل الرد تم في المدة المحددة في الاتفاقية؟",
                             "الرد جه في وقته ولا تأخر عن الاتفاقية؟"),
        },
        "levantine": {
            "ticket_label": ("طلب الدعم", "طلب المساعدة", "تذكرة الدعم"),
            "opened_intro": ("فُتحت التذكرة في", "انفتحت التذكرة في", "بلش الطلب في"),
            "replied_intro": ("وتم الرد في", "ورد فريق الدعم في", "وجا الرد في"),
            "sla_label": ("اتفاقية مستوى الخدمة", "اتفاقية الخدمة", "المدة المتفق عليها للرد"),
            "sla_deadline": ("يجب الرد خلال", "لازم الرد يوصل خلال", "الرد لازم يكون خلال"),
            "policy_label": ("السياسة", "قواعد الشغل", "سياسة المواعيد"),
            "closed_days_label": ("أيام الإغلاق", "أيام العطلة", "الأيام المقفولة"),
            "hours_label": ("ساعات العمل", "أوقات الشغل", "مواعيد الدوام"),
            "from_word": ("من",),
            "to_word": ("إلى", "لحد"),
            "appointment_label": ("الموعد المطلوب", "الموعد المطلوب حجزه", "الطلب"),
            "on_word": ("بتاريخ", "في يوم", "يوم"),
            "for_word": ("لمدة", "بمدة", "لمدة قدرها"),
            "minutes_word": ("دقيقة",),
            "question_business": ("هل يقع الموعد داخل أيام وساعات العمل وفق السياسة المذكورة فقط؟",
                                  "هل الموعد ضمن أيام وساعات العمل حسب السياسة المذكورة بس؟",
                                  "الموعد واقع بأيام وساعات العمل ولا لأ حسب اللي مذكور بس؟"),
            "question_sla": ("هل تم الرد خلال المدة المحددة في الاتفاقية؟",
                             "هل الرد وصل خلال المدة المحددة بالاتفاقية؟",
                             "الرد جا بوقته ولا تأخر عن الاتفاقية؟"),
        },
        "gulf": {
            "ticket_label": ("طلب الدعم", "طلب المساعدة", "تذكرة الدعم"),
            "opened_intro": ("فُتحت التذكرة في", "انفتحت التذكرة في", "بدا الطلب في"),
            "replied_intro": ("وتم الرد في", "ورد فريق الدعم في", "واجا الرد في"),
            "sla_label": ("اتفاقية مستوى الخدمة", "اتفاقية الخدمة", "المدة المتفق عليها للرد"),
            "sla_deadline": ("يجب الرد خلال", "لازم الرد يكون خلال", "الرد لا بد يكون خلال"),
            "policy_label": ("السياسة", "قواعد العمل", "سياسة المواعيد"),
            "closed_days_label": ("أيام الإغلاق", "أيام العطلة", "الأيام المقفلة"),
            "hours_label": ("ساعات العمل", "أوقات العمل", "مواعيد الدوام"),
            "from_word": ("من",),
            "to_word": ("إلى", "لين"),
            "appointment_label": ("الموعد المطلوب", "الموعد المطلوب حجزه", "الطلب"),
            "on_word": ("بتاريخ", "في يوم", "يوم"),
            "for_word": ("لمدة", "بمدة", "لمدة قدرها"),
            "minutes_word": ("دقيقة",),
            "question_business": ("هل يقع الموعد داخل أيام وساعات العمل وفق السياسة المذكورة فقط؟",
                                  "هل الموعد داخل أيام وساعات العمل حسب السياسة المذكورة بس؟",
                                  "الموعد واقع بأيام وساعات العمل ولا لا حسب اللي مذكور بس؟"),
            "question_sla": ("هل تم الرد خلال المدة المحددة في الاتفاقية؟",
                             "هل الرد تم خلال المدة المحددة بالاتفاقية؟",
                             "الرد جا بوقته ولا تأخر عن الاتفاقية؟"),
        },
        "code_switch": {
            "ticket_label": ("طلب الدعم", "طلب المساعدة", "تذكرة الدعم"),
            "opened_intro": ("فُتحت التذكرة في", "التذكرة بدأت في", "فُتح الطلب في"),
            "replied_intro": ("وتم الرد في", "ورد فريق الدعم في", "وكان الرد في"),
            "sla_label": ("اتفاقية مستوى الخدمة", "اتفاقية الخدمة", "المدة المتفق عليها للرد"),
            "sla_deadline": ("يجب الرد خلال", "لازم الرد يتم خلال", "مهلة الرد هي"),
            "policy_label": ("السياسة", "قواعد العمل", "سياسة المواعيد"),
            "closed_days_label": ("أيام الإغلاق", "أيام العطلة", "الأيام المغلقة"),
            "hours_label": ("ساعات العمل", "أوقات العمل", "ساعات الدوام"),
            "from_word": ("من",),
            "to_word": ("إلى", "حتى"),
            "appointment_label": ("الموعد المطلوب", "الموعد المطلوب حجزه", "الطلب"),
            "on_word": ("بتاريخ", "في يوم", "يوم"),
            "for_word": ("لمدة", "بمدة", "لمدة قدرها"),
            "minutes_word": ("دقيقة",),
            "question_business": ("هل يقع الموعد داخل أيام وساعات العمل وفق السياسة المذكورة فقط؟",
                                  "هل الموعد ضمن أيام وساعات العمل حسب السياسة المذكورة بس؟",
                                  "وفق السياسة المذكورة فقط، هل الموعد داخل أيام وساعات العمل؟"),
            "question_sla": ("هل تم الرد خلال المدة المحددة في الاتفاقية؟",
                             "هل جاء الرد ضمن المدة المحددة في الاتفاقية؟",
                             "هل التزم الرد بالمدة المحددة في الاتفاقية؟"),
        },
        "english": {
            "ticket_label": ("Support ticket", "Support request", "Helpdesk ticket"),
            "opened_intro": ("opened at", "was opened at", "started at"),
            "replied_intro": ("and answered at", "with the reply at", "and the reply came at"),
            "sla_label": ("Service level agreement", "The SLA", "The agreed service level"),
            "sla_deadline": ("the reply must arrive within", "the reply is due within",
                             "replies are required within"),
            "policy_label": ("Policy", "Rules", "Scheduling policy"),
            "closed_days_label": ("closed days", "closure days", "days off"),
            "hours_label": ("Working hours", "opening hours", "business hours"),
            "from_word": ("from",),
            "to_word": ("to", "until"),
            "appointment_label": ("Requested appointment", "The requested appointment",
                                  "Appointment"),
            "on_word": ("on", "dated", "falling on"),
            "for_word": ("for", "lasting", "for a duration of"),
            "minutes_word": ("minutes",),
            "question_business": ("Does the appointment fall inside working days and hours on the "
                                  "stated policy only?",
                                  "Is the appointment within working days and hours under the "
                                  "stated policy only?",
                                  "On the stated policy only, does the appointment fall inside "
                                  "working days and hours?"),
            "question_sla": ("Was the reply sent within the agreed response time?",
                             "Did the reply arrive inside the agreed response window?",
                             "Was the agreed response time met?"),
        },
    },
    "policy_logic": {
        "msa": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "head_all": ("يشترط تحقق جميع الشروط: ", "يجب أن تتحقق جميع الشروط: ",
                         "المطلوب تحقق كل الشروط: "),
            "head_any": ("يكفي تحقق أحد الشروط: ", "يُشترط تحقق شرط واحد على الأقل: ",
                         "يكفي أن يتحقق أي شرط: "),
            "provided_label": ("القيم المذكورة", "المعطيات المذكورة", "القيم المتوفرة"),
            "missing_label": ("قيم غير مذكورة", "معطيات غير مذكورة", "قيم غير متوفرة"),
            "question": ("هل يستوفي الطلب الشروط المذكورة وفق المعلومات المتوفرة فقط؟",
                         "هل الطلب مستوفٍ للشروط بناءً على المعلومات المتوفرة فقط؟",
                         "وفق المعلومات المتوفرة فقط، هل يستوفي الطلب الشروط المذكورة؟"),
        },
        "egyptian": {
            "state_label": ("الحالة", "الوضع", "الموقف"),
            "head_all": ("يشترط تحقق جميع الشروط: ", "لازم كل الشروط تتحقق: ",
                         "مطلوب إن كل الشروط تتحقق: "),
            "head_any": ("يكفي تحقق أحد الشروط: ", "يكفي شرط واحد يتحقق: ",
                         "لازم شرط واحد على الأقل يتحقق: "),
            "provided_label": ("القيم المذكورة", "اللي اتقال", "القيم المتاحة"),
            "missing_label": ("قيم غير مذكورة", "قيم مش مذكورة", "معطيات مش متقالة"),
            "question": ("هل الطلب يستوفي الشروط حسب المعلومات المتاحة بس؟",
                         "الطلب مستوفي الشروط ولا لأ حسب اللي متاح بس؟",
                         "حسب المعلومات المتاحة بس، هل الطلب يستوفي الشروط؟"),
        },
        "levantine": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "head_all": ("يشترط تحقق جميع الشروط: ", "لازم كل الشروط تتحقق: ",
                         "مطلوب إنه كل الشروط تتحقق: "),
            "head_any": ("يكفي تحقق أحد الشروط: ", "يكفي شرط واحد يتحقق: ",
                         "لازم شرط واحد على الأقل يتحقق: "),
            "provided_label": ("القيم المذكورة", "اللي انذكر", "القيم المتاحة"),
            "missing_label": ("قيم غير مذكورة", "قيم ما ذكرت", "معطيات ما انذكرت"),
            "question": ("هل الطلب مستوفي الشروط حسب المعلومات المتاحة بس؟",
                         "الطلب مستوفي الشروط ولا لأ حسب اللي متاح بس؟",
                         "حسب المعلومات المتاحة بس، هل الطلب مستوفي الشروط؟"),
        },
        "gulf": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "head_all": ("يشترط تحقق جميع الشروط: ", "لازم كل الشروط تتحقق: ",
                         "مطلوب إن كل الشروط تتحقق: "),
            "head_any": ("يكفي تحقق أحد الشروط: ", "يكفي شرط واحد يتحقق: ",
                         "لازم شرط واحد على الأقل يتحقق: "),
            "provided_label": ("القيم المذكورة", "اللي ذكروه", "القيم المتاحة"),
            "missing_label": ("قيم غير مذكورة", "قيم ما ذُكرت", "معطيات ما ذكروها"),
            "question": ("هل يفي الطلب بالشروط حسب المعلومات المتاحة بس؟",
                         "الطلب فياي بالشروط ولا لا حسب اللي متاح بس؟",
                         "حسب المعلومات المتاحة بس، هل يفي الطلب بالشروط؟"),
        },
        "code_switch": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "head_all": ("يشترط تحقق جميع الشروط: ", "لازم كل الشروط تتحقق: ",
                         "المطلوب تحقق كل الشروط: "),
            "head_any": ("يكفي تحقق أحد الشروط: ", "يكفي شرط واحد يتحقق: ",
                         "يكفي أن يتحقق أي شرط: "),
            "provided_label": ("القيم المذكورة", "اللي اتقال", "القيم المتاحة"),
            "missing_label": ("قيم غير مذكورة", "قيم مش مذكورة", "قيم ناقصة"),
            "question": ("هل يستوفي الطلب الشروط المذكورة وفق المعلومات المتوفرة فقط؟",
                         "هل الطلب يستوفي الشروط حسب المعلومات المتاحة بس؟",
                         "وفق المعلومات المتوفرة فقط، هل يستوفي الطلب الشروط المذكورة؟"),
        },
        "english": {
            "state_label": ("Case", "Situation", "Request"),
            "head_all": ("All conditions must hold: ", "Every condition must hold: ",
                         "All of the conditions must hold: "),
            "head_any": ("At least one condition must hold: ", "Any one condition must hold: ",
                         "One or more conditions must hold: "),
            "provided_label": ("Stated values", "Given values", "Provided values"),
            "missing_label": ("Values not stated", "Missing values", "Unstated values"),
            "question": ("Does the request satisfy the stated conditions on the available "
                         "information only?",
                         "On the available information only, does the request satisfy the stated "
                         "conditions?",
                         "Given only the available information, are the stated conditions "
                         "satisfied?"),
        },
    },
    "budget_check": {
        "msa": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "total_label": ("الميزانية الإجمالية", "الميزانية الكلية", "إجمالي الميزانية"),
            "spent_label": ("المبلغ المستهلك حتى الآن", "المبلغ المصروف حتى الآن",
                            "ما استُهلك حتى الآن"),
            "requested_label": ("المبلغ المطلوب", "المبلغ المطلوب صرفه", "قيمة الطلب"),
            "price_label": ("سعر الطلب", "ثمن الطلب", "قيمة الطلب"),
            "installment_label": ("القسط الشهري", "القسط المحدد شهريًا", "قيمة القسط"),
            "months_label": ("عدد الأشهر", "عدد الأشهر المطلوبة", "عدد الأقساط"),
            "question_single": ("هل تسمح الميزانية المتبقية بصرف المبلغ المطلوب وفق المعلومات "
                                "المذكورة فقط؟",
                                "هل الميزانية المتبقية تكفي للمبلغ المطلوب حسب المعلومات "
                                "المذكورة فقط؟",
                                "وفق المعلومات المذكورة فقط، هل يمكن صرف المبلغ المطلوب من "
                                "الميزانية المتبقية؟"),
            "question_installments": ("هل يغطي مجموع الأقساط كامل السعر وفق المعلومات المذكورة "
                                      "فقط؟",
                                      "هل الأقساط مجتمعة تغطي السعر كاملًا حسب المعلومات "
                                      "المذكورة فقط؟",
                                      "وفق المعلومات المذكورة فقط، هل يغطي مجموع الأقساط "
                                      "السعر بالكامل؟"),
        },
        "egyptian": {
            "state_label": ("الحالة", "الوضع", "الموقف"),
            "total_label": ("الميزانية الإجمالية", "الميزانية كلها", "إجمالي الميزانية"),
            "spent_label": ("المبلغ المستهلك حتى الآن", "المبلغ المستهلك لحد دلوقتي",
                            "اللي اتصرف لحد دلوقتي"),
            "requested_label": ("المبلغ المطلوب", "المبلغ المطلوب صرفه", "قيمة الطلب"),
            "price_label": ("سعر الطلب", "ثمن الطلب", "قيمة الطلب"),
            "installment_label": ("القسط الشهري", "القسط في الشهر", "قيمة القسط"),
            "months_label": ("عدد الأشهر", "عدد الشهور", "عدد الأقساط"),
            "question_single": ("هل الميزانية الباقية تسمح بصرف المبلغ المطلوب حسب المعلومات "
                                "المذكورة بس؟",
                                "الميزانية الباقية تكفي للمبلغ المطلوب ولا لأ حسب اللي مذكور "
                                "بس؟",
                                "حسب المعلومات المذكورة بس، هل ينفع نصرف المبلغ المطلوب؟"),
            "question_installments": ("هل مجموع الأقساط يغطي السعر كامل حسب المعلومات المذكورة "
                                      "بس؟",
                                      "الأقساط كلها مع بعض تغطي السعر ولا لأ حسب اللي مذكور "
                                      "بس؟",
                                      "حسب المعلومات المذكورة بس، هل الأقساط تغطي السعر كامل؟"),
        },
        "levantine": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "total_label": ("الميزانية الإجمالية", "الميزانية الكاملة", "إجمالي الميزانية"),
            "spent_label": ("المبلغ المستهلك حتى الآن", "المبلغ المستهلك هلأ",
                            "اللي انصرف هلأ"),
            "requested_label": ("المبلغ المطلوب", "المبلغ المطلوب صرفه", "قيمة الطلب"),
            "price_label": ("سعر الطلب", "ثمن الطلب", "قيمة الطلب"),
            "installment_label": ("القسط الشهري", "القسط بالشهر", "قيمة القسط"),
            "months_label": ("عدد الأشهر", "عدد الأشهار", "عدد الأقساط"),
            "question_single": ("هل الميزانية الباقية تسمح بصرف المبلغ المطلوب حسب المعلومات "
                                "المذكورة بس؟",
                                "الميزانية الباقية تكفي للمبلغ المطلوب ولا لأ حسب اللي مذكور "
                                "بس؟",
                                "حسب المعلومات المذكورة بس، هل فينا نصرف المبلغ المطلوب؟"),
            "question_installments": ("هل مجموع الأقساط يغطي السعر كامل حسب المعلومات المذكورة "
                                      "بس؟",
                                      "الأقساط كلها مع بعض تغطي السعر ولا لأ حسب اللي مذكور "
                                      "بس؟",
                                      "حسب المعلومات المذكورة بس، هل الأقساط تغطي السعر كامل؟"),
        },
        "gulf": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "total_label": ("الميزانية الإجمالية", "الميزانية كاملها", "إجمالي الميزانية"),
            "spent_label": ("المبلغ المستهلك حتى الآن", "المبلغ المستهلك للحين",
                            "اللي انصرف للحين"),
            "requested_label": ("المبلغ المطلوب", "المبلغ المطلوب صرفه", "قيمة الطلب"),
            "price_label": ("سعر الطلب", "ثمن الطلب", "قيمة الطلب"),
            "installment_label": ("القسط الشهري", "القسط بالشهر", "قيمة القسط"),
            "months_label": ("عدد الأشهر", "عدد الأشهار", "عدد الأقساط"),
            "question_single": ("هل الميزانية المتبقية تسمح بصرف المبلغ المطلوب حسب المعلومات "
                                "المذكورة بس؟",
                                "الميزانية الباقية تكفي للمبلغ المطلوب ولا لا حسب اللي مذكور "
                                "بس؟",
                                "حسب المعلومات المذكورة بس، هل نقدر نصرف المبلغ المطلوب؟"),
            "question_installments": ("هل مجموع الأقساط يغطي السعر كامل حسب المعلومات المذكورة "
                                      "بس؟",
                                      "الأقساط كلها مع بعض تغطي السعر ولا لا حسب اللي مذكور "
                                      "بس؟",
                                      "حسب المعلومات المذكورة بس، هل الأقساط تغطي السعر كامل؟"),
        },
        "code_switch": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "total_label": ("الميزانية الإجمالية", "الميزانية الكلية", "إجمالي الميزانية"),
            "spent_label": ("المبلغ المستهلك حتى الآن", "اللي اتصرف لحد الآن",
                            "المبلغ المصروف حتى الآن"),
            "requested_label": ("المبلغ المطلوب", "المبلغ المطلوب صرفه", "قيمة الطلب"),
            "price_label": ("سعر الطلب", "ثمن الطلب", "قيمة الطلب"),
            "installment_label": ("القسط الشهري", "القسط بالشهر", "قيمة القسط"),
            "months_label": ("عدد الأشهر", "عدد الشهور", "عدد الأقساط"),
            "question_single": ("هل تسمح الميزانية المتبقية بصرف المبلغ المطلوب وفق المعلومات "
                                "المذكورة فقط؟",
                                "هل الميزانية الباقية تكفي للمبلغ المطلوب حسب المعلومات "
                                "المذكورة بس؟",
                                "وفق المعلومات المذكورة فقط، هل يمكن صرف المبلغ المطلوب؟"),
            "question_installments": ("هل يغطي مجموع الأقساط كامل السعر وفق المعلومات المذكورة "
                                      "فقط؟",
                                      "هل الأقساط مجتمعة تغطي السعر كاملًا حسب المعلومات "
                                      "المذكورة بس؟",
                                      "وفق المعلومات المذكورة فقط، هل مجموع الأقساط يغطي "
                                      "السعر بالكامل؟"),
        },
        "english": {
            "state_label": ("Case", "Situation", "Request"),
            "total_label": ("Total budget", "Overall budget", "The total budget"),
            "spent_label": ("Amount spent so far", "Spent so far", "Amount used so far"),
            "requested_label": ("Requested amount", "The requested amount", "Amount requested"),
            "price_label": ("Order price", "Purchase price", "The price"),
            "installment_label": ("Monthly installment", "The monthly installment",
                                  "Installment amount"),
            "months_label": ("Number of months", "Months", "Month count"),
            "question_single": ("Does the remaining budget allow the requested amount on the "
                                "stated information only?",
                                "On the stated information only, does the remaining budget cover "
                                "the requested amount?",
                                "Given only the stated information, can the requested amount be "
                                "spent from the remaining budget?"),
            "question_installments": ("Do the installments together cover the full price on the "
                                      "stated information only?",
                                      "On the stated information only, do the installments add "
                                      "up to the full price?",
                                      "Given only the stated information, do the installments "
                                      "cover the whole price?"),
        },
    },
    "intent_routing": {
        "msa": {
            "domain_label": ("المجال", "نطاق الخدمة", "مجال الطلب"),
            "request_label": ("طلب العميل", "كلام العميل", "رسالة العميل"),
            "intents_label": ("النوايا المتاحة بالترتيب", "قائمة النوايا بالترتيب",
                              "النوايا المحددة بالترتيب"),
            "question": ("ما النية الأقرب لطلب العميل وفق القائمة والقاعدة المذكورة؟",
                         "أي نية من القائمة تناسب طلب العميل أكثر وفق القاعدة المذكورة؟",
                         "وفق القائمة والقاعدة المذكورة، ما النية الأقرب للطلب؟"),
        },
        "egyptian": {
            "domain_label": ("المجال", "نطاق الخدمة", "مجال الطلب"),
            "request_label": ("طلب العميل", "كلام العميل", "اللي العميل قاله"),
            "intents_label": ("النوايا المتاحة بالترتيب", "قائمة النوايا بالترتيب",
                              "النوايا الموجودة بالترتيب"),
            "question": ("إيه أقرب نية للطلب حسب القائمة والقاعدة؟",
                         "أي نية من القائمة أقرب لكلام العميل حسب القاعدة؟",
                         "حسب القائمة والقاعدة، إيه النية الأقرب للطلب؟"),
        },
        "levantine": {
            "domain_label": ("المجال", "نطاق الخدمة", "مجال الطلب"),
            "request_label": ("طلب العميل", "كلام العميل", "اللي قاله العميل"),
            "intents_label": ("النوايا المتاحة بالترتيب", "قائمة النوايا بالترتيب",
                              "النوايا الموجودة بالترتيب"),
            "question": ("شو أقرب نية للطلب حسب القائمة والقاعدة؟",
                         "أي نية من القائمة أقربلكلام العميل حسب القاعدة؟",
                         "حسب القائمة والقاعدة، شو النية الأقرب للطلب؟"),
        },
        "gulf": {
            "domain_label": ("المجال", "نطاق الخدمة", "مجال الطلب"),
            "request_label": ("طلب العميل", "كلام العميل", "اللي قاله العميل"),
            "intents_label": ("النوايا المتاحة بالترتيب", "قائمة النوايا بالترتيب",
                              "النوايا الموجودة بالترتيب"),
            "question": ("وش أقرب نية للطلب حسب القائمة والقاعدة؟",
                         "أي نية من القائمة أقرب لكلام العميل حسب القاعدة؟",
                         "حسب القائمة والقاعدة، وش النية الأقرب للطلب؟"),
        },
        "code_switch": {
            "domain_label": ("المجال", "نطاق الخدمة", "مجال الطلب"),
            "request_label": ("طلب العميل", "كلام العميل", "رسالة العميل"),
            "intents_label": ("النوايا المتاحة بالترتيب", "قائمة النوايا بالترتيب",
                              "النوايا المحددة بالترتيب"),
            "question": ("ما النية الأقرب لطلب العميل وفق القائمة والقاعدة المذكورة؟",
                         "أي نية من القائمة تناسب الطلب أكثر حسب القاعدة؟",
                         "وفق القائمة والقاعدة، ما النية الأقرب للطلب؟"),
        },
        "english": {
            "domain_label": ("Domain", "Service domain", "Area"),
            "request_label": ("Customer request", "The customer request", "Customer message"),
            "intents_label": ("Intents in order", "Available intents in order",
                              "The intent list in order"),
            "question": ("Which intent best matches the customer request under the stated list "
                         "and rule?",
                         "Under the stated list and rule, which intent fits the customer request "
                         "best?",
                         "Which of the listed intents is closest to the request under the stated "
                         "rule?"),
        },
    },
    "safety_gate": {
        "msa": {
            "request_label": ("الطلب", "طلب المستخدم", "الطلب الوارد"),
            "facts_label": ("معلومات الطلب", "بيانات الطلب", "المعلومات المتوفرة عن الطلب"),
            "missing_label": ("معلومات غير مذكورة", "بيانات غير متوفرة", "معلومات ناقصة"),
            "rules_label": ("القواعد بالترتيب", "القواعد مرتبةً", "قائمة القواعد بالترتيب"),
            "default_label": ("إذا لم تنطبق أي قاعدة", "في حال عدم انطباق أي قاعدة",
                              "إن لم تنطبق أي من القواعد"),
            "question": ("ما القرار الصحيح وفق القواعد المذكورة بالترتيب؟",
                         "أي قرار هو الصحيح حسب القواعد المذكورة بالترتيب؟",
                         "وفق القواعد المذكورة بالترتيب، ما القرار الصحيح؟"),
        },
        "egyptian": {
            "request_label": ("الطلب", "طلب المستخدم", "الطلب الجاي"),
            "facts_label": ("معلومات الطلب", "بيانات الطلب", "اللي متاح عن الطلب"),
            "missing_label": ("معلومات غير مذكورة", "معلومات مش مذكورة", "بيانات مش متوفرة"),
            "rules_label": ("القواعد بالترتيب", "القواعد مرتبةً", "قائمة القواعد بالترتيب"),
            "default_label": ("إذا لم تنطبق أي قاعدة", "لو مفيش قاعدة اتنطبقت",
                              "في حال عدم انطباق أي قاعدة"),
            "question": ("إيه القرار الصحيح حسب القواعد بالترتيب؟",
                         "أي قرار هو الصح حسب القواعد المذكورة بالترتيب؟",
                         "حسب القواعد بالترتيب، إيه القرار الصحيح؟"),
        },
        "levantine": {
            "request_label": ("الطلب", "طلب المستخدم", "الطلب الوارد"),
            "facts_label": ("معلومات الطلب", "بيانات الطلب", "اللي متاح عن الطلب"),
            "missing_label": ("معلومات غير مذكورة", "معلومات ما ذكرت", "بيانات ما توفّرت"),
            "rules_label": ("القواعد بالترتيب", "القواعد مرتبةً", "قائمة القواعد بالترتيب"),
            "default_label": ("إذا لم تنطبق أي قاعدة", "إذا ما انطبقت ولا قاعدة",
                              "بحال عدم انطباق أي قاعدة"),
            "question": ("شو القرار الصحيح حسب القواعد بالترتيب؟",
                         "أي قرار هو الصح حسب القواعد المذكورة بالترتيب؟",
                         "حسب القواعد بالترتيب، شو القرار الصحيح؟"),
        },
        "gulf": {
            "request_label": ("الطلب", "طلب المستخدم", "الطلب الوارد"),
            "facts_label": ("معلومات الطلب", "بيانات الطلب", "اللي متاح عن الطلب"),
            "missing_label": ("معلومات غير مذكورة", "معلومات ما ذُكرت", "بيانات ما توفّرت"),
            "rules_label": ("القواعد بالترتيب", "القواعد مرتبةً", "قائمة القواعد بالترتيب"),
            "default_label": ("إذا لم تنطبق أي قاعدة", "إذا ما انطبقت أي قاعدة",
                              "بحال عدم انطباق أي قاعدة"),
            "question": ("وش القرار الصحيح حسب القواعد بالترتيب؟",
                         "أي قرار هو الصح حسب القواعد المذكورة بالترتيب؟",
                         "حسب القواعد بالترتيب، وش القرار الصحيح؟"),
        },
        "code_switch": {
            "request_label": ("الطلب", "طلب المستخدم", "الطلب الوارد"),
            "facts_label": ("معلومات الطلب", "بيانات الطلب", "المعلومات المتوفرة عن الطلب"),
            "missing_label": ("معلومات غير مذكورة", "معلومات مش مذكورة", "معلومات ناقصة"),
            "rules_label": ("القواعد بالترتيب", "القواعد مرتبةً", "قائمة القواعد بالترتيب"),
            "default_label": ("إذا لم تنطبق أي قاعدة", "لو ما انطبقت أي قاعدة",
                              "إن لم تنطبق أي من القواعد"),
            "question": ("ما القرار الصحيح وفق القواعد المذكورة بالترتيب؟",
                         "أي قرار هو الصحيح حسب القواعد المذكورة بالترتيب؟",
                         "وفق القواعد المذكورة بالترتيب، ما القرار الصحيح؟"),
        },
        "english": {
            "request_label": ("Request", "The request", "Incoming request"),
            "facts_label": ("Request facts", "Request data", "Known facts"),
            "missing_label": ("Facts not stated", "Missing facts", "Unstated facts"),
            "rules_label": ("Rules in order", "The rules in order", "Ordered rules"),
            "default_label": ("If no rule applies", "When no rule applies",
                              "If none of the rules apply"),
            "question": ("What is the correct decision under the stated rules in order?",
                         "Under the stated rules in order, which decision is correct?",
                         "Which decision follows from the stated rules, evaluated in order?"),
        },
    },
    "triage_rubric": {
        "msa": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "items_label": ("بنود التقييم", "بنود التصنيف", "معايير التقييم"),
            "levels_label": ("المستويات", "مستويات التصنيف", "الدرجات"),
            "question": ("ما المستوى الصحيح وفق بنود التقييم وعتبات المستويات المذكورة؟",
                         "أي مستوى هو الصحيح حسب البنود والعتبات المذكورة؟",
                         "وفق البنود والعتبات المذكورة، ما المستوى الصحيح؟"),
        },
        "egyptian": {
            "state_label": ("الحالة", "الوضع", "الموقف"),
            "items_label": ("بنود التقييم", "بنود التصنيف", "معايير التقييم"),
            "levels_label": ("المستويات", "مستويات التصنيف", "الدرجات"),
            "question": ("إيه المستوى الصحيح حسب البنود والعتبات؟",
                         "أي مستوى هو الصح حسب البنود والعتبات المذكورة؟",
                         "حسب البنود والعتبات، إيه المستوى الصحيح؟"),
        },
        "levantine": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "items_label": ("بنود التقييم", "بنود التصنيف", "معايير التقييم"),
            "levels_label": ("المستويات", "مستويات التصنيف", "الدرجات"),
            "question": ("شو المستوى الصحيح حسب البنود والعتبات؟",
                         "أي مستوى هو الصح حسب البنود والعتبات المذكورة؟",
                         "حسب البنود والعتبات، شو المستوى الصحيح؟"),
        },
        "gulf": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "items_label": ("بنود التقييم", "بنود التصنيف", "معايير التقييم"),
            "levels_label": ("المستويات", "مستويات التصنيف", "الدرجات"),
            "question": ("وش المستوى الصحيح حسب البنود والعتبات؟",
                         "أي مستوى هو الصح حسب البنود والعتبات المذكورة؟",
                         "حسب البنود والعتبات، وش المستوى الصحيح؟"),
        },
        "code_switch": {
            "state_label": ("الحالة", "الوضع", "المعطيات"),
            "items_label": ("بنود التقييم", "بنود التصنيف", "معايير التقييم"),
            "levels_label": ("المستويات", "مستويات التصنيف", "الدرجات"),
            "question": ("ما المستوى الصحيح وفق بنود التقييم وعتبات المستويات المذكورة؟",
                         "أي مستوى هو الصحيح حسب البنود والعتبات المذكورة؟",
                         "وفق البنود والعتبات، ما المستوى الصحيح؟"),
        },
        "english": {
            "state_label": ("Case", "Situation", "Request"),
            "items_label": ("Rubric items", "Scoring items", "Rubric criteria"),
            "levels_label": ("Levels", "Tiers", "Severity levels"),
            "question": ("Which level is correct under the stated rubric items and thresholds?",
                         "Under the stated items and thresholds, which level is the right one?",
                         "Given the rubric items and thresholds, what is the correct level?"),
        },
    },
    "readback_match": {
        "msa": {
            "expected_label": ("القيمة المتوقعة", "القيمة المفترضة", "القيمة المطلوبة"),
            "readback_label": ("القراءة المسجلة", "القراءة المُعادة", "ما أُعيدت قراءته"),
            "missing_readback": ("لم تُسجَّل أي قراءة", "لا توجد قراءة مسجلة",
                                 "لم تُقرأ أي قيمة"),
            "question": ("هل تطابق القراءة القيمة المتوقعة بعد توحيد الوحدات؟",
                         "بعد توحيد الوحدات، هل القراءة مطابقة للقيمة المتوقعة؟",
                         "هل القراءة المسجلة تعطي القيمة المتوقعة نفسها بعد توحيد الوحدات؟"),
        },
        "egyptian": {
            "expected_label": ("القيمة المتوقعة", "القيمة المفروضة", "القيمة المطلوبة"),
            "readback_label": ("القراءة المسجلة", "اللي اتصاغ تاني", "القراءة المُعادة"),
            "missing_readback": ("لم تُسجَّل أي قراءة", "مفيش قراءة مسجلة",
                                 "ما اتسجلش أي قراءة"),
            "question": ("هل القراءة تطابق القيمة المتوقعة بعد توحيد الوحدات؟",
                         "بعد توحيد الوحدات، القراءة زي القيمة المتوقعة ولا لأ؟",
                         "هل اللي اتصاغ تاني يطابق القيمة المتوقعة بعد توحيد الوحدات؟"),
        },
        "levantine": {
            "expected_label": ("القيمة المتوقعة", "القيمة المفروضة", "القيمة المطلوبة"),
            "readback_label": ("القراءة المسجلة", "اللي انعاد", "القراءة المُعادة"),
            "missing_readback": ("لم تُسجَّل أي قراءة", "ما في قراءة مسجلة",
                                 "ما انسجّلت أي قراءة"),
            "question": ("هل القراءة تطابق القيمة المتوقعة بعد توحيد الوحدات؟",
                         "بعد توحيد الوحدات، القراءة متطابقة مع القيمة المتوقعة ولا لأ؟",
                         "هل اللي انعاد يطابق القيمة المتوقعة بعد توحيد الوحدات؟"),
        },
        "gulf": {
            "expected_label": ("القيمة المتوقعة", "القيمة المفترضة", "القيمة المطلوبة"),
            "readback_label": ("القراءة المسجلة", "اللي انعاد", "القراءة المُعادة"),
            "missing_readback": ("لم تُسجَّل أي قراءة", "ما فيه قراءة مسجلة",
                                 "ما انسجّلت أي قراءة"),
            "question": ("هل القراءة تطابق القيمة المتوقعة بعد توحيد الوحدات؟",
                         "بعد توحيد الوحدات، القراءة مطابقة للقيمة المتوقعة ولا لا؟",
                         "هل اللي انعاد يطابق القيمة المتوقعة بعد توحيد الوحدات؟"),
        },
        "code_switch": {
            "expected_label": ("القيمة المتوقعة", "القيمة المفترضة", "القيمة المطلوبة"),
            "readback_label": ("القراءة المسجلة", "القراءة المُعادة", "ما أُعيدت قراءته"),
            "missing_readback": ("لم تُسجَّل أي قراءة", "لا توجد قراءة مسجلة",
                                 "لم تُقرأ أي قيمة"),
            "question": ("هل تطابق القراءة القيمة المتوقعة بعد توحيد الوحدات؟",
                         "بعد توحيد الوحدات، هل القراءة مطابقة للقيمة المتوقعة؟",
                         "هل القراءة المسجلة تعطي القيمة المتوقعة نفسها بعد توحيد الوحدات؟"),
        },
        "english": {
            "expected_label": ("Expected value", "The expected value", "Target value"),
            "readback_label": ("Recorded readback", "The readback", "Read-back value"),
            "missing_readback": ("no readback was recorded", "no readback exists",
                                 "nothing was read back"),
            "question": ("Does the readback match the expected value after unit normalization?",
                         "After normalizing units, does the readback equal the expected value?",
                         "Is the readback, once units are normalized, the same as the expected "
                         "value?"),
        },
    },
    "slot_assertion": {
        "msa": {
            "form_label": ("النموذج", "استمارة البيانات", "بيانات النموذج"),
            "claim_label": ("الادعاء", "القول", "المعلومة المدّعاة"),
            "claim_intro": ("أن الحقل", "بأن الحقل", "بأن الخانة"),
            "value_intro": ("قيمته", "قيمته هي", "وقيمته"),
            "missing_value": ("غير مذكور في النموذج", "غير موجود في النموذج",
                              "ليس في النموذج"),
            "question": ("هل الادعاء مطابق لمحتوى النموذج؟",
                         "هل ما يُدَّعى يطابق محتوى النموذج؟",
                         "هل يتطابق الادعاء مع ما هو مدوَّن في النموذج؟"),
        },
        "egyptian": {
            "form_label": ("النموذج", "استمارة البيانات", "بيانات النموذج"),
            "claim_label": ("الادعاء", "الكلام", "اللي بيتقال"),
            "claim_intro": ("أن الحقل", "إن الحقل", "بإن الحقل"),
            "value_intro": ("قيمته", "قيمته هي", "وقيمته"),
            "missing_value": ("غير مذكور في النموذج", "مش مذكور في النموذج",
                              "مش موجود في النموذج"),
            "question": ("هل الادعاء مطابق لمحتوى النموذج؟",
                         "الكلام ده مطابق للنموذج ولا لأ؟",
                         "هل اللي بيتقال يطابق اللي في النموذج؟"),
        },
        "levantine": {
            "form_label": ("النموذج", "استمارة البيانات", "بيانات النموذج"),
            "claim_label": ("الادعاء", "الكلام", "اللي بيتقال"),
            "claim_intro": ("أن الحقل", "إن الحقل", "بإن الحقل"),
            "value_intro": ("قيمته", "قيمته هي", "وقيمته"),
            "missing_value": ("غير مذكور في النموذج", "ما ذكر بالنموذج", "ما موجود بالنموذج"),
            "question": ("هل الادعاء مطابق لمحتوى النموذج؟",
                         "الكلام مطابق للنموذج ولا لأ؟",
                         "هل اللي بيتقال يطابق اللي بالنموذج؟"),
        },
        "gulf": {
            "form_label": ("النموذج", "استمارة البيانات", "بيانات النموذج"),
            "claim_label": ("الادعاء", "الكلام", "اللي بيتقال"),
            "claim_intro": ("أن الحقل", "إن الحقل", "بإن الحقل"),
            "value_intro": ("قيمته", "قيمته هي", "وقيمته"),
            "missing_value": ("غير مذكور في النموذج", "ما ذُكر بالنموذج", "ما موجود بالنموذج"),
            "question": ("هل الادعاء مطابق لمحتوى النموذج؟",
                         "الكلام مطابق للنموذج ولا لا؟",
                         "هل اللي بيتقال يطابق اللي بالنموذج؟"),
        },
        "code_switch": {
            "form_label": ("النموذج", "استمارة البيانات", "بيانات النموذج"),
            "claim_label": ("الادعاء", "القول", "المعلومة المدّعاة"),
            "claim_intro": ("أن الحقل", "إن الحقل", "بأن الخانة"),
            "value_intro": ("قيمته", "قيمته هي", "وقيمته"),
            "missing_value": ("غير مذكور في النموذج", "مش مذكور في النموذج",
                              "ليس في النموذج"),
            "question": ("هل الادعاء مطابق لمحتوى النموذج؟",
                         "هل ما يُدَّعى يطابق محتوى النموذج؟",
                         "هل يتطابق الادعاء مع ما هو مدوَّن في النموذج؟"),
        },
        "english": {
            "form_label": ("Form", "The form", "Form data"),
            "claim_label": ("Claim", "The claim", "Assertion"),
            "claim_intro": ("the field", "the form field", "the entry"),
            "value_intro": ("is", "has value", "has the value"),
            "missing_value": ("not stated in the form", "missing from the form",
                              "absent from the form"),
            "question": ("Does the claim match the form content?",
                         "Is the claim consistent with what the form contains?",
                         "Does the claim agree with the form's contents?"),
        },
    },
}

_HAMZA = str.maketrans("أإآؤئ", "اااوي")
_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩٫", "0123456789.")
_FINAL_TA = re.compile(r"ة(?=[\s.,;:]|$)")
_FINAL_ALIF = re.compile(r"ى(?=[\s.,;:]|$)")


def apply_noise(state: str, noise: int) -> str:
    """Deterministically apply one transcript-noise profile to a rendered state."""
    if not 0 <= noise < len(NOISE_PROFILES):
        raise ValueError("unknown noise profile")
    profile = NOISE_PROFILES[noise]
    if profile == "none":
        return state
    if profile == "hamza":
        return state.translate(_HAMZA)
    if profile == "ta_marbuta":
        return _FINAL_ALIF.sub("ي", _FINAL_TA.sub("ه", state))
    if profile == "western_digits":
        return state.translate(_WESTERN)
    # "punct": transcript pipelines routinely drop list commas.
    return state.replace("، ", " ").replace(", ", " ")


def phrase_count(kind: str, style: str) -> int:
    """Number of distinct phrasings available for one (kind, style)."""
    vocab = VOCAB[kind][style]
    count = 1
    for slot in SLOTS[kind]:
        count *= len(vocab[slot])
    return count


def phrase_slots(kind: str, style: str, index: int) -> dict[str, str]:
    """Mixed-radix decomposition of a phrase index over the kind's slots."""
    if not 0 <= index < phrase_count(kind, style):
        raise ValueError("phrase index outside the bank")
    vocab = VOCAB[kind][style]
    chosen: dict[str, str] = {}
    for slot in SLOTS[kind]:
        choices = vocab[slot]
        chosen[slot] = choices[index % len(choices)]
        index //= len(choices)
    return chosen


def _merge_extensions(extensions) -> None:
    """Append generated question extensions to each kind's final slot.

    The extensions mapping has the shape {kind: {style: (choice, ...)}} and
    may only add choices to the final slot of a kind - the one append-safe
    position (see module docstring) - so every recorded phrase index keeps
    its exact meaning. Malformed or duplicate extensions fail loudly.
    """
    for kind, style_choices in extensions.items():
        if kind not in SLOTS:
            raise ValueError(f"phrasings_llm: unknown kind {kind!r}")
        for style, choices in style_choices.items():
            if style not in VOCAB[kind]:
                raise ValueError(f"phrasings_llm: unknown style {kind}/{style}")
            final = SLOTS[kind][-1]
            if any(type(choice) is not str or not choice.strip() for choice in choices):
                raise ValueError(f"phrasings_llm: invalid choices: {kind}/{style}/{final}")
            vocab = VOCAB[kind][style]
            merged = vocab[final] + tuple(choices)
            if len(set(merged)) != len(merged):
                raise ValueError(f"phrasings_llm: duplicate choices: {kind}/{style}/{final}")
            vocab[final] = merged


def _load_extensions():
    """Return QUESTION_EXTENSIONS from the generated phrasings_llm module,
    or None when the bank carries no LLM-authored extensions."""
    name = f"{__package__}.phrasings_llm"
    if importlib.util.find_spec(name) is None:
        return None
    module = importlib.import_module(name)
    try:
        return module.QUESTION_EXTENSIONS
    except AttributeError as exc:
        raise ValueError("phrasings_llm.py must define QUESTION_EXTENSIONS") from exc


def _validate_bank() -> None:
    styles = ("msa", "egyptian", "levantine", "gulf", "code_switch", "english")
    for kind, slots in SLOTS.items():
        for style in styles:
            vocab = VOCAB[kind][style]
            if tuple(vocab) != slots:
                raise ValueError(f"bank slot mismatch: {kind}/{style}")
            for slot in slots:
                choices = vocab[slot]
                if not choices or any(
                        type(choice) is not str or not choice.strip() for choice in choices):
                    raise ValueError(f"empty/invalid choices: {kind}/{style}/{slot}")
                if len(set(choices)) != len(choices):
                    raise ValueError(f"duplicate choices: {kind}/{style}/{slot}")
            if phrase_count(kind, style) < 12:
                raise ValueError(f"bank too small: {kind}/{style}")


_EXTENSIONS = _load_extensions()
if _EXTENSIONS:
    _merge_extensions(_EXTENSIONS)
_validate_bank()
