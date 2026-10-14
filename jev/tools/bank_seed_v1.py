#!/usr/bin/env python3
"""Deterministic seed candidate tables for the prod-v1 bank author.

Used by ``tools/bank_author.py --source seed`` so the whole campaign can be
rehearsed without the datagen server.  Every composer returns MORE candidates
than its bank target, so the rejection-sampling engine still exercises its
validity and distance filters.  Seed material is deliberately disjoint from
the voice-v1 fixed tables (which the prod plan replaces precisely because
they were the exact-duplicate source): no seed string equals or closely
echoes a pinned v1 label, entity, fact name or value.  Provenance stays
honest - drafts that come from here are labeled source=seed.

Pure stdlib, no jev imports, byte-deterministic on every run.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

SEED = "prod-v1-2026-11"

REGIONS = ("egypt", "levant", "gulf", "maghreb", "neutral")
FORMALITIES = ("casual", "neutral", "formal")
URGENCIES = ("low", "normal", "high")


def domains() -> list[dict]:
    """16 routing-domain candidates (target 14): id, bilingual label, exactly
    four intents with three single-word bilingual keywords each, and three
    filler words that never collide with the domain's keywords."""
    return [
        {"id": "insurance", "label": {"ar": "التأمين", "en": "insurance services"},
         "intents": [
             {"id": "ins_claim", "label": {"ar": "تقديم مطالبة", "en": "filing a claim"},
              "keywords": [{"ar": "مطالبة", "en": "claim"}, {"ar": "تقديم", "en": "file"},
                           {"ar": "تعويض", "en": "compensation"}]},
             {"id": "ins_policy", "label": {"ar": "تجديد الوثيقة", "en": "policy renewal"},
              "keywords": [{"ar": "وثيقة", "en": "policy"}, {"ar": "تجديد", "en": "renewal"},
                           {"ar": "سريان", "en": "validity"}]},
             {"id": "ins_quote", "label": {"ar": "عرض سعر تأميني", "en": "an insurance quote"},
              "keywords": [{"ar": "عرض", "en": "quote"}, {"ar": "سعر", "en": "price"},
                           {"ar": "تسعير", "en": "quotation"}]},
             {"id": "ins_cover", "label": {"ar": "استفسار عن التغطية", "en": "coverage inquiry"},
              "keywords": [{"ar": "تغطية", "en": "coverage"}, {"ar": "استفسار", "en": "inquiry"},
                           {"ar": "استثناء", "en": "exclusion"}]}],
         "fillers": [{"ar": "شركة", "en": "company"}, {"ar": "مندوب", "en": "agent"},
                     {"ar": "مساعدة", "en": "help"}]},
        {"id": "utilities", "label": {"ar": "الخدمات العامة", "en": "utility services"},
         "intents": [
             {"id": "ut_bill", "label": {"ar": "فاتورة الكهرباء", "en": "the electricity bill"},
              "keywords": [{"ar": "فاتورة", "en": "bill"}, {"ar": "كهرباء", "en": "electricity"},
                           {"ar": "استهلاك", "en": "consumption"}]},
             {"id": "ut_outage", "label": {"ar": "بلاغ انقطاع", "en": "an outage report"},
              "keywords": [{"ar": "انقطاع", "en": "outage"}, {"ar": "بلاغ", "en": "report"},
                           {"ar": "عطل", "en": "fault"}]},
             {"id": "ut_move", "label": {"ar": "نقل الخدمة", "en": "service relocation"},
              "keywords": [{"ar": "نقل", "en": "relocation"}, {"ar": "عنوان", "en": "address"},
                           {"ar": "اشتراك", "en": "subscription"}]},
             {"id": "ut_meter", "label": {"ar": "قراءة العداد", "en": "meter reading"},
              "keywords": [{"ar": "عداد", "en": "meter"}, {"ar": "قراءة", "en": "reading"},
                           {"ar": "قياس", "en": "measurement"}]}],
         "fillers": [{"ar": "خدمة", "en": "service"}, {"ar": "سؤال", "en": "question"},
                     {"ar": "عاجل", "en": "urgent"}]},
        {"id": "education", "label": {"ar": "التعليم", "en": "education services"},
         "intents": [
             {"id": "ed_enroll", "label": {"ar": "تسجيل مقرر", "en": "course enrollment"},
              "keywords": [{"ar": "تسجيل", "en": "enrollment"}, {"ar": "مقرر", "en": "course"},
                           {"ar": "فصل", "en": "term"}]},
             {"id": "ed_records", "label": {"ar": "كشف الدرجات", "en": "a transcript request"},
              "keywords": [{"ar": "كشف", "en": "transcript"}, {"ar": "درجات", "en": "grades"},
                           {"ar": "شهادة", "en": "certificate"}]},
             {"id": "ed_fees", "label": {"ar": "الرسوم الدراسية", "en": "tuition fees"},
              "keywords": [{"ar": "رسوم", "en": "fees"}, {"ar": "دراسية", "en": "tuition"},
                           {"ar": "سداد", "en": "payment"}]},
             {"id": "ed_schedule", "label": {"ar": "تغيير الجدول", "en": "a schedule change"},
              "keywords": [{"ar": "جدول", "en": "schedule"}, {"ar": "تغيير", "en": "change"},
                           {"ar": "محاضرة", "en": "lecture"}]}],
         "fillers": [{"ar": "جامعة", "en": "university"}, {"ar": "طالب", "en": "student"},
                     {"ar": "قاعة", "en": "classroom"}]},
        {"id": "logistics", "label": {"ar": "الخدمات اللوجستية", "en": "logistics services"},
         "intents": [
             {"id": "lg_track", "label": {"ar": "متابعة الإرسالية", "en": "shipment tracking"},
              "keywords": [{"ar": "إرسالية", "en": "consignment"}, {"ar": "متابعة", "en": "tracking"},
                           {"ar": "وصول", "en": "arrival"}]},
             {"id": "lg_rates", "label": {"ar": "تسعير الشحن", "en": "freight pricing"},
              "keywords": [{"ar": "تسعير", "en": "pricing"}, {"ar": "وزن", "en": "weight"},
                           {"ar": "شحن", "en": "freight"}]},
             {"id": "lg_pickup", "label": {"ar": "جدولة الاستلام", "en": "pickup scheduling"},
              "keywords": [{"ar": "استلام", "en": "pickup"}, {"ar": "جدولة", "en": "scheduling"},
                           {"ar": "سائق", "en": "driver"}]},
             {"id": "lg_customs", "label": {"ar": "تخليص جمركي", "en": "customs clearance"},
              "keywords": [{"ar": "تخليص", "en": "clearance"}, {"ar": "جمركي", "en": "customs"},
                           {"ar": "بضاعة", "en": "cargo"}]}],
         "fillers": [{"ar": "شحنة", "en": "shipment"}, {"ar": "مستودع", "en": "warehouse"},
                     {"ar": "تسليم", "en": "delivery"}]},
        {"id": "automotive", "label": {"ar": "خدمات السيارات", "en": "automotive services"},
         "intents": [
             {"id": "am_service", "label": {"ar": "صيانة دورية", "en": "scheduled maintenance"},
              "keywords": [{"ar": "صيانة", "en": "maintenance"}, {"ar": "موعد", "en": "appointment"},
                           {"ar": "ميكانيكي", "en": "mechanic"}]},
             {"id": "am_repair", "label": {"ar": "إصلاح أعطال", "en": "fault repair"},
              "keywords": [{"ar": "إصلاح", "en": "repair"}, {"ar": "عطل", "en": "breakdown"},
                           {"ar": "ورشة", "en": "workshop"}]},
             {"id": "am_parts", "label": {"ar": "قطع غيار", "en": "spare parts"},
              "keywords": [{"ar": "قطع", "en": "parts"}, {"ar": "غيار", "en": "spares"},
                           {"ar": "أصلي", "en": "genuine"}]},
             {"id": "am_inspect", "label": {"ar": "فحص دوري", "en": "periodic inspection"},
              "keywords": [{"ar": "فحص", "en": "inspection"}, {"ar": "دوري", "en": "periodic"},
                           {"ar": "تقييم", "en": "assessment"}]}],
         "fillers": [{"ar": "سيارة", "en": "car"}, {"ar": "حجز", "en": "booking"},
                     {"ar": "فرع", "en": "branch"}]},
        {"id": "dining", "label": {"ar": "خدمات المطاعم", "en": "restaurant services"},
         "intents": [
             {"id": "dn_reserve", "label": {"ar": "حجز طاولة", "en": "a table reservation"},
              "keywords": [{"ar": "حجز", "en": "reservation"}, {"ar": "طاولة", "en": "table"},
                           {"ar": "عشاء", "en": "dinner"}]},
             {"id": "dn_delivery", "label": {"ar": "توصيل وجبات", "en": "meal delivery"},
              "keywords": [{"ar": "توصيل", "en": "delivery"}, {"ar": "وجبة", "en": "meal"},
                           {"ar": "طلب", "en": "order"}]},
             {"id": "dn_menu", "label": {"ar": "قائمة الطعام", "en": "menu questions"},
              "keywords": [{"ar": "قائمة", "en": "menu"}, {"ar": "أصناف", "en": "dishes"},
                           {"ar": "أسعار", "en": "prices"}]},
             {"id": "dn_allergy", "label": {"ar": "استفسار عن المكونات", "en": "allergen inquiry"},
              "keywords": [{"ar": "مكونات", "en": "ingredients"}, {"ar": "حساسية", "en": "allergy"},
                           {"ar": "خيارات", "en": "options"}]}],
         "fillers": [{"ar": "مطعم", "en": "restaurant"}, {"ar": "شكراً", "en": "thanks"},
                     {"ar": "هاتف", "en": "phone"}]},
        {"id": "aviation", "label": {"ar": "خدمات الطيران", "en": "aviation services"},
         "intents": [
             {"id": "av_book", "label": {"ar": "حجز رحلة", "en": "flight booking"},
              "keywords": [{"ar": "حجز", "en": "booking"}, {"ar": "رحلة", "en": "flight"},
                           {"ar": "درجة", "en": "cabin"}]},
             {"id": "av_checkin", "label": {"ar": "تسجيل الصعود", "en": "check-in"},
              "keywords": [{"ar": "تسجيل", "en": "check-in"}, {"ar": "صعود", "en": "boarding"},
                           {"ar": "بوابة", "en": "gate"}]},
             {"id": "av_baggage", "label": {"ar": "أمتعة السفر", "en": "baggage"},
              "keywords": [{"ar": "أمتعة", "en": "baggage"}, {"ar": "وزن", "en": "weight"},
                           {"ar": "سماح", "en": "allowance"}]},
             {"id": "av_refund", "label": {"ar": "استرداد التذكرة", "en": "ticket refund"},
              "keywords": [{"ar": "استرداد", "en": "refund"}, {"ar": "تذكرة", "en": "ticket"},
                           {"ar": "إلغاء", "en": "cancellation"}]}],
         "fillers": [{"ar": "شركة", "en": "airline"}, {"ar": "مطار", "en": "airport"},
                     {"ar": "مساعدة", "en": "assistance"}]},
        {"id": "subscriptions", "label": {"ar": "الاشتراكات", "en": "subscription services"},
         "intents": [
             {"id": "sb_upgrade", "label": {"ar": "ترقية الباقة", "en": "plan upgrade"},
              "keywords": [{"ar": "باقة", "en": "plan"}, {"ar": "ترقية", "en": "upgrade"},
                           {"ar": "بيانات", "en": "data"}]},
             {"id": "sb_pause", "label": {"ar": "تعليق الاشتراك", "en": "pausing a subscription"},
              "keywords": [{"ar": "تعليق", "en": "pause"}, {"ar": "إيقاف", "en": "hold"},
                           {"ar": "مؤقت", "en": "temporary"}]},
             {"id": "sb_invoice", "label": {"ar": "فاتورة الاشتراك", "en": "subscription invoice"},
              "keywords": [{"ar": "فاتورة", "en": "invoice"}, {"ar": "تجديد", "en": "renewal"},
                           {"ar": "مبلغ", "en": "amount"}]},
             {"id": "sb_cancel", "label": {"ar": "إلغاء الاشتراك", "en": "cancelling"},
              "keywords": [{"ar": "إلغاء", "en": "cancel"}, {"ar": "إنهاء", "en": "terminate"},
                           {"ar": "انسحاب", "en": "withdraw"}]}],
         "fillers": [{"ar": "حساب", "en": "account"}, {"ar": "مزايا", "en": "perks"},
                     {"ar": "سؤال", "en": "question"}]},
        {"id": "events", "label": {"ar": "الفعاليات", "en": "event services"},
         "intents": [
             {"id": "ev_tickets", "label": {"ar": "شراء التذاكر", "en": "buying tickets"},
              "keywords": [{"ar": "تذاكر", "en": "tickets"}, {"ar": "شراء", "en": "purchase"},
                           {"ar": "مقاعد", "en": "seats"}]},
             {"id": "ev_schedule", "label": {"ar": "جدول الفعاليات", "en": "event schedule"},
              "keywords": [{"ar": "جدول", "en": "schedule"}, {"ar": "موعد", "en": "timing"},
                           {"ar": "برنامج", "en": "program"}]},
             {"id": "ev_refund", "label": {"ar": "استرجاع قيمة التذكرة", "en": "ticket reimbursement"},
              "keywords": [{"ar": "استرجاع", "en": "reimbursement"}, {"ar": "قيمة", "en": "value"},
                           {"ar": "تذكرة", "en": "ticket"}]},
             {"id": "ev_venue", "label": {"ar": "معلومات المكان", "en": "venue information"},
              "keywords": [{"ar": "مكان", "en": "venue"}, {"ar": "قاعة", "en": "hall"},
                           {"ar": "دخول", "en": "entry"}]}],
         "fillers": [{"ar": "فعالية", "en": "event"}, {"ar": "حضور", "en": "attendance"},
                     {"ar": "تاريخ", "en": "date"}]},
        {"id": "real_estate", "label": {"ar": "العقار", "en": "real-estate services"},
         "intents": [
             {"id": "re_view", "label": {"ar": "معاينة عقار", "en": "property viewing"},
              "keywords": [{"ar": "معاينة", "en": "viewing"}, {"ar": "عقار", "en": "property"},
                           {"ar": "زيارة", "en": "visit"}]},
             {"id": "re_rent", "label": {"ar": "عقد إيجار", "en": "lease contract"},
              "keywords": [{"ar": "إيجار", "en": "lease"}, {"ar": "عقد", "en": "contract"},
                           {"ar": "سعر", "en": "rent"}]},
             {"id": "re_fix", "label": {"ar": "صيانة العقار", "en": "property maintenance"},
              "keywords": [{"ar": "صيانة", "en": "maintenance"}, {"ar": "بلاغ", "en": "report"},
                           {"ar": "مالك", "en": "owner"}]},
             {"id": "re_docs", "label": {"ar": "أوراق التوثيق", "en": "title documents"},
              "keywords": [{"ar": "أوراق", "en": "documents"}, {"ar": "توثيق", "en": "deeds"},
                           {"ar": "صك", "en": "title"}]}],
         "fillers": [{"ar": "شقة", "en": "apartment"}, {"ar": "وسيط", "en": "broker"},
                     {"ar": "حي", "en": "district"}]},
        {"id": "payments", "label": {"ar": "المدفوعات", "en": "payment services"},
         "intents": [
             {"id": "pm_transfer", "label": {"ar": "تحويل أموال", "en": "money transfer"},
              "keywords": [{"ar": "تحويل", "en": "transfer"}, {"ar": "مستفيد", "en": "beneficiary"},
                           {"ar": "رسوم", "en": "fee"}]},
             {"id": "pm_refund", "label": {"ar": "استرداد مبلغ", "en": "refunding a payment"},
              "keywords": [{"ar": "استرداد", "en": "refund"}, {"ar": "مبلغ", "en": "amount"},
                           {"ar": "مرتجع", "en": "chargeback"}]},
             {"id": "pm_limit", "label": {"ar": "حد التحويل اليومي", "en": "transfer limits"},
              "keywords": [{"ar": "حد", "en": "limit"}, {"ar": "يومي", "en": "daily"},
                           {"ar": "سقف", "en": "cap"}]},
             {"id": "pm_dispute", "label": {"ar": "اعتراض على خصم", "en": "disputing a charge"},
              "keywords": [{"ar": "خصم", "en": "charge"}, {"ar": "اعتراض", "en": "dispute"},
                           {"ar": "عملية", "en": "transaction"}]}],
         "fillers": [{"ar": "محفظة", "en": "wallet"}, {"ar": "رقم", "en": "number"},
                     {"ar": "عاجل", "en": "urgent"}]},
        {"id": "delivery", "label": {"ar": "التوصيل", "en": "delivery services"},
         "intents": [
             {"id": "dl_track", "label": {"ar": "تتبع الطلب", "en": "order tracking"},
              "keywords": [{"ar": "تتبع", "en": "tracking"}, {"ar": "طلب", "en": "order"},
                           {"ar": "حالة", "en": "status"}]},
             {"id": "dl_address", "label": {"ar": "تغيير عنوان التسليم", "en": "address change"},
              "keywords": [{"ar": "عنوان", "en": "address"}, {"ar": "تغيير", "en": "change"},
                           {"ar": "تسليم", "en": "delivery"}]},
             {"id": "dl_retry", "label": {"ar": "إعادة محاولة التوصيل", "en": "failed redelivery"},
              "keywords": [{"ar": "فاشلة", "en": "failed"}, {"ar": "إعادة", "en": "redelivery"},
                           {"ar": "محاولة", "en": "attempt"}]},
             {"id": "dl_zone", "label": {"ar": "رسوم المناطق", "en": "zone fees"},
              "keywords": [{"ar": "منطقة", "en": "zone"}, {"ar": "رسوم", "en": "fees"},
                           {"ar": "سعر", "en": "price"}]}],
         "fillers": [{"ar": "مندوب", "en": "courier"}, {"ar": "شكراً", "en": "thanks"},
                     {"ar": "خدمة", "en": "service"}]},
        {"id": "fitness", "label": {"ar": "اللياقة", "en": "fitness services"},
         "intents": [
             {"id": "ft_join", "label": {"ar": "عضوية جديدة", "en": "new membership"},
              "keywords": [{"ar": "عضوية", "en": "membership"}, {"ar": "تسجيل", "en": "signup"},
                           {"ar": "انضمام", "en": "joining"}]},
             {"id": "ft_classes", "label": {"ar": "حصص تدريبية", "en": "training classes"},
              "keywords": [{"ar": "حصة", "en": "class"}, {"ar": "مواعيد", "en": "timetable"},
                           {"ar": "استوديو", "en": "studio"}]},
             {"id": "ft_freeze", "label": {"ar": "تجميد العضوية", "en": "freezing a membership"},
              "keywords": [{"ar": "تجميد", "en": "freeze"}, {"ar": "إيقاف", "en": "suspend"},
                           {"ar": "سفر", "en": "travel"}]},
             {"id": "ft_coach", "label": {"ar": "مدرّب شخصي", "en": "personal training"},
              "keywords": [{"ar": "تدريب", "en": "training"}, {"ar": "خاص", "en": "personal"},
                           {"ar": "برنامج", "en": "program"}]}],
         "fillers": [{"ar": "نادي", "en": "gym"}, {"ar": "وقت", "en": "hours"},
                     {"ar": "عروض", "en": "offers"}]},
        {"id": "municipal", "label": {"ar": "خدمات البلدية", "en": "municipal services"},
         "intents": [
             {"id": "mu_permit", "label": {"ar": "تصريح أعمال", "en": "a works permit"},
              "keywords": [{"ar": "تصريح", "en": "permit"}, {"ar": "أعمال", "en": "works"},
                           {"ar": "ترخيص", "en": "licensing"}]},
             {"id": "mu_clean", "label": {"ar": "شكاوى النظافة", "en": "sanitation complaints"},
              "keywords": [{"ar": "نظافة", "en": "sanitation"}, {"ar": "شكوى", "en": "complaint"},
                           {"ar": "تجميع", "en": "collection"}]},
             {"id": "mu_parking", "label": {"ar": "المواقف العامة", "en": "public parking"},
              "keywords": [{"ar": "مواقف", "en": "parking"}, {"ar": "مركبة", "en": "vehicle"},
                           {"ar": "غرامة", "en": "fine"}]},
             {"id": "mu_report", "label": {"ar": "الإبلاغ عن مخالفة", "en": "reporting a violation"},
              "keywords": [{"ar": "إبلاغ", "en": "reporting"}, {"ar": "مخالفة", "en": "violation"},
                           {"ar": "رصد", "en": "noticing"}]}],
         "fillers": [{"ar": "بلدية", "en": "municipality"}, {"ar": "حي", "en": "neighborhood"},
                     {"ar": "خدمة", "en": "service"}]},
        {"id": "transport", "label": {"ar": "النقل العام", "en": "public transport"},
         "intents": [
             {"id": "tr_card", "label": {"ar": "بطاقة النقل", "en": "transit card"},
              "keywords": [{"ar": "بطاقة", "en": "card"}, {"ar": "رصيد", "en": "balance"},
                           {"ar": "تعبئة", "en": "top-up"}]},
             {"id": "tr_route", "label": {"ar": "مسارات الخطوط", "en": "route information"},
              "keywords": [{"ar": "مسار", "en": "route"}, {"ar": "محطة", "en": "station"},
                           {"ar": "خط", "en": "line"}]},
             {"id": "tr_delay", "label": {"ar": "تأخر الرحلات", "en": "delayed trips"},
              "keywords": [{"ar": "تأخر", "en": "delayed"}, {"ar": "رحلة", "en": "trip"},
                           {"ar": "إشعار", "en": "notice"}]},
             {"id": "tr_pass", "label": {"ar": "اشتراك شهري", "en": "a monthly pass"},
              "keywords": [{"ar": "اشتراك", "en": "pass"}, {"ar": "شهري", "en": "monthly"},
                           {"ar": "تجديد", "en": "renewal"}]}],
         "fillers": [{"ar": "حافلة", "en": "bus"}, {"ar": "تذكرة", "en": "ticket"},
                     {"ar": "استفسار", "en": "inquiry"}]},
        {"id": "hospitality", "label": {"ar": "الضيافة", "en": "hospitality services"},
         "intents": [
             {"id": "hs_checkin", "label": {"ar": "تسجيل الوصول", "en": "hotel check-in"},
              "keywords": [{"ar": "تسجيل", "en": "check-in"}, {"ar": "وصول", "en": "arrival"},
                           {"ar": "غرفة", "en": "room"}]},
             {"id": "hs_checkout", "label": {"ar": "تسجيل المغادرة", "en": "checkout"},
              "keywords": [{"ar": "مغادرة", "en": "departure"}, {"ar": "فاتورة", "en": "bill"},
                           {"ar": "تأخير", "en": "late"}]},
             {"id": "hs_room", "label": {"ar": "خدمة الغرف", "en": "room service"},
              "keywords": [{"ar": "خدمة", "en": "service"}, {"ar": "طلب", "en": "order"},
                           {"ar": "توصيل", "en": "delivery"}]},
             {"id": "hs_amenity", "label": {"ar": "مرافق الفندق", "en": "hotel amenities"},
              "keywords": [{"ar": "مسبح", "en": "pool"}, {"ar": "صالة", "en": "lounge"},
                           {"ar": "موقف", "en": "parking"}]}],
         "fillers": [{"ar": "فندق", "en": "hotel"}, {"ar": "إقامة", "en": "stay"},
                     {"ar": "عاجل", "en": "urgent"}]},
    ]


_REGION_CITY = {"egypt": ("القاهرة", "Cairo"), "levant": ("بيروت", "Beirut"),
                "gulf": ("الدمام", "Dammam"), "maghreb": ("الرباط", "Rabat"),
                "neutral": ("الرياض", "Riyadh")}
_TONE = {"formal": ("نبرة رسمية", "a formal tone"),
         "casual": ("نبرة ودية", "a friendly tone"),
         "neutral": ("نبرة عادية", "a matter-of-fact tone")}
_PACE = {"high": ("الطلب عاجل", "the request is urgent"),
         "normal": ("الطلب عادي", "a routine request"),
         "low": ("لا استعجال في الطلب", "no rush on the request")}
_CALLER = (("عميل", "customer"), ("مستخدم", "user"), ("متصل", "caller"))


def personas() -> list[dict]:
    """45 persona candidates (target 25), interleaved so any prefix keeps all
    regions represented; each (region, formality, urgency) triple is unique."""
    out = []
    for index in range(len(REGIONS) * len(FORMALITIES) * len(URGENCIES)):
        region = REGIONS[index % len(REGIONS)]
        formality = FORMALITIES[(index // len(REGIONS)) % len(FORMALITIES)]
        urgency = URGENCIES[(index // (len(REGIONS) * len(FORMALITIES)))
                            % len(URGENCIES)]
        city_ar, city_en = _REGION_CITY[region]
        tone_ar, tone_en = _TONE[formality]
        pace_ar, pace_en = _PACE[urgency]
        caller_ar, caller_en = _CALLER[index % len(_CALLER)]
        out.append({
            "id": f"persona-{index + 1:02d}",
            "region": region, "formality": formality, "urgency": urgency,
            "note_ar": f"{caller_ar} من {city_ar}، {tone_ar}، {pace_ar}",
            "note_en": f"{caller_en} from {city_en}, {tone_en}, {pace_en}",
        })
    return out


def _grid(heads: tuple, tails: tuple, join: str = " ") -> list[dict]:
    """Bilingual cross-product, shuffled deterministically per call site."""
    return [{"ar": f"{head[0]}{join}{tail[0]}", "en": f"{tail[1]} {head[1]}"}
            for head in heads for tail in tails]


_COMPARISON_HEADS = (("طلب", "request"), ("عقد", "contract"), ("فاتورة", "invoice"),
                     ("رصيد", "balance"), ("حد", "limit"), ("تقرير", "report"),
                     ("خطة", "plan"), ("قسط", "installment"), ("معدل", "rate"),
                     ("نسبة", "share"), ("كمية", "quantity"), ("شهادة", "certificate"))
_COMPARISON_TAILS = (("التمويل", "financing"), ("الإيجار", "rental"),
                     ("التأمين", "insurance"), ("الصيانة", "maintenance"),
                     ("الاشتراك", "subscription"), ("التوريد", "supply"),
                     ("الترخيص", "licensing"), ("التدريب", "training"),
                     ("الشحن", "shipping"), ("الاستثمار", "investment"),
                     ("الخدمة", "service"), ("المشتريات", "procurement"))


def comparison_entities() -> list[dict]:
    rng = random.Random(f"{SEED}:entities:comparison")
    combos = _grid(_COMPARISON_HEADS, _COMPARISON_TAILS)
    rng.shuffle(combos)
    return combos


_POLICY_PREFIX = (("طلب استرداد", "refund request"), ("طلب تسجيل", "registration request"),
                  ("طلب تغطية", "coverage request"), ("طلب ضمان", "guarantee request"),
                  ("طلب ترقية", "upgrade request"), ("طلب إعفاء", "exemption request"),
                  ("طلب تصحيح", "correction request"), ("طلب نقل", "transfer request"))
_POLICY_OBJECTS = (("مشتريات", "purchases"), ("طالب جديد", "a new student"),
                   ("ممتلكات", "property"), ("قرض", "a loan"),
                   ("خدمة مميزة", "premium service"), ("رسوم", "fees"),
                   ("سجل", "records"), ("ملكية", "ownership"))


def policy_entities() -> list[dict]:
    rng = random.Random(f"{SEED}:entities:policy_logic")
    combos = [{"ar": f"{prefix[0]} {obj[0]}", "en": f"{obj[1]} {prefix[1]}"}
              for prefix in _POLICY_PREFIX for obj in _POLICY_OBJECTS]
    rng.shuffle(combos)
    return combos


_BUDGET_HEADS = (("طلب صرف", "spend request"), ("أمر شراء", "purchase order"),
                 ("خطة تقسيط", "installment plan"), ("فاتورة", "invoice"),
                 ("طلب تعويض", "compensation request"), ("مشروع", "project"),
                 ("عقد صيانة", "maintenance contract"), ("طلب دفعة", "payment request"))
_BUDGET_QUALIFIERS = (("إضافي", "additional"), ("جديد", "new"), ("مقترح", "proposed"),
                      ("سنوي", "annual"), ("طارئ", "emergency"), ("مؤجل", "deferred"),
                      ("مبدئي", "preliminary"))


def budget_entities() -> list[dict]:
    rng = random.Random(f"{SEED}:entities:budget_check")
    combos = [{"ar": f"{head[0]} {qual[0]}", "en": f"{qual[1]} {head[1]}"}
              for head in _BUDGET_HEADS for qual in _BUDGET_QUALIFIERS]
    rng.shuffle(combos)
    return combos


_SAFETY_ACTIONS = (("سؤال عن", "a question about"), ("طلب", "a request for"),
                   ("استفسار حول", "an inquiry about"), ("تعديل", "editing"),
                   ("إلغاء", "cancelling"), ("تحويل", "transferring"),
                   ("حذف", "deleting"), ("مشاركة", "sharing"))
_SAFETY_TOPICS = (("جرعة دواء", "a medication dose"), ("بيانات حساسة", "sensitive data"),
                  ("حساب مسروق", "a hijacked account"), ("مبلغ كبير", "a large amount"),
                  ("وصفة طبية", "a medical prescription"), ("تقرير طبي", "a medical report"),
                  ("عقد موقع", "a signed contract"), ("بيانات عميل", "customer data"),
                  ("عمولة تحويل", "a transfer fee"), ("إشعار أمني", "a security alert"))


def safety_entities() -> list[dict]:
    rng = random.Random(f"{SEED}:entities:safety_gate")
    combos = [{"ar": f"{action[0]} {topic[0]}", "en": f"{action[1]} {topic[1]}"}
              for action in _SAFETY_ACTIONS for topic in _SAFETY_TOPICS]
    rng.shuffle(combos)
    return combos


_TRIAGE_HEADS = (("تذكرة دعم", "support ticket"), ("بلاغ عطل", "fault report"),
                 ("شكوى خدمة", "service complaint"), ("طلب صيانة", "maintenance request"),
                 ("بلاغ أمني", "security report"), ("طلب استرجاع", "recovery request"),
                 ("تقرير حادث", "incident report"), ("طلب مراجعة", "review request"))
_TRIAGE_QUALIFIERS = (("عاجل", "urgent"), ("فني", "technical"), ("متكرر", "recurring"),
                      ("مُصدَّر", "escalated"), ("داخلي", "internal"),
                      ("خارجي", "external"), ("معلّق", "pending"))


def triage_entities() -> list[dict]:
    rng = random.Random(f"{SEED}:entities:triage_rubric")
    combos = [{"ar": f"{head[0]} {qual[0]}", "en": f"{qual[1]} {head[1]}"}
              for head in _TRIAGE_HEADS for qual in _TRIAGE_QUALIFIERS]
    rng.shuffle(combos)
    return combos


def entities(kind: str) -> list[dict]:
    return {"comparison": comparison_entities, "policy_logic": policy_entities,
            "budget_check": budget_entities, "safety_gate": safety_entities,
            "triage_rubric": triage_entities}[kind]()


# (fact, ar, en, type, op, value) - all names and labels fresh vs voice-v1.
_POLICY_FACTS = (
    ("days_since_purchase", "أيام منذ الشراء", "days since purchase", "number", "<=", 14),
    ("package_intact", "العبوة سليمة", "package intact", "bool", "==", True),
    ("receipt_kept", "الاحتفاظ بالإيصال", "receipt kept", "bool", "==", True),
    ("applicant_age", "عمر المتقدم", "applicant age", "number", ">=", 18),
    ("guardian_approval", "موافقة ولي الأمر موثقة", "documented guardian approval",
     "bool", "==", True),
    ("property_years", "عمر الممتلكات بالسنوات", "property age in years", "number", "<=", 30),
    ("claim_free", "سجل خالٍ من المطالبات", "claim-free record", "bool", "==", True),
    ("cover_active", "الغطاء ساري", "active cover", "bool", "==", True),
    ("income_monthly", "الدخل الشهري", "monthly income", "number", ">=", 5000),
    ("guarantor_present", "كفيل متاح", "guarantor available", "bool", "==", True),
    ("subscription_months", "أشهر الاشتراك", "months subscribed", "number", ">=", 6),
    ("account_balance", "رصيد الحساب", "account balance", "number", ">=", 100),
    ("prior_repairs", "عدد الإصلاحات السابقة", "prior repairs", "number", "<=", 2),
    ("email_verified", "بريد إلكتروني موثق", "verified email", "bool", "==", True),
    ("terms_accepted", "قبول الشروط", "terms accepted", "bool", "==", True),
    ("credit_score", "درجة الائتمان", "credit score", "number", ">=", 600),
    ("loan_term_months", "مدة التمويل بالأشهر", "financing term in months",
     "number", "<=", 60),
    ("deposit_paid", "سداد الدفعة المقدمة", "deposit paid", "bool", "==", True),
    ("license_current", "رخصة سارية", "current license", "bool", "==", True),
    ("experience_years", "سنوات الخبرة", "years of experience", "number", ">=", 3),
    ("enrollment_active", "تسجيل ساري", "active enrollment", "bool", "==", True),
    ("min_score_met", "تحقيق الحد الأدنى", "minimum score met", "bool", "==", True),
    ("distance_km", "المسافة بالكيلومترات", "distance in km", "number", "<=", 50),
    ("insurance_current", "تأمين ساري", "current insurance", "bool", "==", True),
    ("verification_code", "رمز التحقق مستلم", "verification code received",
     "bool", "==", True),
    ("shipment_days", "أيام منذ الشحن", "days since shipment", "number", "<=", 14),
)


def policy_facts() -> list[dict]:
    return [{"fact": fact, "ar": ar, "en": en, "type": type_, "op": op, "value": value}
            for fact, ar, en, type_, op, value in _POLICY_FACTS]


# (fact, ar, en, type) - names and labels fresh vs voice-v1's SAFETY_FACT_LABELS.
_SAFETY_FACTS = (
    ("consent_given", "موافقة مُسجّلة", "recorded consent", "bool"),
    ("clinician_review", "مراجعة سريرية", "clinical review", "bool"),
    ("transfer_value", "قيمة التحويل", "transfer value", "number"),
    ("identity_proof", "إثبات الهوية", "proof of identity", "bool"),
    ("otp_passed", "اجتياز الرمز المؤقت", "one-time code passed", "bool"),
    ("threat_score", "درجة التهديد", "threat score", "number"),
    ("caller_age", "عمر المتصل", "caller age", "number"),
    ("external_party", "طرف خارجي", "an external party", "bool"),
    ("prescription_valid", "وصفة معتمدة", "approved prescription", "bool"),
    ("dosage_recorded", "جرعة مسجلة", "recorded dosage", "bool"),
    ("wallet_balance", "رصيد المحفظة", "wallet balance", "number"),
    ("location_confirmed", "تأكيد الموقع", "confirmed location", "bool"),
    ("trusted_device", "جهاز موثوق", "trusted device", "bool"),
    ("idle_days", "أيام دون نشاط", "idle days", "number"),
    ("transfers_today", "تحويلات اليوم", "transfers today", "number"),
    ("document_signed", "مستند موقع", "signed document", "bool"),
    ("caller_minor", "المتصل قاصر", "caller is a minor", "bool"),
    ("account_flagged", "حساب مؤشر", "flagged account", "bool"),
    ("code_verified", "رمز موثق", "verified code", "bool"),
    ("credit_cap", "سقف الائتمان", "credit cap", "number"),
    ("hours_pending", "ساعات بانتظار الرد", "hours awaiting reply", "number"),
    ("camera_grant", "إذن الكاميرا", "camera permission", "bool"),
    ("export_requested", "طلب تصدير البيانات", "data export requested", "bool"),
    ("admin_rights", "صلاحيات إدارية", "administrative rights", "bool"),
    ("session_open", "جلسة مفتوحة", "open session", "bool"),
    ("failed_attempts", "محاولات فاشلة", "failed attempts", "number"),
)


def safety_facts() -> list[dict]:
    return [{"fact": fact, "ar": ar, "en": en, "type": type_}
            for fact, ar, en, type_ in _SAFETY_FACTS]


def fact_catalog(kind: str) -> list[dict]:
    return {"policy_logic": policy_facts, "safety_gate": safety_facts}[kind]()


_TRIAGE_ITEMS = (
    ("production", "أثر على بيئة الإنتاج", "production environment affected"),
    ("integration", "تعطل التكامل", "a broken integration"),
    ("payment_flow", "توقف تدفق المدفوعات", "payment flow stopped"),
    ("login_outage", "تعطل تسجيل الدخول", "login outage"),
    ("reporting", "تأخر التقارير", "delayed reporting"),
    ("migration", "تعثر ترحيل البيانات", "a stalled data migration"),
    ("backup_stale", "نسخ احتياطي قديم", "a stale backup"),
    ("security_patch", "رقعة أمنية غائبة", "a missing security patch"),
    ("capacity", "قرب بلوغ السعة", "capacity nearly exhausted"),
    ("slowdown", "تباطؤ الأداء", "performance degradation"),
    ("corruption", "تلف بيانات جزئي", "partial data corruption"),
    ("vendor_down", "مزود خارجي متوقف", "an unresponsive vendor"),
    ("legal_hold", "تعليق قانوني", "a legal hold"),
    ("audit_finding", "ملاحظة تدقيق", "an audit finding"),
    ("untrained_team", "فريق غير مدرّب", "an untrained team"),
    ("coverage_gap", "غياب الفريق المناوب", "on-call coverage gap"),
    ("duplicates", "سجلات مكررة", "duplicate records"),
    ("sync_failure", "فشل المزامنة", "a sync failure"),
    ("notifications", "إشعارات متوقفة", "notifications stopped"),
    ("localization", "خلل الترجمة", "a localization defect"),
    ("permissions", "صلاحيات خاطئة", "wrong permissions"),
    ("contract_risk", "عقد على المحك", "a contract at risk"),
    ("churn_risk", "خطر فقدان عميل", "churn risk"),
    ("inventory_gap", "جرد غير مطابق", "inventory mismatch"),
    ("shipment_delay", "تأخر الشحنات", "shipment delays"),
    ("double_booking", "حجوزات مزدوجة", "double bookings"),
    ("refund_backlog", "طلبات استرداد متراكمة", "a refund backlog"),
    ("api_outage", "انقطاع واجهة برمجية", "an API outage"),
    ("app_down", "تعطل تطبيق الجوال", "the mobile app down"),
    ("printing", "توقف الطباعة", "printing stopped"),
    ("internal_network", "انقطاع الشبكة الداخلية", "an internal network outage"),
    ("scheduler", "تعطل المجدول", "the scheduler stuck"),
    ("queue_stuck", "طابور معلق", "a stuck queue"),
    ("cache_cold", "ذاكرة مؤقتة فارغة", "a cold cache"),
    ("timezone_off", "فرق توقيت خاطئ", "a wrong timezone"),
    ("locale_mixed", "لغات مختلطة في الواجهة", "mixed interface languages"),
    ("upload_failed", "فشل الرفع", "failed uploads"),
    ("search_broken", "تعطل البحث", "broken search"),
    ("map_errors", "أخطاء الخرائط", "map errors"),
    ("billing_drift", "فروق في الفوترة", "billing drift"),
    ("seat_limit", "بلوغ حد المقاعد", "seat limit reached"),
    ("roster_conflict", "تعارض الجدول", "a roster conflict"),
    ("access_revoked", "صلاحيات ملغاة", "revoked access"),
    ("cert_expiring", "شهادة تنتهي قريبًا", "a certificate expiring soon"),
)


def triage_items() -> list[dict]:
    return [{"id": item, "ar": ar, "en": en} for item, ar, en in _TRIAGE_ITEMS]


_OPENS = ("06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
          "13:00", "16:00", "20:00", "22:00")
_CLOSES = ("12:00", "14:00", "15:00", "17:00", "18:00", "20:00",
           "22:00", "23:00", "01:00", "02:00")


def temporal_profiles() -> list[dict]:
    """98 opening-hour profiles (target 40); opens > closes crosses midnight,
    exactly as the temporal rule supports."""
    return [{"opens": opens, "closes": closes}
            for opens in _OPENS for closes in _CLOSES if opens != closes]


def weekend_conventions() -> list[dict]:
    return [{"weekend": ["friday", "saturday"]},
            {"weekend": ["thursday", "friday"]},
            {"weekend": ["saturday", "sunday"]},
            {"weekend": ["friday"]},
            {"weekend": ["sunday"]}]


_FIRSTS = (("محمد", "Mohammed"), ("أحمد", "Ahmed"), ("خالد", "Khalid"),
           ("عمر", "Omar"), ("يوسف", "Yousef"), ("فيصل", "Faisal"),
           ("عبدالله", "Abdullah"), ("سلطان", "Sultan"), ("سارة", "Sarah"),
           ("نورة", "Noura"), ("منى", "Mona"), ("هند", "Hind"),
           ("لطيفة", "Latifa"), ("ريم", "Reem"), ("دلال", "Dalal"),
           ("فاطمة", "Fatima"), ("عبدالعزيز", "Abdulaziz"), ("ناصر", "Nasser"),
           ("ماجد", "Majed"), ("تركي", "Turki"))
_FAMILIES = (("العتيبي", "Alotaibi"), ("القحطاني", "Alqahtani"),
             ("الشهري", "Alshehri"), ("الحربي", "Alharbi"),
             ("المطيري", "Almutairi"), ("الدوسري", "Aldosari"),
             ("الغامدي", "Alghamdi"), ("الزهراني", "Alzahrani"),
             ("الشمري", "Alshammari"), ("العنزي", "Aleanzi"),
             ("السبيعي", "Alsubaie"), ("البقمي", "Albuqami"),
             ("الرشيد", "Alrashid"), ("المالكي", "Almaliki"),
             ("الجهني", "Aljuhani"), ("البلوي", "Albalawi"),
             ("الخالدي", "Alkhalidi"), ("العمري", "Alomri"),
             ("الفهد", "Alfahad"), ("الصالح", "Alsalih"))


def names() -> list[dict]:
    rng = random.Random(f"{SEED}:values:names")
    combos = [{"ar": f"{first[0]} {family[0]}", "en": f"{first[1]} {family[1]}"}
              for first in _FIRSTS for family in _FAMILIES]
    rng.shuffle(combos)
    return combos


_CITIES = (
    ("الرياض", "Riyadh"), ("جدة", "Jeddah"), ("مكة المكرمة", "Mecca"),
    ("المدينة المنورة", "Medina"), ("الدمام", "Dammam"), ("الخبر", "Khobar"),
    ("الظهران", "Dhahran"), ("الأحساء", "Al-Ahsa"), ("الطائف", "Taif"),
    ("تبوك", "Tabuk"), ("بريدة", "Buraidah"), ("أبها", "Abha"),
    ("خميس مشيط", "Khamis Mushait"), ("نجران", "Najran"), ("جازان", "Jazan"),
    ("حائل", "Hail"), ("ينبع", "Yanbu"), ("الجبيل", "Jubail"),
    ("القطيف", "Qatif"), ("حفر الباطن", "Hafar Al-Batin"), ("سكاكا", "Sakaka"),
    ("عرعر", "Arar"), ("الباحة", "Al-Baha"), ("الجوف", "Al-Jouf"),
    ("الكويت", "Kuwait City"), ("المنامة", "Manama"), ("الدوحة", "Doha"),
    ("مسقط", "Muscat"), ("صلالة", "Salalah"), ("أبوظبي", "Abu Dhabi"),
    ("دبي", "Dubai"), ("الشارقة", "Sharjah"), ("عجمان", "Ajman"),
    ("رأس الخيمة", "Ras Al Khaimah"), ("الفجيرة", "Fujairah"),
    ("أم القيوين", "Umm Al Quwain"), ("القاهرة", "Cairo"),
    ("الإسكندرية", "Alexandria"), ("الجيزة", "Giza"), ("أسيوط", "Asyut"),
    ("الأقصر", "Luxor"), ("أسوان", "Aswan"), ("بيروت", "Beirut"),
    ("طرابلس", "Tripoli"), ("عمان", "Amman"), ("إربد", "Irbid"),
    ("بغداد", "Baghdad"), ("البصرة", "Basra"), ("الموصل", "Mosul"),
    ("دمشق", "Damascus"), ("حلب", "Aleppo"), ("حمص", "Homs"),
    ("اللاذقية", "Latakia"), ("الرباط", "Rabat"), ("الدار البيضاء", "Casablanca"),
    ("مراكش", "Marrakesh"), ("فاس", "Fez"), ("طنجة", "Tangier"),
    ("أكادير", "Agadir"), ("تونس", "Tunis"), ("الجزائر", "Algiers"),
    ("وهران", "Oran"), ("قسنطينة", "Constantine"), ("الخرطوم", "Khartoum"),
    ("صنعاء", "Sanaa"), ("عدن", "Aden"),
)


def cities() -> list[dict]:
    return [{"ar": ar, "en": en} for ar, en in _CITIES]


_STREETS = (("الملك فهد", "King Fahd"), ("الأمير سلطان", "Prince Sultan"),
            ("العليا", "Olaya"), ("التحرير", "Tahrir"),
            ("الجمهورية", "Jamhouriya"), ("الاستقلال", "Istiqlal"),
            ("النخيل", "Nakheel"), ("الحمراء", "Hamra"),
            ("الأمير محمد", "Prince Mohammed"), ("الصفا", "As-Safa"))


def addresses() -> list[dict]:
    out = []
    for street_ar, street_en in _STREETS:
        for city_ar, city_en in _CITIES[:4]:
            out.append({"ar": f"شارع {street_ar}، {city_ar}",
                        "en": f"{street_en} Street, {city_en}"})
    return out


def phones() -> list[dict]:
    rng = random.Random(f"{SEED}:values:phones")
    seen: set[str] = set()
    out: list[dict] = []
    while len(out) < 200:
        phone = f"05{rng.randrange(10 ** 8):08d}"
        if phone not in seen:
            seen.add(phone)
            out.append({"phone": phone})
    return out


def dates() -> list[dict]:
    rng = random.Random(f"{SEED}:values:dates")
    start = date(2026, 1, 1)
    offsets = rng.sample(range(730), 200)
    return [{"date": (start + timedelta(days=offset)).isoformat()}
            for offset in sorted(offsets)]


def amounts() -> list[dict]:
    rng = random.Random(f"{SEED}:values:amounts")
    seen: set[str] = set()
    out: list[dict] = []
    while len(out) < 200:
        value = rng.randrange(1000, 9_900_000)
        amount = f"{value // 100}.{value % 100:02d}"
        if amount not in seen:
            seen.add(amount)
            out.append({"amount": amount})
    return out


def emails() -> list[dict]:
    return [{"email": f"user{index:03d}@example.com"} for index in range(1, 61)]


def references() -> list[dict]:
    rng = random.Random(f"{SEED}:values:references")
    seen: set[str] = set()
    out: list[dict] = []
    while len(out) < 60:
        reference = f"REF-{rng.randrange(10 ** 6):06d}"
        if reference not in seen:
            seen.add(reference)
            out.append({"reference": reference})
    return out
