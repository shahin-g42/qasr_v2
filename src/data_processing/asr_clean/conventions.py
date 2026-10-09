"""Per-language formatting and ITN conventions, plus the deterministic canon.

Two halves:

- ``FORMAT`` / ``ITN``: what the corrector LLM is told. One complete
  specification per language, written for SPOKEN-language targets: numbers
  the speaker said as a quantity become digits; numbers that are words of
  the language (articles, idioms, ordinals, approximations, names,
  scripture) stay words. Never infer what was not said.
- ``canonicalize()``: what is fixed in code, after the LLM, because it must
  be identical in every record and the model drifts on it (test3,
  2026-10-09): "%" vs "٪" (112 vs 652 records), fathatan on the alef vs
  before it (905 vs 1,892 originals), Malayalam "ൻ്റ" vs "ന്റ" (two
  encodings of one sound), legacy ZWJ chillus, Arabic-Indic/Devanagari
  digits, "|" for danda, tatweel, non-speech tags.
"""

from __future__ import annotations

import re
import unicodedata

# ----------------------------------------------------------------- the LLM --

_COMMON_ITN = """\
- Convert ONLY what the speaker said as a number, EXACTLY as said: never \
expand, round, complete or infer (a year, century, AM/PM, unit or currency \
that was not spoken is never added).
- Digit sequences said one by one (phone numbers, IDs, codes) become \
contiguous digits with no invented separators.
- Ranges: convert each number and keep the connecting word as spoken \
("5 to 10", "من 5 إلى 10"); never invent a dash.
- NEVER convert numbers inside names, titles, proverbs, idioms, \
religious texts (Quran, hadith, scripture) or quoted poetry.
- Negative numbers: "minus five" -> "-5" for temperatures and arithmetic.
- Arithmetic: numbers become digits, operator words stay as spoken \
("2 plus 2 equals 4"); no symbols the speaker did not say.
- Scores: digits with a hyphen only for a sports/game score ("2-1"); \
versions and models: "version two point oh" -> "version 2.0", "iPhone 15".
- Web: spoken addresses in written form when they are unmistakably an \
email/URL/handle ("info at example dot com" -> "info@example.com", \
"example dot com slash help" -> "example.com/help").
- Latin letters spelled out one by one form an acronym in capitals without \
spaces or periods ("U S A" -> "USA"). An acronym written in the language's \
own script stays in that script exactly as the ORIGINAL writes it \
("ഇ. എസ്. ഐ. സി", "यूपीआई", "بي بي سي"): never convert it to Latin.
- When ORIGINAL and ASR write the same code-switched word in different \
scripts ("Playstation" / "بلاي ستيشن"), use the ORIGINAL's.
"""

ITN = {
    "ar": """\
INVERSE TEXT NORMALIZATION (Arabic), Western digits 0-9 only (never ٠-٩):
- Quantities: "ثلاثة آلاف" -> "3000", "مية وخمسين" -> "150", "تلاتين" -> "30". \
The counted noun stays EXACTLY as spoken, even when colloquial agreement \
differs from MSA ("خمس مرة" -> "5 مرة").
- Millions and above: digits + the scale word ("ثلاثة ملايين" -> "3 ملايين", \
"مليون ونص" -> "1.5 مليون"); a bare "مليون"/"مليار" (one million) stays a word.
- Decimals said with فاصلة: "ثلاثة فاصلة خمسة" -> "3.5". Fraction words stay \
words ("نص", "ربع", "تلت", "ساعة ونص").
- Percent: digits + "٪" ("خمسين بالمية" -> "50٪").
- Money: digits + the currency word as spoken ("خمسين ريال" -> "50 ريال", \
"ألف درهم" -> "1000 درهم"); never a symbol ($, €, ر.س).
- Years: "سنة ألفين وعشرين" -> "سنة 2020", "عام ألف وتسعمية وتسعين" -> "عام 1990".
- Times: H:MM when minutes are spoken ("الساعة تلاتة ونص" -> "الساعة 3:30", \
"الساعة اتنين وربع" -> "الساعة 2:15", "الساعة خمسة إلا ربع" -> "الساعة 4:45"); \
bare digits otherwise ("الساعة تلاتة" -> "الساعة 3"). Keep the spoken period \
word (صباحًا، مساءً، العصر); never a 24-hour clock that was not said.
- Dates: digits for day and year, the month exactly as spoken ("الحادي \
والعشرين من مارس ألفين وأربعة وعشرين" -> "21 مارس 2024", Levantine "21 \
آذار", Hijri "15 رمضان"); a month SPOKEN as a number keeps the spoken \
order with "/" ("واحد وعشرين عشرة" -> "21/10"); never convert Hijri <-> \
Gregorian, never add هـ / م.
- The counted noun is never "corrected": "100 دراهم" stays "100 دراهم" even \
though MSA would say "100 درهم".
- KEEP AS WORDS: ordinals (الأول، التاني، العاشر، القرن العشرين); "واحد/وحدة" \
used as an article or pronoun ("واحد صاحبي", "كل واحد"); "واحد" and "اتنين" \
after a noun as emphasis ("كتاب واحد"); idioms (ألف عافية، ألف مبروك، ألف \
شكر، مية بالمية، مرة وحدة، سبعة أيام بلياليها); vague quantities (عشرات، مئات، \
كم واحد، شوية).
""" + _COMMON_ITN,
    "en": """\
INVERSE TEXT NORMALIZATION (English):
- Quantities: one to nine stay WORDS ("five years", "three times"), 10 and \
up become digits ("twenty five" -> "25", "three thousand" -> "3000", "ten \
thousand" -> "10,000"; comma grouping from 10,000 up, none for 4 digits). \
(The sources write small numbers as words ~30:1; test5 showed the model \
cannot hold a 1-9 -> digits rule consistently.)
- Always digits, even below 10: money, percentages, times, dates, years, \
ages and measurements with a unit ("5 kilometers", "$3", "4%", "3 PM", \
"May 5", "a 7-year-old"), decimals, scores, versions, and numbers that \
name something ("Chapter 3", "Formula 1", "PlayStation 2").
- Millions and above: "two million" -> "2 million", "three point five \
billion" -> "3.5 billion". Decimals: "three point five" -> "3.5", "point \
five" -> "0.5". Simple fractions stay words ("a half", "three quarters").
- Percent: "ten percent" -> "10%". Money: symbols only for dollars, pounds \
and euros ("fifty dollars" -> "$50", "five dollars fifty" -> "$5.50", \
"thirty thousand pounds" -> "£30,000", "ten euros" -> "€10", "two million \
dollars" -> "$2 million"); other currencies keep their word ("50 rupees", \
"20 dirhams", "100 yen"); "fifty cents" -> "50 cents"; "a buck" stays.
- Units stay words: "five kilometers" -> "5 kilometers", "twenty degrees" -> \
"20 degrees" (no km, °).
- Times: "three thirty pm" -> "3:30 PM" (always "AM"/"PM", capitals, no \
periods), "three o'clock" -> "3 o'clock"; time zones stay as spoken \
("Eastern Time", "GMT"); \
"noon", "midnight", "half past three", "quarter to five" stay words.
- Dates keep the spoken order and the ordinal only if spoken as one: \
"March twenty first" -> "March 21st", "March twenty one" -> "March 21", \
"the second of October" -> "the 2nd of October", "thirty one July" -> \
"31 July"; full date with year: "March 21st, 2024".
- Years: "March twenty first twenty twenty four" -> "March 21st, 2024", \
"nineteen ninety" -> "1990", "two thousand and five" -> "2005", "the \
nineties" -> "the '90s". "oh" read as zero becomes 0 ("room two oh one" -> \
"room 201").
- Ordinals: 1st-10th stay words ("the third time", "first of all"); 11th and \
up become digits ("the 21st century", "her 40th birthday"); in dates per above.
- KEEP AS WORDS: pronoun/idiomatic "one" ("one of them", "no one", \
"someone", "the one", "one day", "at one point", "one another"); "a couple", \
"a dozen", "a few", "hundreds of", "thousands of"; Roman numerals in names \
as conventionally written ("World War II", "Henry VIII").
""" + _COMMON_ITN,
    "zh": """\
INVERSE TEXT NORMALIZATION (Chinese):
- Become digits: years and dates ("二零二一年三月二十一日" -> "2021年3月21日"), \
clock times ("三点半" -> "3:30", "八点十五分" -> "8:15"), percentages \
("百分之五十" -> "50%"), money ("五十块" -> "50块", "三百元" -> "300元"), \
measurements with units ("五公里" -> "5公里"), decimals ("三点五" -> "3.5"), \
large numbers with 万/亿 kept as units ("十万" -> "10万", "三亿" -> "3亿", \
"两万五" -> "2.5万"), counts of 10 and up ("二十个人" -> "20个人").
- STAY CHARACTERS: 1-9 with a measure word in narrative ("三个计划", "两个人", \
"一次"); approximations (几个, 三四个, 十几个, 上百); ordinals (第一, 第二次); \
fraction words (三分之一, 一半); weekdays (星期三); words that contain a \
number (一些, 一起, 一样, 一定, 一直, 十分, 万一, 一下); chengyu and idioms \
(一心一意, 三心二意, 一石二鸟).
""" + _COMMON_ITN,
    "hi": """\
INVERSE TEXT NORMALIZATION (Hindi), Western digits only (never १२३):
- Quantities: "पचास" -> "50", "तीन हज़ार" -> "3000"; लाख/करोड़ stay as the \
scale word after the digits ("दो लाख" -> "2 लाख", "पाँच करोड़" -> "5 करोड़").
- Percent: "पचास प्रतिशत/फ़ीसदी" -> "50%". Money: digits + the word as \
spoken ("पचास रुपये" -> "50 रुपये"); never ₹.
- Times: "साढ़े तीन बजे" -> "3:30 बजे", "सवा चार बजे" -> "4:15 बजे", \
"पौने पाँच बजे" -> "4:45 बजे", "डेढ़ बजे" -> "1:30 बजे", "ढाई बजे" -> "2:30 बजे".
- Dates and years: "इक्कीस मार्च" -> "21 मार्च", "दो हज़ार चौबीस" -> "2024".
- STAY WORDS: "एक" as the article "a" ("एक आदमी आया"); डेढ़, ढाई, सवा, साढ़े, \
पौने with quantities ("डेढ़ घंटा", "ढाई साल"); ordinals (पहला, दूसरा, \
तीसरी बार); idioms ("एक-दो बार", "दो-चार दिन", "सौ बात की एक बात").
""" + _COMMON_ITN,
    "ml": """\
INVERSE TEXT NORMALIZATION (Malayalam), Western digits only:
- Quantities: "അമ്പത്" -> "50", "ആയിരം" -> "1000", "മൂവായിരം" -> "3000"; \
ലക്ഷം/കോടി stay as the scale word after the digits ("രണ്ട് ലക്ഷം" -> \
"2 ലക്ഷം", "അഞ്ച് കോടി" -> "5 കോടി").
- Percent: "അമ്പത് ശതമാനം" -> "50%". Money: digits + the word as spoken \
("അമ്പത് രൂപ" -> "50 രൂപ"); never ₹.
- Times: "മൂന്നര മണി" -> "3:30", "മൂന്ന് മണി" -> "3 മണി". Dates: "2024 മാർച്ച് 21" \
(fields as spoken).
- STAY WORDS: "ഒരു" as the article "a" ("ഒരു ദിവസം"); "ഒന്ന്" in idioms \
("ഒന്ന് നോക്കൂ"); ordinals (ഒന്നാമത്തെ, രണ്ടാം); fraction words (അര, കാൽ).
""" + _COMMON_ITN,
}

FORMAT = {
    "ar": """\
ORTHOGRAPHY (Arabic), for MSA and dialect alike -- spell the SAME word right, \
never replace it:
- Hamza on its standard seat (أنا، إلى، إن، سأل، مسؤول، شيء), madda where due (آخر).
- Ta marbuta for the feminine ending (مدرسة، كبيرة), ـه for the pronoun "his/him" \
(لصالحه، يعطيه، تبغاه); alef maqsura (على، مستشفى) vs ya (في، علي) by the word.
- No tatweel (ـ). Dialect-specific spellings are words of the dialect: keep them \
("ليش", "هلق", "دلوقتي", "بيعمل"); do not respell them as MSA.
- Code-switched words stay in the script the transcripts use: "campus" stays \
Latin; established loanwords keep their Arabic spelling (كمبيوتر، موبايل). \
Embedded English: proper nouns and acronyms capitalized, other words lower \
case unless they start an English sentence.
""",
    "hi": """\
ORTHOGRAPHY (Hindi): standard Devanagari spelling of the SAME word: matras, \
halant, anusvara/chandrabindu as in standard usage; keep nukta (ज़, फ़, क़) \
exactly as the transcripts have it -- never add or strip it. English words, \
acronyms and loanwords stay in the script the ORIGINAL uses: do NOT convert \
Devanagari ("यूपीआई", "टेस्ट") to Latin or Latin to Devanagari. Latin-script English: proper
nouns and acronyms capitalized, other words as the ORIGINAL writes them.
""",
    "ml": """\
ORTHOGRAPHY (Malayalam): standard spelling of the SAME word with every vowel \
sign, virama and chillu present; never change a word's inflection or tense. \
English words and loanwords stay in the script the ORIGINAL uses: do NOT \
convert Malayalam-script loanwords ("ടെസ്റ്റ്", "ഫീഡ്ബാക്ക്") to Latin or back. Latin-script English: proper
nouns and acronyms capitalized, other words as the ORIGINAL writes them.
""",
    "zh": """\
ORTHOGRAPHY (Chinese): Simplified characters; no spaces between Chinese \
characters, and none between Chinese and an embedded Latin word or number \
("很intense", "college可以"); spaces only between consecutive Latin words. Keep erhua (儿) and regional words as spoken. \
Embedded English: proper nouns and acronyms capitalized (TV, NBA, iPhone), \
other words as the ORIGINAL writes them. Money keeps its Chinese unit \
(元, 块, 美元, 欧元) -- never $ or ¥.
""",
    "en": """\
CAPITALIZATION AND ORTHOGRAPHY (English): sentence case -- capital at the \
start of every sentence; capitalize "I", names of people and places, \
organizations, brands as their owners write them ("iPhone", "YouTube"), \
days, months, nationalities, languages and religions; acronyms in capitals \
(NASA, AI, UPI); titles with a period (Mr., Mrs., Ms., Dr., St.). Never \
ALL CAPS for emphasis. Contractions as spoken ("don't", "gonna" stay); \
fillers spelled "uh", "um", "hmm"; American spelling unless the ORIGINAL \
consistently uses British.
""",
}

PUNCTUATION = {
    "ar": "PUNCTUATION (Arabic): ، ؛ ؟ ! . : «» -- never the Latin , ; ?. "
          "End complete sentences; a cut-off utterance ends with \"...\". One line: never a line break.",
    "hi": "PUNCTUATION (Hindi): । ends a sentence (never |), ? for questions, standard commas. "
          "A cut-off utterance ends with \"...\". One line: never a line break.",
    "ml": "PUNCTUATION (Malayalam): . , ? ! as in standard Malayalam. "
          "A cut-off utterance ends with \"...\". One line: never a line break.",
    "zh": "PUNCTUATION (Chinese): full-width 。，？！；：、“” and 《》 for titles of works only; a cut-off utterance "
          "ends with ……. One line: never a line break.",
    "en": "PUNCTUATION (English): standard . , ? ! ; : and \"double quotes\". A cut-off "
          "utterance ends with \"...\". One line: never a line break.",
}


def conventions(lang: str) -> str:
    return "\n".join(x for x in (FORMAT.get(lang, ""), PUNCTUATION.get(lang, ""), ITN.get(lang, _COMMON_ITN)) if x)


# -------------------------------------------------------------- the canon --

# Non-speech annotations that some sources put in the label. A record whose
# label is nothing but these is a non-speech clip, not a transcript.
_NON_SPEECH = re.compile(
    r"\[\s*(?:music|noise|laughter|laugh|laughs|applause|silence|inaudible|unintelligible|"
    r"crosstalk|cough|breath|sigh|unk|foreign|"
    r"موسيقى|موسيقي|ضوضاء|ضحك|ضحكة|تصفيق|صمت|غير مفهوم|كلام غير مفهوم|"
    r"音乐|笑声|噪音|掌声|静音|"
    r"संगीत|शोर|हँसी|हंसी|तालियाँ|"
    r"സംഗീതം|ശബ്ദം|ചിരി)\s*\]"
    r"|<\s*(?:unk|noise|music|laughter|sil)\s*>"
    r"|\(\s*(?:music|laughter|laughs|applause|inaudible|noise)\s*\)",
    re.IGNORECASE,
)

_AR_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ML_DIGITS = str.maketrans("൦൧൨൩൪൫൬൭൮൯", "0123456789")
# Legacy chillu (consonant + virama + ZWJ) -> atomic chillu (Unicode 5.1+).
_ML_CHILLU = {"ണ്‍": "ൺ", "ന്‍": "ൻ", "ര്‍": "ർ",
              "ല്‍": "ൽ", "ള്‍": "ൾ", "ക്‍": "ൿ"}


# Leftovers of an earlier LLM pass at the start of some labels (test3: 18
# English originals began "Corrected Transcript: ...").
_META_PREFIX = re.compile(r"^\s*(?:corrected|cleaned|final)?\s*transcript(?:ion)?\s*:\s*", re.IGNORECASE)


_AR_ALEF_TANWEEN = re.compile("([\u0621-\u064a])([\u064b-\u0652\u0670]*)\u0627\u064b")


def _alef_tanween(m: re.Match) -> str:
    letter, marks = m.group(1), m.group(2)
    if "\u064b" in marks:  # already before the alef: drop the duplicate
        return letter + marks + "\u0627"
    return letter + marks + "\u064b\u0627"


def strip_non_speech(text: str) -> str:
    return " ".join(_NON_SPEECH.sub(" ", _META_PREFIX.sub("", text)).split())


_EN_LOWER_START = {"iphone", "ipad", "ipod", "imac", "ios", "ebay", "e-mail", "etc."}


_MERIDIEM = re.compile(r"(\d)\s*([AaPp])(\.[Mm]\.|\.?\s?[Mm]\b)")


def _meridiem(text: str) -> str:
    """One form after a time: "8 p.m." / "10 am" / "3P.M." -> "8 PM" / "10 AM" / "3 PM".

    A dotted "p.m." at the end of a sentence also carried the full stop, so
    the stop is kept when a new sentence (capital letter) or the text ends.
    """
    def fix(m: re.Match) -> str:
        out = f"{m.group(1)} {m.group(2).upper()}M"
        rest = text[m.end():]
        if m.group(3).endswith(".") and (not rest.strip() or re.match(r"\s+[A-Z]", rest)):
            out += "."
        return out

    return _MERIDIEM.sub(fix, text)


_ORD_WORDS = {"1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth", "5th": "fifth",
              "6th": "sixth", "7th": "seventh", "8th": "eighth", "9th": "ninth", "10th": "tenth"}
_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
           "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
_SMALL_ORD = re.compile(rf"(?<!\d)\b(10th|[1-9](?:st|nd|rd|th))\b(?!\s+(?:of\s+)?(?:{_MONTHS})\b)")


def _small_ordinals(text: str) -> str:
    """1st-10th as words except in dates (test5: "seventh place" -> "7th place")."""
    def word(m: re.Match) -> str:
        before = text[max(0, m.start() - 12):m.start()]
        if re.search(rf"\b(?:{_MONTHS})\.?\s*$", before):  # "January 9th"
            return m.group(1)
        return _ORD_WORDS[m.group(1)]

    return _SMALL_ORD.sub(word, text)


def _english_case(text: str) -> str:
    """Sentence-initial capitals and the pronoun "I" (test3: 759 English records
    started lower case -- many sources are all lower case)."""
    text = re.sub(r"(?<![\w'])i(?=(?:'m|'ve|'ll|'d)?(?![\w']))", "I", text)

    def cap(m: re.Match) -> str:
        word = m.group(2)
        return m.group(1) + (word if word.lower() in _EN_LOWER_START else word[0].upper() + word[1:])

    return re.sub(r"(^|[.?!]\s+|[.?!][\"”]\s+)([a-z][\w.'-]*)", cap, text)


def canonicalize(text: str, lang: str) -> str:
    """The deterministic final form: identical conventions in every record."""
    text = unicodedata.normalize("NFC", strip_non_speech(text))
    if lang == "ar":
        text = text.translate(_AR_INDIC_DIGITS).replace("ـ", "")
        text = re.sub(r"(?<=\d)\s*%", "٪", text)
        # fathatan before the final alef ("شكرًا"), not on it ("شكراً"); when both
        # forms met (test5: 487 "أيضًاً", "جدًّاً") the one on the alef goes
        text = _AR_ALEF_TANWEEN.sub(_alef_tanween, text)
        # one tanween per letter ("خارجيّةًٍ" -> "خارجيّةً")
        text = re.sub("([\u064b\u064c\u064d])[\u064b\u064c\u064d]+", "\\1", text)
        text = unicodedata.normalize("NFC", text)
    elif lang == "en":
        text = _meridiem(text)
        text = _english_case(_small_ordinals(text))
    elif lang == "zh":
        # no space at a Chinese/Latin-or-digit boundary (test5: the model added
        # and removed them at random); spaces between Latin words stay
        text = re.sub(r"(?<=[\u3400-\u9fff\u3000-\u303f\uff00-\uffef])\s+(?=[A-Za-z0-9])", "", text)
        text = re.sub(r"(?<=[A-Za-z0-9.,!?%])\s+(?=[\u3400-\u9fff\u3000-\u303f\uff00-\uffef])", "", text)
        text = re.sub(r"(?<=[\u3400-\u9fff\u3000-\u303f\uff00-\uffef])\s+(?=[\u3400-\u9fff\u3000-\u303f\uff00-\uffef])",
                      "", text)
    elif lang == "hi":
        # danda attaches to the word before it ("है।"); "|" is a keyboard stand-in
        text = re.sub(r"\s*[|।](?!।)", "।", text.translate(_DEVANAGARI_DIGITS))
    elif lang == "ml":
        text = text.translate(_ML_DIGITS)
        for old, new in _ML_CHILLU.items():
            text = text.replace(old, new)
        # "nta": the Unicode 5.1+ recommended encoding is chillu-n + virama + rra
        # (L2/07-279, L2/19-345r2); na + virama + rra is the legacy form. Most
        # originals already use it; the LLM was converting them away. (The ZWJ
        # variant was already read as chillu-n above, its legacy meaning "ൻറ".)
        text = text.replace("\u0d28\u0d4d\u0d31", "\u0d7b\u0d4d\u0d31")
    return " ".join(text.split())
