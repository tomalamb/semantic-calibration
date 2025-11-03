import re
import string
from dateparser.search import search_dates
from nltk.tokenize import sent_tokenize
from number_parser import parse

# Precompiled regex patterns for performance
dash_pattern = re.compile(r"[–—]")
MONTH_CUE_RE = re.compile(
    r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|'
    r'may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|'
    r'oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b',
    flags=re.IGNORECASE
)
NUMERIC_DATE_RE = re.compile(
    r'\b\d{1,2}/\d{1,2}/\d{2,4}[.,]?\b'
)
MONTH_YEAR_RE = re.compile(
    r'^\s*(?P<month>(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
    r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|'
    r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?))\s*,?\s*'
    r'(?P<year>\d{4})\s*[.,]?\s*$',
    flags=re.IGNORECASE
)
NUM_MONTH_YEAR_RE = re.compile(
    r'^\s*(?P<month>\d{1,2})/(?P<year>\d{4})\s*[.,]?\s*$',
)
STANDALONE_MONTH_RE = re.compile(
    r'^\s*(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
    r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|'
    r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[.,]?\s*$',
    flags=re.IGNORECASE
)

# Precompile hyphen and stray dot patterns
HYPHEN_RE = re.compile(r'(?<=[A-Za-z])-(?=[A-Za-z])')
DOT_RE = re.compile(r'(?<!\d)\.(?!\d)')

# Build translate table to remove punctuation except dot and hyphen
_PUNCT_TO_REMOVE = ''.join(c for c in string.punctuation if c not in '.-')
PUNCT_TRANSLATE = str.maketrans('', '', _PUNCT_TO_REMOVE)

# Module-level month mapping
MONTH_MAP = {
    'jan': '01', 'january': '01',
    'feb': '02', 'february': '02',
    'mar': '03', 'march': '03',
    'apr': '04', 'april': '04',
    'may': '05',
    'jun': '06', 'june': '06',
    'jul': '07', 'july': '07',
    'aug': '08', 'august': '08',
    'sep': '09', 'sept': '09', 'september': '09',
    'oct': '10', 'october': '10',
    'nov': '11', 'november': '11',
    'dec': '12', 'december': '12'
}


def exclude_instruction(prompt: str, dset:str) -> str:
    if dset == "squad":
        parts = prompt.split("Context:", maxsplit=1)
        return "Context: " + parts[1].strip() if len(parts) == 2 else prompt.strip()
    else:
        parts = prompt.rsplit("Question:", maxsplit=1)
        return "Question: " + parts[1].strip() if len(parts) == 2 else prompt.strip()
        
def extract_question_only(prompt: str) -> str:
    parts = prompt.split("Question:", maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else prompt.strip()


def clean_response(resp, dset):
    if dset == "gsm8k" and "###" in resp:
        return resp.split("###")[-1]
    return resp


def check_numeric_date_format(text: str) -> bool:
    return bool(NUMERIC_DATE_RE.search(text))


def contains_date_cue(text: str) -> bool:
    return bool(MONTH_CUE_RE.search(text)) or check_numeric_date_format(text)


def normalize_month_year(text: str) -> str | None:
    m = MONTH_YEAR_RE.fullmatch(text)
    if not m:
        return None
    month_key = m.group('month').lower()
    return f"{m.group('year')}-{MONTH_MAP[month_key]}"


def normalize_numeric_month_year(text: str) -> str | None:
    m = NUM_MONTH_YEAR_RE.fullmatch(text)
    if not m:
        return None
    return f"{m.group('year')}-{m.group('month').zfill(2)}"


def normalize_dates_in_text(text: str, date_format: str = "%Y-%m-%d") -> str:
    if not text or text.isspace():
        return text.lower()
    stripped = text.strip()
    lower = stripped.lower()

    # 1) MonthName + Year
    my_norm = normalize_month_year(stripped)
    if my_norm:
        return my_norm

    # 2) Numeric MM/YYYY
    num_norm = normalize_numeric_month_year(stripped)
    if num_norm:
        return num_norm

    # 3) Standalone month
    if STANDALONE_MONTH_RE.fullmatch(stripped):
        return lower

    # 4) Complex dates via search_dates
    if contains_date_cue(stripped):
        results = search_dates(stripped, settings={"PREFER_DAY_OF_MONTH": "first"})
        if results:
            normalized = stripped
            for substr, dt_obj in results:
                iso = normalize_month_year(substr) \
                      or normalize_numeric_month_year(substr) \
                      or dt_obj.date().strftime(date_format)
                normalized = re.sub(re.escape(substr), iso, normalized, flags=re.IGNORECASE)
            return normalized.lower()

    # 5) No date cues
    return lower


def clean_and_preprocess_text(text: str) -> str:
    original = text
    if not isinstance(text, str) or not text.strip():
        return original.lower() if isinstance(text, str) else text

    s = text.strip()

    # Date normalization
    if contains_date_cue(s):
        s = normalize_dates_in_text(s)

    # First sentence via NLTK
    if '\n' in s:
        s = s.split('\n', 1)[0]
    sentences = sent_tokenize(s)
    s = sentences[0] if sentences else s

    if not s.strip():
        return original.lower()

    # Punctuation & symbol cleanup
    s = s.replace('_', ' ')
    s = dash_pattern.sub('-', s)
    s = s.translate(PUNCT_TRANSLATE)
    s = HYPHEN_RE.sub(' ', s)
    s = DOT_RE.sub('', s)

    if not s.strip():
        return original.lower()

    # Number-word conversion
    try:
        s = parse(s)
    except:
        pass

    if not s.strip():
        return original.lower()

    # Collapse whitespace & lowercase once
    return ' '.join(s.split()).lower()


def clean_generated_text(text: str) -> str:
    if not text or text.isspace():
        return text
    if '\n' in text:
        text = text.split('\n', 1)[0].strip()
    sentences = sent_tokenize(text)
    return sentences[0].strip() if sentences else text