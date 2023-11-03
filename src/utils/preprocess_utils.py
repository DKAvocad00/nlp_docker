import re
from gensim.parsing import strip_tags, strip_numeric, \
    strip_multiple_whitespaces, strip_punctuation, \
    remove_stopwords, preprocess_string

CLEAN_FILTERS = [strip_tags,
                 strip_numeric,
                 strip_punctuation,
                 strip_multiple_whitespaces,
                 remove_stopwords]


def cleaning_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]+", r' ', text)
    text = re.sub(r'\s+\w{1}\s+', r' ', text)
    processed_text = preprocess_string(text, CLEAN_FILTERS)

    return processed_text
