"""
Shared text preprocessing helpers.
"""
import re

NEGATION_TERMS = {
    "not",
    "no",
    "never",
    "cannot",
    "can't",
    "dont",
    "don't",
    "doesnt",
    "doesn't",
    "isnt",
    "isn't",
    "wasnt",
    "wasn't",
    "werent",
    "weren't",
    "wont",
    "won't",
    "wouldnt",
    "wouldn't",
    "shouldnt",
    "shouldn't",
    "couldnt",
    "couldn't",
    "didnt",
    "didn't",
    "havent",
    "haven't",
    "hasnt",
    "hasn't",
    "hadnt",
    "hadn't",
    "n't",
}

NEGATION_PUNCT = {".", "!", "?", ";", ":"}


def negation_preprocess(text: str) -> str:
    """
    Append _NEG to tokens after negation words until punctuation.
    Example: "not good" -> "not good_NEG".
    """
    tokens = re.findall(r"\w+|[^\w\s]", str(text).lower())
    output = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMS:
            negate = True
            output.append(token)
            continue
        if token in NEGATION_PUNCT:
            negate = False
            output.append(token)
            continue
        if negate and token.isalnum():
            output.append(f"{token}_NEG")
        else:
            output.append(token)
    return " ".join(output)
