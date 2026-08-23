"""
Shared text helpers used by both search and extraction.

The stopword list here is deliberately aggressive about two categories
that a general-purpose list would keep:

  question words  — "where", "what", "how". Queries are usually phrased
                    as questions, but the answer never contains the
                    question word, so keeping them only adds noise.

  self-reference  — "user", "i", "my". Every memory is a statement about
                    the user, so these tokens match *everything*. In BM25
                    that turns the keyword ranker into a coin flip: every
                    document matches, so nothing is discriminated.

Both of those actively degraded retrieval before they were removed.
"""

import re

WORD = re.compile(r"[A-Za-z0-9]+")

STOPWORDS = frozenset({
    # articles, conjunctions, prepositions
    "the", "a", "an", "and", "or", "but", "if", "of", "in", "on", "at", "to",
    "for", "with", "as", "by", "from", "into", "onto", "up", "down", "out",
    "over", "under", "again", "further", "about", "against", "between",
    # pronouns and self-reference (see note above)
    "i", "me", "my", "mine", "myself", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "user", "users",
    # auxiliaries and copulas
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "doing", "have", "has", "had", "having", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must",
    # question words (see note above)
    "what", "whats", "when", "where", "who", "whom", "whose", "which", "why",
    "how", "tell", "know",
    # determiners, degree, filler
    "this", "that", "these", "those", "there", "here", "then", "than", "so",
    "such", "no", "not", "only", "own", "same", "too", "very", "just", "now",
    "also", "any", "some", "all", "both", "each", "more", "most", "other",
})


def tokens(text: str) -> list[str]:
    """All alphanumeric tokens, lowercased."""
    return WORD.findall(text.lower())


def significant_tokens(text: str, min_length: int = 2) -> list[str]:
    """
    Content-bearing tokens: stopwords dropped, order and repeats kept.

    Order is preserved because callers that build query strings want the
    user's phrasing; callers that want a set can wrap this in one.
    """
    return [
        t for t in tokens(text)
        if t not in STOPWORDS and len(t) >= min_length
    ]


def significant_set(text: str, min_length: int = 3) -> set[str]:
    """Content tokens as a set, for overlap comparisons."""
    return set(significant_tokens(text, min_length=min_length))


def normalise_name(name: str) -> str:
    """
    Matching key for a named thing: case- and punctuation-insensitive.

    Stopwords are intentionally *not* removed — "The Beatles" and "Bank
    of America" are names, and stripping their function words would merge
    genuinely distinct entities.
    """
    return " ".join(WORD.findall(name.lower()))
