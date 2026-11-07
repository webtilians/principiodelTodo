"""Two memories about the same thing that say something different: a date moved, a price changed, a yes became a no.

INFINITO does not choose between them. When something is saved, remember names the earlier memories it may
disagree with, so the agent can tell the user both, with their dates and sources, instead of picking one; the
briefing lists the recent ones. A work record is never rewritten: a new memory corrects the old one, and the user
says which is right.

It is worked out from the words alone, with no model, so every app sees the same and nothing leaves the server.
Two memories may disagree when they share most of what they are about (the shorter one's words, roughly) and then
differ in one of three ways:
- each has a number, date, weekday, month or version the other lacks ("el 15" / "el 20", "3.000 €" / "2.500 €");
- one says no and the other does not ("Acme quiere vídeo" / "Acme no quiere vídeo");
- one approves, accepts or confirms and the other rejects, cancels or discards.
A detail added to an older memory ("en noviembre" / "el 20 de noviembre") is not a disagreement: the older one
has no value the newer one lacks. The agent reads both and decides; these are only the ones worth a look.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional

# Words that carry no subject, in Spanish and English, written without accents.
_STOPWORDS = frozenset("""
a al algo alguien algun alguna algunas alguno algunos ante antes aqui asi aun cada como con contra cual cuando de del
desde donde dos durante e el ella ellas ellos en entre era eran es esa esas ese eso esos esta estaba estado estan estar
estas este esto estos fue fueron ha habia han has hasta hay he hemos la las le les lo los mas me mi mis mucho muy nos
nuestra nuestro o otra otras otro otros para pero poco por porque pues que se sea segun ser si sido sobre son su sus
tambien tan tanto te tengo tenemos tiene tienen todo toda todas todos tu tus un una unas uno unos va van vez y ya yo
about after all also an and any are as at be been before being but by can could did do does for from had has have he
her his i if in into is it its me my of on or our she so than that the their them then there these they this those to
too up us was we were what when which who will with would you your
""".split())
_NEGATIONS = frozenset("no nunca jamas tampoco ningun ninguna ninguno sin not never nor cannot without".split())
_MONTHS = frozenset("""enero febrero marzo abril mayo junio julio agosto septiembre setiembre octubre noviembre diciembre
january february march april june july august september october november december""".split())
_WEEKDAYS = frozenset("""lunes martes miercoles jueves viernes sabado domingo
monday tuesday wednesday thursday friday saturday sunday""".split())
# Beginnings of words that say yes or no to something.
_YES = ("aprob", "aprueb", "acept", "confirm", "approv", "accept", "agreed")
_NO = ("rechaz", "cancel", "anul", "deneg", "descart", "reject", "declin", "refus")
# A word this long is cut to its first letters, so "entrega" and "entregas" are the same subject.
_STEM = 5
# Of the shorter memory's subject words, at least this share must be in the other one too, and of the longer
# one's, this: a new memory often adds why ("...pasa al 20 porque falta material").
_SHARED = 0.6
_SHARED_OF_LONGER = 0.3


@dataclass(frozen=True)
class Words:
    """What a memory says, as far as telling whether two disagree needs."""

    subject: FrozenSet[str]
    values: FrozenSet[str]
    negated: bool
    says: FrozenSet[str]  # "yes", "no", both or neither


def _plain(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.replace("n't", " not")
    # A number after # names something (PR #81), like a code: it says what the memory is about, not a value.
    folded = re.sub(r"#\s?(\d+)", r"n\1", folded)
    # 3.000, 2,5 and 15:30 are one value each.
    return re.sub(r"(?<=\d)[.,:'](?=\d)", "", folded)


def words_of(text: str, ignore: Iterable[str] = ()) -> Words:
    """ignore: words every compared memory shares anyway, such as the client's name.

    A number, a month or a weekday is a value. A word with letters and digits (F12, 11db6cf, n81 from #81) is a
    name for something, so two memories about PR #81 and PR #82 are about different things, not two values."""
    skipped = {_plain(word) for name in ignore for word in re.findall(r"\w+", name)}
    subject, values, says, negated = set(), set(), set(), False
    for word in re.findall(r"\w+", _plain(text)):
        if word in skipped:
            continue
        if word in _NEGATIONS:
            negated = True
        elif word.isdigit() or word in _MONTHS or word in _WEEKDAYS:
            values.add(word)
        elif re.fullmatch(r"v\d+", word):  # a version, v0.6.0
            values.add(word[1:])
        elif any(c.isdigit() for c in word):
            subject.add(word)
        elif word.startswith(_YES):
            says.add("yes")
        elif word.startswith(_NO):
            says.add("no")
        elif word not in _STOPWORDS and len(word) > 1:
            subject.add(word[:_STEM])
    return Words(frozenset(subject), frozenset(values), negated, frozenset(says))


def same_subject(older: Words, newer: Words) -> bool:
    """Most of the shorter one's subject is in the other, and it is not a few words lost in a long report."""
    shorter, longer = sorted((len(older.subject), len(newer.subject)))
    if not shorter:
        return False
    shared = len(older.subject & newer.subject)
    return shared >= min(2, shorter) and shared / shorter >= _SHARED and shared / longer >= _SHARED_OF_LONGER


def disagreement(older: Words, newer: Words) -> Optional[str]:
    """Why two memories about the same thing may disagree, in a few words for the agent, or None. Either order gives
    the same answer."""
    if not same_subject(older, newer):
        return None
    if older.values - newer.values and newer.values - older.values:
        return "a different date, number or version"
    if older.negated != newer.negated:
        return "one of them says no"
    if older.says and newer.says and older.says != newer.says:
        return "one approves or confirms, the other rejects or cancels"
    return None


def same_text(one: str, other: str) -> bool:
    """The same words, whatever the accents, case and punctuation: one memory, not two that disagree."""
    return re.findall(r"\w+", _plain(one)) == re.findall(r"\w+", _plain(other))
