from Levenshtein import ratio

from xumi.text import Text
from xumi.perturbations.perturbations import RemoveRandomWord, PerturbRandomWord, Identity, UpperCase, LowerCase


def test_identity():
    text = Text(original="This is a sample sentence")
    before = text.transformed

    assert Identity.is_applicable(text=text)
    Identity.perturb(text=text)

    assert ratio(before, text.transformed) == 1


def test_remove_random_word():
    text = Text(original="This is a sample sentence")
    before = text.transformed

    assert RemoveRandomWord.is_applicable(text=text)
    RemoveRandomWord.perturb(text=text)

    assert 0.7 < ratio(before, text.transformed) < 1
    assert len(before) > len(text.transformed)


def test_remove_random_word_empty_sentence():
    text = Text(original="")
    before = text.transformed

    assert RemoveRandomWord.is_applicable(text=text)
    RemoveRandomWord.perturb(text=text)

    assert before == text.transformed


def test_perturb_random_word():
    text = Text(original="This is a sample sentence")
    before = text.transformed

    assert PerturbRandomWord.is_applicable(text=text)
    PerturbRandomWord.perturb(text=text)

    assert 0.7 < ratio(before, text.transformed) < 1


def test_uppercase():
    text = Text(original="This is a sample sentence")

    assert UpperCase.is_applicable(text=text)
    UpperCase.perturb(text=text)

    assert text.transformed.lower() == text.original.lower()


def test_lowercase():
    text = Text(original="This is a sample sentence")

    assert LowerCase.is_applicable(text=text)
    LowerCase.perturb(text=text)

    assert text.transformed.lower() == text.original.lower()
