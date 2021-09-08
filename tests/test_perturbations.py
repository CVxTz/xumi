from Levenshtein import ratio

from xumi.perturbations.perturbations import (
    RemoveRandomWord,
    PerturbRandomWord,
    Identity,
    UpperCase,
    LowerCase,
    AddSpellingError,
    ChangeVerbForm,
    CommonErrors,
    PERTURBATIONS,
    WEIGHTS,
)
from xumi.text import Text


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

    assert 0.5 < ratio(before, text.transformed) < 1
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

    assert 0.5 < ratio(before, text.transformed) < 1


def test_uppercase():
    text = Text(original="This is a sample sentence")

    assert UpperCase.is_applicable(text=text)
    UpperCase.perturb(text=text)

    assert text.transformed.lower() == text.original.lower()
    assert text.transformed.isupper()


def test_lowercase():
    text = Text(original="This is a sample sentence")

    assert LowerCase.is_applicable(text=text)
    LowerCase.perturb(text=text)

    assert text.transformed.lower() == text.original.lower()
    assert text.transformed.islower()


def test_spelling():
    text = Text(original="This is a sample sentence")
    before = text.transformed

    assert AddSpellingError.is_applicable(text=text)
    AddSpellingError.perturb(text=text)

    assert 0.5 < ratio(before, text.transformed) < 1


def test_spelling_not_applicable():
    text = Text(original="aaaa")

    assert not AddSpellingError.is_applicable(text=text)


def test_change_verb_form():
    text = Text(original="This is a sample sentence")
    before = text.transformed

    assert ChangeVerbForm.is_applicable(text=text)
    ChangeVerbForm.perturb(text=text)

    assert 0.5 < ratio(before, text.transformed) < 1


def test_change_verb_form_not_applicable():
    text = Text(original="aaaa")

    assert not ChangeVerbForm.is_applicable(text=text)


def test_perturbations():
    assert len(PERTURBATIONS) == len(WEIGHTS)
    assert len(PERTURBATIONS) > 0
    assert abs(sum(WEIGHTS) - 1) < 1e-9
    assert len(PERTURBATIONS) == 8


def test_common_errors():
    text = Text(original="This is a sample sentence")
    before = text.transformed

    assert CommonErrors.is_applicable(text=text)
    CommonErrors.perturb(text=text)

    assert 0.5 < ratio(before, text.transformed) < 1

