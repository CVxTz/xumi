from Levenshtein import ratio

from xumi.perturbations.utilities import (
    remove_character,
    add_character,
    substitute_character,
)


def test_remove_character():
    word = "word"
    transformed = remove_character(word=word)

    assert len(transformed) == len(word) - 1
    assert ratio(transformed, word) == 0.8571428571428571


def test_remove_character_single_char():
    word = "w"
    transformed = remove_character(word=word)

    assert len(transformed) == len(word) - 1
    assert transformed == ""


def test_remove_character_empty():
    word = ""
    transformed = remove_character(word=word)

    assert transformed == ""


def test_add_character():
    word = "word"
    transformed = add_character(word=word)

    assert len(transformed) == len(word) + 1
    assert ratio(transformed, word) == 0.8888888888888888


def test_add_character_single_char():
    word = "w"
    transformed = add_character(word=word)

    assert len(transformed) == len(word) + 1


def test_add_character_empty():
    word = ""
    transformed = add_character(word=word)

    assert len(transformed) == 1


def test_substitute_character():
    word = "word"
    transformed = substitute_character(word=word)

    assert len(transformed) == len(word)
    assert ratio(transformed, word) == 0.75


def test_substitute_character_empty():
    word = ""
    transformed = substitute_character(word=word)

    assert len(transformed) == 0
