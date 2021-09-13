import abc
import inspect
import json
import sys
from pathlib import Path
from random import choice
from typing import List

from nltk import word_tokenize

from xumi.perturbations.utilities import (
    substitute_character,
    remove_character,
    add_character,
    uppercase,
)
from xumi.text import Text

resources_base_path = Path(__file__).parents[2] / "resources"


class Perturbation(abc.ABC):
    weight = 0

    @classmethod
    def is_applicable(cls, text: Text):
        return True

    @classmethod
    @abc.abstractmethod
    def perturb(cls, text: Text):
        pass


class Identity(Perturbation):
    weight = 5

    def __init__(self):
        pass

    @classmethod
    def perturb(cls, text: Text) -> None:
        pass


class RemoveRandomWord(Perturbation):
    weight = 1

    def __init__(self):
        pass

    @classmethod
    def perturb(cls, text: Text) -> None:
        words = word_tokenize(text.transformed)
        if words:
            word = choice(words)
            text.transformed = text.transformed.replace(word, "")


class PerturbRandomWord(Perturbation):
    weight = 20

    def __init__(self):
        pass

    @classmethod
    def perturb(cls, text: Text) -> None:
        words = word_tokenize(text.transformed)
        if words:
            word = choice(words)
            func = choice(
                [substitute_character, remove_character, add_character, uppercase]
            )
            transformed = func(word)
            text.transformed = text.transformed.replace(word, transformed)


class LowerCase(Perturbation):
    weight = 3

    def __init__(self):
        pass

    @classmethod
    def perturb(cls, text: Text) -> None:
        text.transformed = text.transformed.lower()


class UpperCase(Perturbation):
    weight = 0.1

    def __init__(self):
        pass

    @classmethod
    def perturb(cls, text: Text) -> None:
        text.transformed = text.transformed.upper()


class AddSpellingError(Perturbation):
    weight = 5

    spell_errors_path = resources_base_path / "spell-errors.json"

    with open(spell_errors_path) as f:
        spell_error = json.load(f)

    def __init__(self):
        pass

    @classmethod
    def is_applicable(cls, text: Text):
        words = word_tokenize(text.transformed)
        return any(word.lower() in cls.spell_error for word in words)

    @classmethod
    def perturb(cls, text: Text) -> None:
        words = word_tokenize(text.transformed)
        available_words = [word for word in words if word.lower() in cls.spell_error]
        if available_words:
            word = choice(available_words)
            transformed = choice(cls.spell_error[word.lower()])
            text.transformed = text.transformed.replace(word, transformed)


class ChangeVerbForm(Perturbation):
    weight = 5

    verbs_all_path = resources_base_path / "verbs.json"

    with open(verbs_all_path) as f:
        verbs_all = json.load(f)

    verbs = {}

    for verb_forms in verbs_all:
        for current_form in verb_forms:
            verbs[current_form] = verb_forms.copy()
            verbs[current_form].remove(current_form)

    def __init__(self):
        pass

    @classmethod
    def is_applicable(cls, text: Text):
        words = word_tokenize(text.transformed)
        return any(word.lower() in cls.verbs for word in words)

    @classmethod
    def perturb(cls, text: Text) -> None:

        words = word_tokenize(text.transformed)
        available_words = [word for word in words if word.lower() in cls.verbs]
        if available_words:
            word = choice(available_words)
            transformed = choice(cls.verbs[word.lower()])
            text.transformed = text.transformed.replace(" " + word, " " + transformed)


class CommonErrors(Perturbation):
    weight = 10

    common_errors_path = resources_base_path / "common-errors.json"

    with open(common_errors_path) as f:
        errors_all = json.load(f)

    errors = {}

    for l_errors in errors_all:
        for current_form in l_errors:
            if current_form:
                errors[current_form] = l_errors.copy()
                errors[current_form].remove(current_form)

    def __init__(self):
        pass

    @classmethod
    def is_applicable(cls, text: Text):
        words = word_tokenize(text.transformed)
        return any(word.lower() in cls.errors for word in words)

    @classmethod
    def perturb(cls, text: Text) -> None:

        words = word_tokenize(text.transformed)
        available_words = [word for word in words if word.lower() in cls.errors]
        if available_words:
            word = choice(available_words)
            transformed = choice(cls.errors[word.lower()])
            text.transformed = text.transformed.replace(" " + word, " " + transformed)


_, cls_members = zip(*inspect.getmembers(sys.modules[__name__], inspect.isclass))

PERTURBATIONS: List[Perturbation] = [
    cls for cls in cls_members if issubclass(cls, Perturbation) and cls.weight
]
WEIGHTS = [cls.weight for cls in PERTURBATIONS]
WEIGHTS = [w / sum(WEIGHTS) for w in WEIGHTS]
