import abc
from random import choice

from xumi.text import Text
from xumi.perturbations.utilities import substitute_character, remove_character, add_character


class Perturbation(abc.ABC):
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
        words = text.transformed.split(" ")
        if words:
            word = choice(words)
            text.transformed = text.transformed.replace(word, "")


class PerturbRandomWord(Perturbation):
    weight = 5

    def __init__(self):
        pass

    @classmethod
    def perturb(cls, text: Text) -> None:

        words = text.transformed.split(" ")
        if words:
            word = choice(words)
            func = choice([substitute_character, remove_character, add_character])
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
