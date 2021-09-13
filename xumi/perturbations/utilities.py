import string
from random import choice, randint


def remove_character(word: str):
    characters = [a for a in word]
    if not characters:
        return word
    _len = len(characters)
    i = randint(0, _len - 1)
    characters.pop(i)

    return "".join(characters)


def add_character(word: str):
    characters = [a for a in word]
    _len = len(characters)
    char = choice(string.printable)
    if _len:
        i = randint(0, _len - 1)
        characters.insert(i, char)
    else:
        characters = [char]

    return "".join(characters)


def substitute_character(word: str):
    characters = [a for a in word]
    if not characters:
        return word
    _len = len(characters)
    i = randint(0, _len - 1)
    char = choice(string.printable)
    characters[i] = char

    return "".join(characters)


def uppercase(word: str):
    return word.upper()
