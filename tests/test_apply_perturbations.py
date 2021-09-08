from Levenshtein import ratio

from xumi.perturbations.apply_perturbations import apply_perturbation_to_text
from xumi.text import Text


def test_apply_perturbation_to_text():
    text = Text(original="This is a sample sentence")
    before = text.transformed

    apply_perturbation_to_text(text)

    assert 0.6 < ratio(before, text.transformed) <= 1
