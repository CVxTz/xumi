import random
from random import choices

from xumi.perturbations.perturbations import PERTURBATIONS, WEIGHTS
from xumi.text import Text


def apply_perturbation_to_text(
    text: Text, freq: int = 4, skip_probability: float = 0.5
) -> None:
    if random.uniform(0, 1) < skip_probability:
        return
    perturbations = choices(PERTURBATIONS, weights=WEIGHTS, k=freq)
    for perturbation in perturbations:
        if perturbation.is_applicable(text):
            perturbation.perturb(text)
