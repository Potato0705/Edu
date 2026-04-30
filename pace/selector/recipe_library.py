"""Fixed recipe library for prompt-aware PACE selection.

The selector deliberately chooses from a small discrete library. This keeps
PARS auditable and prevents it from becoming an unconstrained hyperparameter
search over test folds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    description: str
    lambda_qwk: float
    decode_mode: str
    blend_alpha: float = 1.0
    max_delta_frac: float = 1.0
    mmd_enable: bool = False
    lr: float = 1.0e-3
    hidden_dim: int = 512
    dropout: float = 0.1
    lambda_sep: float = 0.0
    mmd_raw_boundary_epsilon: float = 1.0
    mmd_num_bands: int = 3
    mmd_project_dim: int = 128
    mmd_project_dropout: float = 0.0
    mmd_warmup_epochs: int = 5
    simplicity_rank: int = 0
    extra: Dict[str, float | str | bool] = field(default_factory=dict)

    @property
    def uses_blend(self) -> bool:
        return self.decode_mode == "blend_round"


def recipe_library_v1() -> Dict[str, Recipe]:
    """Return the first fixed PARS recipe library.

    R1/R2 use the moderate QWK surrogate that worked for mid-range prompts.
    R3/R4 use the stronger QWK surrogate and conservative blend decode that
    stabilized wide-range prompts in the current experiments.
    """
    recipes = [
        Recipe(
            recipe_id="R0",
            description="ordinal-only threshold calibrator",
            lambda_qwk=0.0,
            decode_mode="threshold",
            mmd_enable=False,
            lr=1.0e-3,
            simplicity_rank=0,
        ),
        Recipe(
            recipe_id="R1",
            description="qwkfix threshold calibrator",
            lambda_qwk=0.25,
            decode_mode="threshold",
            mmd_enable=False,
            lr=1.0e-3,
            simplicity_rank=1,
        ),
        Recipe(
            recipe_id="R2",
            description="qwkfix threshold calibrator with boundary MMD",
            lambda_qwk=0.25,
            decode_mode="threshold",
            mmd_enable=True,
            lr=1.0e-4,
            lambda_sep=0.05,
            mmd_raw_boundary_epsilon=1.0,
            simplicity_rank=3,
        ),
        Recipe(
            recipe_id="R3",
            description="wide-range qwkfix blend-round calibrator",
            lambda_qwk=2.0,
            decode_mode="blend_round",
            blend_alpha=0.65,
            max_delta_frac=0.10,
            mmd_enable=False,
            lr=3.0e-4,
            simplicity_rank=2,
        ),
        Recipe(
            recipe_id="R4",
            description="wide-range qwkfix blend-round calibrator with boundary MMD",
            lambda_qwk=2.0,
            decode_mode="blend_round",
            blend_alpha=0.65,
            max_delta_frac=0.10,
            mmd_enable=True,
            lr=3.0e-4,
            lambda_sep=0.05,
            mmd_raw_boundary_epsilon=2.0,
            simplicity_rank=4,
        ),
    ]
    return {recipe.recipe_id: recipe for recipe in recipes}


def recipe_library_v2() -> Dict[str, Recipe]:
    """Expanded PARS library for wide-range raw-collapse cases.

    v2 keeps R0-R4 unchanged and adds a small number of controlled wide-range
    alternatives. These are meant to be selected by inner-val only, not by
    looking at outer-test performance.
    """
    recipes = recipe_library_v1()
    recipes.update(
        {
            "R5": Recipe(
                recipe_id="R5",
                description="wide-range qwkfix blend-round calibrator with medium cap",
                lambda_qwk=2.0,
                decode_mode="blend_round",
                blend_alpha=0.65,
                max_delta_frac=0.20,
                mmd_enable=False,
                lr=3.0e-4,
                simplicity_rank=5,
            ),
            "R6": Recipe(
                recipe_id="R6",
                description="wide-range qwkfix blend-round calibrator with medium cap and boundary MMD",
                lambda_qwk=2.0,
                decode_mode="blend_round",
                blend_alpha=0.65,
                max_delta_frac=0.20,
                mmd_enable=True,
                lr=3.0e-4,
                lambda_sep=0.05,
                mmd_raw_boundary_epsilon=2.0,
                simplicity_rank=6,
            ),
            "R7": Recipe(
                recipe_id="R7",
                description="wide-range strong-qwk threshold calibrator",
                lambda_qwk=2.0,
                decode_mode="threshold",
                mmd_enable=False,
                lr=3.0e-4,
                simplicity_rank=7,
            ),
        }
    )
    return recipes


def manual_prompt_recipe_v1(prompt: int) -> str:
    """Current hand-written recipe mapping used as a comparison baseline."""
    if prompt == 2:
        return "R2"
    if prompt in {3, 4}:
        return "R0"
    if prompt == 7:
        return "R4"
    if prompt == 8:
        return "R3"
    return "R1"


def get_recipe_library(name: str = "v1") -> Dict[str, Recipe]:
    if name == "v1":
        return recipe_library_v1()
    if name == "v2":
        return recipe_library_v2()
    raise ValueError(f"Unsupported recipe library: {name}")


def recipe_ids() -> List[str]:
    return list(recipe_library_v1().keys())
