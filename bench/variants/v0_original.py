"""v0_original — the unmodified mlx-lm code. This is the reference every other variant is judged against."""

NAME = "v0_original"
EXACT = True          # must match the golden reference bit for bit (it IS the reference)
DESCRIPTION = "Unmodified mlx_lm/models/muse_glimmer.py (mlx-lm 0.32.2)."


def apply():
    """Called BEFORE the weights are loaded. Patch classes here. v0 changes nothing."""
    pass


def post_load(model):
    """Called AFTER the weights are loaded. Transform weights here. v0 changes nothing."""
    pass
