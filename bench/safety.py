"""safety.py — static gate for model-written variant files, applied BEFORE anything runs.

Glimmer's proposals are Python that will be executed on this machine. A variant only
needs MLX and Glimmer's own model module, so anything else is refused:
  * imports outside a small allow-list (no os, subprocess, shutil, pathlib, socket, ...)
  * calls to eval / exec / compile / open / __import__ / getattr-on-strings tricks
  * missing pieces of the variant interface (NAME, EXACT, apply, post_load)
This does not make arbitrary code safe; it rejects everything a legitimate variant
has no reason to contain.
"""
import ast

ALLOWED_IMPORTS = {
    "mlx", "mlx.core", "mlx.nn", "mlx.utils",
    "mlx_lm", "mlx_lm.models", "mlx_lm.models.muse_glimmer", "mlx_lm.models.base",
    "math", "functools", "typing", "dataclasses",
}
BANNED_NAMES = {"eval", "exec", "compile", "open", "__import__", "input",
                "globals", "locals", "vars", "breakpoint", "exit", "quit"}
REQUIRED = {"NAME", "EXACT", "apply", "post_load"}


def check_source(src: str):
    """Return a list of problems (empty list = passes the gate)."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    problems, defined = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    problems.append(f"import not allowed: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            full = {f"{mod}.{a.name}" for a in node.names}
            if mod not in ALLOWED_IMPORTS and not (full & ALLOWED_IMPORTS):
                problems.append(f"import not allowed: from {mod}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and \
                node.attr not in {"__call__", "__init__"}:
            problems.append(f"dunder attribute not allowed: {node.attr}")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    missing = REQUIRED - defined
    if missing:
        problems.append(f"missing required top-level definitions: {sorted(missing)}")
    return problems


MODEL_PARTS = ["Attention", "MLP", "DecoderLayer", "MuseGlimmerModel", "Model", "QuantizedLinear",
               "quantized_matmul", "scaled_dot_product_attention", "RoPE", "compile", "gate_proj",
               "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj", "lm_head", "rope"]


def identifiers(src: str) -> set:
    """Every identifier the code actually uses (names, attributes, imported names) — comments,
    docstrings and strings are ignored."""
    ids = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Name):
            ids.add(node.id)
        elif isinstance(node, ast.Attribute):
            ids.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            ids |= {a.name for a in node.names}
    return ids


def only_norm_change(src: str) -> bool:
    """True if the code uses CenteredRMSNorm and none of the model's other parts
    (heuristic used to reject repeats of the already-measured H1 idea in the guided arm)."""
    try:
        ids = identifiers(src)
    except SyntaxError:
        return False
    return "CenteredRMSNorm" in ids and not (ids & set(MODEL_PARTS))
