"""Language modes: the CuTe DSL mode, the CUDA mode and the TileLang mode, kept apart.

A loop ``language`` (loop.toml ``[loop].language`` / ``forge loop --language``)
used to steer one thing only -- the ``{LANGUAGE}`` token in the student and
distill prompts. Everything else the loop hands a run (the prompts, the hw_kb
tree, the timeline instrumentation, the census and pricing harnesses) was
written for a Python CuTe DSL ``kernel.py``. A CUDA C++ student (``kernel.cu``,
language ``cuda-sol``) then read DSL recipes, was told to keep a Python
``def _seg`` marker, and bought timeline / pricing jobs that could not read or
outrun its kernel.

This module is the one place that says which MODE a language belongs to and
what each mode reads, so the modes stay independent:

- the **cutedsl mode** (``CUTEDSL_LANGUAGES``) reads ``meta/<prompt>``,
  ``hw_kb/cutedsl/<arch>/``, ``docs/kb-provenance/``. Nothing here changes a byte
  of its path.
- the **cuda mode** (``CUDA_LANGUAGES``) reads its own prompt set
  (``meta/cuda/<prompt>`` wherever one exists, the shared language-neutral prompt
  otherwise), its own KB root (``hw_kb/cuda/<arch>/``) and files its provenance
  under ``docs/kb-provenance/cuda/``. Its tools key on the kernel file's suffix
  (``.cu``), which ``language_problem`` keeps in step with the language.
- the **tilelang mode** (``TILELANG_LANGUAGES``) is TileLang's own Python DSL, on
  the Ascend NPU backend here: its KB root is ``hw_kb/tilelang-ascend/<arch>/``
  (``arch = ascend950pr``), its provenance lands under
  ``docs/kb-provenance/tilelang-ascend/`` and its prompt overlay is
  ``meta/tilelang-ascend/`` (absent today: the shared prompt is used, which is
  what a language-neutral prompt is for).

**A language belongs to exactly one mode, and a language this module does not
know is an error, not a default.** ``mode_of`` used to answer "anything that is
not CUDA is the CuTe DSL mode". Registering ``tilelang-ascend`` under that rule
would have handed a TileLang student the CuTe DSL knowledge base, the DSL
recipes and the ``cutlass.cute`` probe tools -- silently, with no line anywhere
saying the run was reading a KB for an API its kernel cannot compile. So the
three sets are spelled out and ``UnknownLanguage`` is raised for anything else;
a new language is registered here first (mode, KB root, provenance directory,
prompt overlay), and only then does a run get past the launch gate.
"""
from __future__ import annotations

from pathlib import Path

MODE_CUTEDSL = "cutedsl"
MODE_CUDA = "cuda"
MODE_TILELANG = "tilelang"

# The languages whose student writes a CUDA C++ kernel file the card builds with
# nvcc. cuda-sol is the leaderboard cuda_cpp solution (one .cu with its own
# PYBIND11_MODULE binding, built the way SOL-ExecBench builds it); it is the only
# CUDA language with a card harness behind it.
CUDA_LANGUAGES = frozenset({"cuda-sol"})
CUDA_KERNEL_SUFFIXES = (".cu",)

# The CuTe DSL mode's languages. Spelled out rather than "everything that is not
# CUDA": that default is exactly what let an unclassified language read another
# mode's KB unnoticed (loop/lang_mode.py's module docstring).
CUTEDSL_LANGUAGES = frozenset({"cutedsl", "cuda-cpp", "inline-ptx", "triton",
                               "cutlass-cpp"})

# TileLang's Python DSL (tilelang.language) on the Ascend NPU backend: the student
# edits a kernel.py the card builds with tilelang's own compiler.
TILELANG_LANGUAGES = frozenset({"tilelang-ascend"})
TILELANG_KERNEL_SUFFIXES = (".py",)

MODES = (MODE_CUTEDSL, MODE_CUDA, MODE_TILELANG)

KB_ROOT = {MODE_CUTEDSL: "hw_kb/cutedsl", MODE_CUDA: "hw_kb/cuda",
           MODE_TILELANG: "hw_kb/tilelang-ascend"}
# Under the shared docs/kb-provenance/ root, so every "startswith docs/kb-provenance/"
# listing (harvest leftovers, publish) still covers it, while an entry name two
# KBs share (gpu_probe_ladder, ...) keeps one provenance file per KB.
PROVENANCE_SUBDIR = {MODE_CUTEDSL: "", MODE_CUDA: "cuda", MODE_TILELANG: "tilelang-ascend"}
PROMPT_OVERLAY_DIR = {MODE_CUTEDSL: None, MODE_CUDA: "cuda", MODE_TILELANG: "tilelang-ascend"}

DEFAULT_LANGUAGE = "cutedsl"
DEFAULT_CUDA_LANGUAGE = "cuda-sol"
DEFAULT_TILELANG_LANGUAGE = "tilelang-ascend"
DEFAULT_MODE = MODE_CUTEDSL


class UnknownLanguage(ValueError):
    """A loop language no mode claims: the run cannot tell what its student writes,
    which KB it reads or which prompts it gets, so nothing is guessed."""

    def __init__(self, language):
        self.language = language
        known = sorted(CUTEDSL_LANGUAGES | CUDA_LANGUAGES | TILELANG_LANGUAGES)
        super().__init__(
            f"loop language {language!r} belongs to no mode: register it in "
            f"forgekernel/loop/lang_mode.py (its mode, KB root, provenance directory "
            f"and prompt overlay) instead of letting it read another mode's KB "
            f"(known languages: {known})")


def mode_of(language: str | None) -> str:
    """The mode a loop language belongs to.

    ``None`` / ``""`` is "no language named": the run's default, the CuTe DSL mode
    (what every card written before the key existed gets). Any other string has to be
    one of the registered languages -- an unclassified one raises rather than being
    read as the DSL mode.
    """
    if language is None or language == "":
        return DEFAULT_MODE
    if language in CUDA_LANGUAGES:
        return MODE_CUDA
    if language in TILELANG_LANGUAGES:
        return MODE_TILELANG
    if language in CUTEDSL_LANGUAGES:
        return MODE_CUTEDSL
    raise UnknownLanguage(language)


def is_cuda(language: str | None) -> bool:
    return mode_of(language) == MODE_CUDA


def is_tilelang(language: str | None) -> bool:
    return mode_of(language) == MODE_TILELANG


def is_cuda_kernel_file(kernel_file: str | Path | None) -> bool:
    return str(kernel_file or "").endswith(CUDA_KERNEL_SUFFIXES)


def kb_root_name(language: str | None) -> str:
    """The KB root directory (under the repo) the language's mode reads."""
    return KB_ROOT[mode_of(language)]


def kb_repo_root(kb_dir: Path) -> Path:
    """The repository a mode's KB root (``hw_kb/<mode>``) sits in."""
    return Path(kb_dir).parents[1]


def provenance_dir(repo_root: Path, language: str | None) -> Path:
    """Where the mode's KB provenance lives (human review only, no agent read set)."""
    base = Path(repo_root) / "docs" / "kb-provenance"
    sub = PROVENANCE_SUBDIR[mode_of(language)]
    return base / sub if sub else base


def owns_provenance(rel_path: str, language: str | None) -> bool:
    """Whether a repo-relative ``docs/kb-provenance/...`` path belongs to the
    language's mode: a mode with its own subdirectory owns that subtree, the CuTe DSL
    mode owns everything else under the root (the shared per-op ``gaps/`` archive
    included)."""
    mode = mode_of(language)
    path = str(rel_path)
    sub = PROVENANCE_SUBDIR[mode]
    if sub:
        return path.startswith(f"docs/kb-provenance/{sub}/")
    taken = tuple(f"docs/kb-provenance/{s}/" for m, s in PROVENANCE_SUBDIR.items()
                  if s and m != mode)
    return not path.startswith(taken)


def prompt_path(meta_dir: Path, name: str, language: str | None) -> Path:
    """``meta/<overlay>/<name>`` for a mode with a prompt overlay when that file
    exists, else ``meta/<name>``. A prompt with no overlay is language-neutral and
    shared."""
    overlay = PROMPT_OVERLAY_DIR[mode_of(language)]
    if overlay:
        p = Path(meta_dir) / overlay / name
        if p.is_file():
            return p
    return Path(meta_dir) / name


def language_for_kernel_file(kernel_file: str | Path | None) -> str | None:
    """The language a kernel file implies when the run names none: a ``.cu`` is
    the CUDA mode's student; anything else leaves the old default alone."""
    return DEFAULT_CUDA_LANGUAGE if is_cuda_kernel_file(kernel_file) else None


def language_problem(language: str, kernel_file: str | Path) -> str | None:
    """Why ``language`` cannot drive a student that edits ``kernel_file``, or None.

    The language picks the prompts and the KB, the file picks what the card
    builds; out of step, the student is told to write CuTe DSL into a ``.cu`` (or
    C++ into a ``.py``) and every tool reads the wrong thing."""
    mode = mode_of(language)
    cu = is_cuda_kernel_file(kernel_file)
    if mode == MODE_CUDA and not cu:
        return (f"language {language!r} is the CUDA mode but the kernel file is "
                f"{kernel_file!r}: a CUDA student edits a .cu (scaffold the card with "
                f"--kernel-file kernel.cu)")
    if mode == MODE_CUTEDSL and cu:
        return (f"the kernel file {kernel_file!r} is CUDA C++ but language is "
                f"{language!r}: set loop.toml [loop].language = "
                f"{DEFAULT_CUDA_LANGUAGE!r} (or leave it unset to follow the file)")
    if mode == MODE_TILELANG and cu:
        return (f"the kernel file {kernel_file!r} is CUDA C++ but language is "
                f"{language!r}, which is the TileLang mode (a Python kernel.py): set "
                f"loop.toml [loop].language = {DEFAULT_CUDA_LANGUAGE!r} for a .cu")
    return None
