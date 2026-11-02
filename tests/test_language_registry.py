"""The language registry: every loop language has ONE mode, and an unknown one stops
the run instead of silently reading another mode's knowledge base.

``loop/lang_mode.mode_of()`` used to answer "anything that is not CUDA is the CuTe DSL
mode". That default is harmless only while the DSL mode IS the fallback; the moment a
second non-CUDA mode exists (``tilelang-ascend``) every language the registry forgot
-- a typo, a new name, a card copied from another branch -- reads the CuTe DSL KB, the
DSL prompts and the ``cutlass.cute`` tools, with nothing anywhere saying the run is
reading a KB for an API its kernel cannot compile. So the registry is explicit and
unknown is an error; these tests pin that, the three modes' read sets, and the arch
spelling each language accepts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from forgekernel import task as task_mod
from forgekernel.loop import lang_mode
from forgekernel.task import TaskError, load_task

SHAPES = [{"name": "small", "axes": {"m": 128, "n": 256}}]


# -- the registry -------------------------------------------------------------------------

def test_every_registered_language_belongs_to_exactly_one_mode():
    seen: dict[str, str] = {}
    for mode, langs in ((lang_mode.MODE_CUTEDSL, lang_mode.CUTEDSL_LANGUAGES),
                        (lang_mode.MODE_CUDA, lang_mode.CUDA_LANGUAGES),
                        (lang_mode.MODE_TILELANG, lang_mode.TILELANG_LANGUAGES)):
        for lang in langs:
            assert lang not in seen, f"{lang} is claimed by {seen.get(lang)} and {mode}"
            seen[lang] = mode
            assert lang_mode.mode_of(lang) == mode
    # Every mode is reachable, and every table below covers every mode.
    assert set(seen.values()) == set(lang_mode.MODES)
    for table in (lang_mode.KB_ROOT, lang_mode.PROVENANCE_SUBDIR,
                  lang_mode.PROMPT_OVERLAY_DIR):
        assert set(table) == set(lang_mode.MODES)


def test_an_unknown_language_raises_instead_of_reading_the_cutedsl_mode():
    """The whole point: this must NOT quietly return MODE_CUTEDSL. A typo'd language
    that fell back to the DSL mode is a run reading the wrong KB with a straight face."""
    for bogus in ("tilelang", "tilelang_ascend", "ascend", "TileLang-Ascend", "cutedsl2"):
        with pytest.raises(lang_mode.UnknownLanguage) as e:
            lang_mode.mode_of(bogus)
        assert repr(bogus) in str(e.value)
        # the message names the fix, not just the failure
        assert "lang_mode.py" in str(e.value) and "known languages" in str(e.value)
    # ... and every function that goes through mode_of refuses it too, rather than
    # defaulting one of its tables.
    for call in (lambda: lang_mode.is_cuda("tilelang"),
                 lambda: lang_mode.is_tilelang("nope"),
                 lambda: lang_mode.kb_root_name("nope"),
                 lambda: lang_mode.provenance_dir(Path("/tmp"), "nope"),
                 lambda: lang_mode.owns_provenance("docs/kb-provenance/x.md", "nope"),
                 lambda: lang_mode.prompt_path(Path("/tmp"), "p.md", "nope"),
                 lambda: lang_mode.language_problem("nope", "kernel.py")):
        with pytest.raises(lang_mode.UnknownLanguage):
            call()


def test_no_language_named_is_still_the_default_mode():
    """None / "" is "this run names no language" (older cards, the default card), not an
    unknown language: it must keep reading exactly what it always did."""
    for empty in (None, ""):
        assert lang_mode.mode_of(empty) == lang_mode.MODE_CUTEDSL
        assert lang_mode.kb_root_name(empty) == "hw_kb/cutedsl"


def test_the_tilelang_mode_reads_its_own_kb_provenance_and_prompt_overlay(tmp_path):
    assert lang_mode.mode_of("tilelang-ascend") == lang_mode.MODE_TILELANG
    assert lang_mode.is_tilelang("tilelang-ascend") and not lang_mode.is_cuda("tilelang-ascend")
    assert lang_mode.kb_root_name("tilelang-ascend") == "hw_kb/tilelang-ascend"
    # The three modes' roots are three different directories: no run can read two.
    roots = {lang_mode.kb_root_name(lang) for lang in ("cutedsl", "cuda-sol", "tilelang-ascend")}
    assert len(roots) == 3
    assert lang_mode.provenance_dir(tmp_path, "tilelang-ascend") == (
        tmp_path / "docs" / "kb-provenance" / "tilelang-ascend")
    assert lang_mode.kb_repo_root(tmp_path / "hw_kb" / "tilelang-ascend") == tmp_path


def test_provenance_ownership_is_exclusive_across_the_three_modes():
    dsl, cu, tl = ("docs/kb-provenance/x.md", "docs/kb-provenance/cuda/x.md",
                   "docs/kb-provenance/tilelang-ascend/x.md")
    gaps = "docs/kb-provenance/gaps/op.md"
    # each mode owns its own subtree and nothing else
    assert lang_mode.owns_provenance(dsl, "cutedsl") and lang_mode.owns_provenance(gaps, "cutedsl")
    assert not lang_mode.owns_provenance(cu, "cutedsl")
    assert not lang_mode.owns_provenance(tl, "cutedsl")
    assert lang_mode.owns_provenance(cu, "cuda-sol")
    assert not lang_mode.owns_provenance(dsl, "cuda-sol")
    assert not lang_mode.owns_provenance(tl, "cuda-sol")
    assert lang_mode.owns_provenance(tl, "tilelang-ascend")
    assert not lang_mode.owns_provenance(dsl, "tilelang-ascend")
    assert not lang_mode.owns_provenance(cu, "tilelang-ascend")
    assert not lang_mode.owns_provenance(gaps, "tilelang-ascend")


def test_the_tilelang_mode_reads_its_prompt_overlay_and_falls_back_to_the_shared_prompt(tmp_path):
    (tmp_path / "tilelang-ascend").mkdir()
    for n in ("student_prompt.md", "diagnose_prompt.md"):
        (tmp_path / n).write_text("shared")
    (tmp_path / "tilelang-ascend" / "student_prompt.md").write_text("tilelang")
    got = lang_mode.prompt_path(tmp_path, "student_prompt.md", "tilelang-ascend")
    assert got == tmp_path / "tilelang-ascend" / "student_prompt.md"
    # a prompt with no overlay is language-neutral and shared
    assert lang_mode.prompt_path(tmp_path, "diagnose_prompt.md", "tilelang-ascend") == (
        tmp_path / "diagnose_prompt.md")
    # and the tilelang student never reads the CUDA overlay
    assert lang_mode.prompt_path(tmp_path, "student_prompt.md", "tilelang-ascend") != (
        tmp_path / "cuda" / "student_prompt.md")


def test_a_kernel_file_only_implies_a_cuda_language_never_another_mode():
    assert lang_mode.language_for_kernel_file("kernel.cu") == lang_mode.DEFAULT_CUDA_LANGUAGE
    for f in ("kernel.py", "kernel.tl", "", None):
        assert lang_mode.language_for_kernel_file(f) is None


def test_the_language_and_the_kernel_file_must_agree_in_all_three_modes():
    assert lang_mode.language_problem("cuda-sol", "kernel.cu") is None
    assert lang_mode.language_problem("cutedsl", "kernel.py") is None
    assert lang_mode.language_problem("tilelang-ascend", "kernel.py") is None
    assert "CUDA mode" in lang_mode.language_problem("cuda-sol", "kernel.py")
    assert "cuda-sol" in lang_mode.language_problem("cutedsl", "kernel.cu")
    # A TileLang student writes a Python kernel; a .cu would be built by the wrong toolchain.
    tl = lang_mode.language_problem("tilelang-ascend", "kernel.cu")
    assert tl and "TileLang" in tl and "cuda-sol" in tl


# -- task registration --------------------------------------------------------------------

def _kb(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for kb in ("hw_kb/cutedsl/sm100", "hw_kb/cuda/sm100",
               "hw_kb/tilelang-ascend/ascend950pr"):
        (root / kb).mkdir(parents=True, exist_ok=True)
        (root / kb / "index.md").write_text("# index\n", encoding="utf-8")
    return root


def _task(tmp_path: Path, toml: str) -> Path:
    d = tmp_path / "tasks" / "rmsnorm"
    d.mkdir(parents=True)
    (d / "task.toml").write_text(toml)
    (d / "description.md").write_text("out = x / rms(x) * w\n")
    import json
    (d / "shapes.json").write_text(json.dumps(SHAPES))
    return d


def test_tilelang_ascend_is_a_registered_task_language(tmp_path):
    assert "tilelang-ascend" in task_mod.LANGUAGES
    assert task_mod.LOOP_LANGUAGE["tilelang-ascend"] == "tilelang-ascend"
    toml = '[task]\nlanguage = "tilelang-ascend"\narch = "ascend950pr"\n'
    t = load_task(_task(tmp_path, toml), kb_root=_kb(tmp_path))
    assert t.language == "tilelang-ascend" and t.arch == "ascend950pr"
    assert t.loop_language == "tilelang-ascend"


def test_every_task_language_has_a_loop_language_and_an_arch_spelling():
    for lang in task_mod.LANGUAGES:
        assert lang in task_mod.LOOP_LANGUAGE
        assert lang in task_mod.ARCH_RE, lang
        # the loop language it maps to is one lang_mode knows (no silent fallback)
        assert lang_mode.mode_of(task_mod.LOOP_LANGUAGE[lang]) in lang_mode.MODES


def test_an_ascend_arch_passes_validation_and_an_sm_arch_does_not_for_tilelang(tmp_path):
    """The old regex was ^sm\\d+[a-z]?$ -- an NPU part could not be written at all."""
    ok = load_task(_task(tmp_path, '[task]\nlanguage = "tilelang-ascend"\n'
                                     'arch = "ascend950pr"\n'), kb_root=_kb(tmp_path))
    assert ok.arch == "ascend950pr"
    bad = _task(tmp_path / "bad", '[task]\nlanguage = "tilelang-ascend"\narch = "sm100"\n')
    with pytest.raises(TaskError) as e:
        load_task(bad, kb_root=_kb(tmp_path))
    assert any("ascend950pr" in p for p in e.value.problems), e.value.problems
    # ... and a CUDA language still refuses an NPU arch
    bad2 = _task(tmp_path / "bad2", '[task]\nlanguage = "cutedsl"\narch = "ascend950pr"\n')
    with pytest.raises(TaskError) as e2:
        load_task(bad2, kb_root=_kb(tmp_path))
    assert any("sm90" in p for p in e2.value.problems), e2.value.problems


def test_the_kb_dir_follows_the_language(tmp_path):
    root = _kb(tmp_path)
    t = load_task(_task(tmp_path, '[task]\nlanguage = "tilelang-ascend"\n'
                                  'arch = "ascend950pr"\n'), kb_root=root)
    kb = task_mod._kb_dir(root, t.loop_language, t.arch)
    assert kb == root / "hw_kb" / "tilelang-ascend" / "ascend950pr"
    assert kb.is_dir()


def test_a_tilelang_task_with_no_kb_tree_is_refused_naming_the_missing_tree(tmp_path):
    root = _kb(tmp_path)
    import shutil
    shutil.rmtree(root / "hw_kb" / "tilelang-ascend" / "ascend950pr")
    with pytest.raises(TaskError, match="no hardware knowledge base"):
        load_task(_task(tmp_path, '[task]\nlanguage = "tilelang-ascend"\n'
                                  'arch = "ascend950pr"\n'), kb_root=root)


# -- the card -----------------------------------------------------------------------------

def test_the_tilelang_card_gets_a_python_kernel_file():
    from forgekernel import taskcard
    assert taskcard.KERNEL_FILE["tilelang-ascend"] == "kernel.py"


# -- the DSL source -----------------------------------------------------------------------

def test_the_tilelang_dsl_source_is_resolved_for_the_tilelang_language(monkeypatch, tmp_path):
    """A tilelang-ascend run's dsl_source_dirs is the tilelang package, not the CuTe DSL
    one: the student is offline and can only read what this points at."""
    from forgekernel.loop import dsl_source
    pkg = tmp_path / "tilelang"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    monkeypatch.setattr(dsl_source, "probe_tilelang_source", lambda *a, **k: pkg.resolve())
    got = dsl_source.resolve({"loop": {"language": "tilelang-ascend"}},
                             language="tilelang-ascend", submit_env={}, python="python3",
                             probe_env={})
    assert got.dirs == (pkg.resolve(),)
    assert got.origin == dsl_source.ORIGIN_DEFAULT
    assert "tilelang" in " ".join(got.notes)


def test_the_tilelang_source_can_be_declared_and_wins(tmp_path):
    from forgekernel.loop import dsl_source
    declared = tmp_path / "mytilelang"
    declared.mkdir()
    got = dsl_source.resolve({"loop": {"tilelang_source_dirs": [str(declared)]}},
                             language="tilelang-ascend", submit_env={}, python="python3",
                             probe_env={})
    assert got.dirs == (declared.resolve(),) and got.origin == dsl_source.ORIGIN_EXPLICIT


def test_a_tilelang_source_that_overlaps_the_reference_is_not_defaulted(monkeypatch, tmp_path):
    from forgekernel.loop import dsl_source
    pkg = tmp_path / "refs" / "reference" / "tilelang"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    monkeypatch.setattr(dsl_source, "probe_tilelang_source", lambda *a, **k: pkg.resolve())
    got = dsl_source.resolve({"loop": {"ref_source_paths": [str(tmp_path / "refs")]}},
                             language="tilelang-ascend", submit_env={}, python="python3",
                             probe_env={})
    assert got.dirs == () and got.origin != dsl_source.ORIGIN_DEFAULT
    assert any("reference" in n for n in got.notes)


def test_the_cutedsl_language_still_resolves_the_cutedsl_way(monkeypatch, tmp_path):
    """The tilelang branch must not change what a cutedsl run reads."""
    from forgekernel.loop import dsl_source
    called = {"tilelang": False}

    def _boom(*a, **k):
        called["tilelang"] = True
        return None

    monkeypatch.setattr(dsl_source, "probe_tilelang_source", _boom)
    monkeypatch.setattr(dsl_source, "probe_interpreter",
                        lambda *a, **k: {"version": None, "roots": []})
    got = dsl_source.resolve({}, language=dsl_source.LANGUAGE_CUTEDSL, submit_env={},
                             python="python3", probe_env={})
    assert not called["tilelang"]
    assert got.dirs == () and got.origin != dsl_source.ORIGIN_EXPLICIT
