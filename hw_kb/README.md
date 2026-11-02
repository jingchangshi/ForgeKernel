# Hardware knowledge base (hw_kb)

An operator-agnostic library of device-certified hardware-interface recipes, one tree per kernel
language and, inside it, one tree per GPU / NPU architecture:

```
hw_kb/
  cutedsl/           # CuTe DSL runs (task language "cutedsl")
    README.md        # rules, layout and fill strategy
    TEMPLATE.md
    sm90/  sm100/
  cuda/              # CUDA C++ runs (task language "cuda")
    README.md
    TEMPLATE.md
    sm100/
  tilelang-ascend/   # TileLang Python DSL runs on the Ascend NPU (task language "tilelang-ascend")
    README.md
    TEMPLATE.md
    ascend950pr/
```

A run reads and writes exactly one tree, `hw_kb/<language>/<arch>/`, picked by the task's `language`
and `arch` (`src/forgekernel/loop/lang_mode.py`). The language trees are independent: a reading
certified with one language's probe is not carried over to another. See each tree's README for
the rules.
