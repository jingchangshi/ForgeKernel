# Reproduction and verification scripts

These scripts produce the measurements recorded in `../evidence/`. Paths are not
hardcoded: `_paths.py` resolves the workspace roots from the environment or from
sibling checkouts, so the directory can be moved and the scripts still run.

| Script | Purpose |
| --- | --- |
| `_paths.py` |  |
| `audit_perf_report.py` | check every number in the perf report against the evidence files |
| `canary_model.py` | does the configured model answer correctly |
| `canary_provider.py` | does the model provider answer at all |
| `check_docs.py` | validate markdown fences/links and that the evidence JSON parses |
| `check_fp64_device.py` | whether float64 is real on the device |
| `check_fp64_downgrade.py` | the silent float64 -> float32 downgrade on .double() |
| `check_vendor_api.py` | vendor contract: shapes, dtypes, the rstd second return |
| `check_vendor_eps.py` | vendor epsilon placement against the task formula |
| `contention_bias.py` | shift of the paired ratio under a controlled saturated co-tenant |
| `contention_intensity.py` | ratio shift as a function of contention intensity |
| `e2e_gpu_entry.sh` | the job entrypoint the ForgeKernel job protocol calls |
| `fk_ascend_smoke.py` | smoke path through the adapter against a real card |
| `hostbound_aggregation_loss.py` | how a per-workload max differs from a re-derived geomean |
| `inspect_environment.py` | CANN / torch / torch_npu / TileLang versions and fingerprints |
| `inspect_msprof_attribution.py` | per-pipe attribution (aiv_vec / mte2 / mte3) from an msprof trace |
| `lead_e2e_fk_job.py` | drive a real ForgeKernel job end to end on the NPU |
| `measure_canary_cost.py` | cost of one in-window canary measurement |
| `negative_controls.py` | the five negative controls, each of which must fail |
| `npu_background_load.py` | a controlled in-window co-tenant used to test the witnesses |
| `paired_bench.py` | interleaved A/B and B/A paired timing in one window |
| `perturbation_test.py` | single-process perturbation of the timing |
| `precision_discrimination.py` | does the precision gate separate a correct from an accumulating kernel |
| `probe_bf16_rmsnorm.py` | compile and time the BF16 RMSNorm example against the vendor op |
| `recompute_headline.py` | recompute the headline ratios from the raw per-window samples |
| `replay_under_new_rules.py` | re-judge recorded windows under the current exclusion rule |
| `sample_size_study.py` | bootstrap interval width against the number of pairs |
| `screen_cheat_vectors.py` | what a name regex does and does not catch |
| `sweep_rows.py` | row-tile sweep; produced the 1.76x at rows=1->4 and the rows=8 UB ceiling |
| `ulp_bound_analysis.py` | the 6-combination ulp table behind the precision floor |
| `verified_contention.py` | contention effect with witnesses recorded on both sides |
| `verify_exclusion_behaviour.py` | contended/sparse windows stay out of the decision sample |
| `verify_no_contamination.py` | a verdict passes the quiet-fraction gate without leaking |
| `verify_registry.py` | the language registry resolves tilelang-ascend and rejects unknowns |
| `verify_sandbox_modes.py` | the three agent-policy modes differ only where they should |
| `verify_tool_access.py` | which tools are reachable under the sandbox policy |
| `watch_for_contention.py` | poll a card for genuine co-tenant activity |
| `xprocess_perturbation.py` | cross-process perturbation of the timing |
