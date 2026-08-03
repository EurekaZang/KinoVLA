# A7 Method Ablations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete A7 method-ablation evidence to publication-grade by normalizing artifacts, closing available real-stack gaps, generating `A实验/A7.md`, updating project progress docs, and verifying the pipeline.

**Architecture:** A7 is a reducer-plus-targeted-eval experiment over frozen A0-A6 artifacts and trained VLA adapters. The implementation keeps `configs/eval/a7.yaml` as the source of truth, extends `kino_vla/eval/a7_ablation.py` with deterministic reducers, updates `scripts/a7_eval.py` and `scripts/a7_report.py` to emit publication-ready summaries, and records unavailable real dependencies as explicit blocked artifacts rather than substitutes.

**Tech Stack:** Python 3.11, PyYAML, NumPy, pytest, existing KiNO VLA SFT/eval stack, JSON artifacts under `outputs/eval/a7`, markdown reports under `A实验/`.

## Global Constraints

- REAL STACK ONLY — NO SURROGATE: do not replace missing real `ApiOracle`, CLIP, VLA, Isaac, or trained encoder artifacts with synthetic/CPU substitutes.
- DETERMINISM IS A RED LINE: do not add new closed-loop numbers unless they inherit A0.1-gated artifacts.
- Do not edit `kino-vla-v2.md`.
- Every A-experiment needs `A实验/<Exp>.md` with Why, Claim/logical role, Method, Results with CIs + controls, claim bridge/falsifier, and Honest scope.
- Every number must be traceable to config hash, seeds, commit, and source hashes.
- Use `env -u PYTHONPATH ...` for tests/commands to avoid ROS/lark collection issues.
- Do not commit changes unless the user explicitly asks; use verification and `git diff` instead of commit checkpoints.
- Use at most five subagents total for this A7 completion pass.
- Do not repeat the existing seed0 conflict-dose experiments for doses 0/5/10/20/40; reuse cached seed0 adapters/per-item artifacts and only aggregate them.

---

## File Structure

Modify these files:

- `kino_vla/eval/a7_ablation.py`: add reusable reducers for seed aggregation, ERS/regret summaries, status counting, and report-safe formatting inputs.
- `scripts/a7_eval.py`: normalize all A7 artifacts under the current config hash; aggregate text-schema, conflict-dose, CoT, encoder, ERS/regret, OOD-theta, and theta-star artifacts; write a complete manifest.
- `scripts/a7_report.py`: generate a publication-grade `A实验/A7.md` that satisfies the A-experiment structure and does not overclaim missing arms.
- `tests/test_a7_ablation.py`: add unit tests for new reducers.
- `tests/test_a7_datasets.py`: add dataset-card/build-summary regression checks if build output changes.
- `CLAUDE.md`: after successful verification, update §2 A7 status, §4 completed log, and §5 live scope items.

Create or regenerate these files:

- `outputs/eval/a7/datasets/build_summary.json`
- `outputs/eval/a7/text_schema_existing.json`
- `outputs/eval/a7/text_schema/summary.json`
- `outputs/eval/a7/conflict_dose/summary.json`
- `outputs/eval/a7/a3_method_slices.json`
- `outputs/eval/a7/addons.json`
- `outputs/eval/a7/cot_filter_summary.json`
- `outputs/eval/a7/encoder_summary.json`
- `outputs/eval/a7/ers_regret.json`
- `outputs/eval/a7/a7_eval_manifest.json`
- `A实验/A7.md`

Do not create additional abstractions unless a test requires them.

---

### Task 1: Add A7 reducer helpers and tests

**Files:**
- Modify: `kino_vla/eval/a7_ablation.py`
- Modify: `tests/test_a7_ablation.py`

**Interfaces:**
- Consumes: existing `wilson(k: int, n: int)`, `mcnemar(a_correct: list[bool], b_correct: list[bool])`, `artifact_meta(...)`, `load_json(...)`, `write_json(...)`.
- Produces:
  - `status_counts(rows: dict[str, dict[str, Any]]) -> dict[str, int]`
  - `aggregate_dose_curves(curves: dict[str, dict[str, Any]]) -> dict[str, Any]`
  - `mean_rate_cells(cells: list[dict[str, Any]]) -> dict[str, Any]`
  - `ers_regret_from_agent_rows(agent_rows: dict[str, list[dict[str, Any]]], m_cost: dict[str, dict[str, float]], canonical_label: dict[str, str]) -> dict[str, Any]`

- [ ] **Step 1: Add failing tests for status and dose aggregation**

Append this to `tests/test_a7_ablation.py`:

```python
from kino_vla.eval.a7_ablation import aggregate_dose_curves, status_counts


def test_status_counts_counts_rows_by_status():
    rows = {
        "seed0_dose00": {"status": "available"},
        "seed0_dose05": {"status": "available"},
        "seed1_dose00": {"status": "requires_run"},
        "seed1_dose05": {"status": "blocked"},
    }
    assert status_counts(rows) == {"available": 2, "blocked": 1, "requires_run": 1}


def test_aggregate_dose_curves_reports_mean_best_and_nonmonotone():
    curves = {
        "seed0": {
            "curve": [
                {"dose": 0, "rate": 0.4, "n": 10, "k": 4, "ci": [0.2, 0.6]},
                {"dose": 5, "rate": 0.6, "n": 10, "k": 6, "ci": [0.3, 0.8]},
                {"dose": 10, "rate": 0.5, "n": 10, "k": 5, "ci": [0.2, 0.7]},
            ],
            "monotone_non_decreasing": False,
        },
        "seed1": {
            "curve": [
                {"dose": 0, "rate": 0.5, "n": 10, "k": 5, "ci": [0.2, 0.7]},
                {"dose": 5, "rate": 0.7, "n": 10, "k": 7, "ci": [0.4, 0.9]},
                {"dose": 10, "rate": 0.6, "n": 10, "k": 6, "ci": [0.3, 0.8]},
            ],
            "monotone_non_decreasing": False,
        },
    }
    out = aggregate_dose_curves(curves)
    assert out["n_seeds"] == 2
    assert out["monotone_all_seeds"] is False
    assert out["best_mean_dose"] == 5
    assert out["by_dose"]["5"]["mean_rate"] == 0.65
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
env -u PYTHONPATH pytest tests/test_a7_ablation.py::test_status_counts_counts_rows_by_status tests/test_a7_ablation.py::test_aggregate_dose_curves_reports_mean_best_and_nonmonotone -q
```

Expected: FAIL with `ImportError` for `aggregate_dose_curves` or `status_counts`.

- [ ] **Step 3: Implement status and dose aggregation helpers**

Append this near `dose_curve_summary` in `kino_vla/eval/a7_ablation.py`:

```python
def status_counts(rows: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Count A7 row statuses in stable sorted-key order."""
    counts: Counter[str] = Counter()
    for row in rows.values():
        counts[str(row.get("status", "missing"))] += 1
    return dict(sorted(counts.items()))


def mean_rate_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Average already-reduced proportion cells across seeds without pretending raw trials pooled."""
    if not cells:
        return {"n_cells": 0, "mean_rate": 0.0, "min_rate": 0.0, "max_rate": 0.0}
    rates = [float(c.get("rate", c.get("acc", 0.0))) for c in cells]
    return {
        "n_cells": len(cells),
        "mean_rate": round(sum(rates) / len(rates), 3),
        "min_rate": round(min(rates), 3),
        "max_rate": round(max(rates), 3),
    }


def aggregate_dose_curves(curves: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed dose curves for report-level interpretation."""
    by_dose: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for curve in curves.values():
        for rec in curve.get("curve", []):
            by_dose[str(int(rec["dose"]))].append(rec)
    reduced = {dose: mean_rate_cells(cells) for dose, cells in sorted(by_dose.items(), key=lambda kv: int(kv[0]))}
    best_dose = None
    if reduced:
        best_dose = max(reduced.items(), key=lambda kv: (kv[1]["mean_rate"], -int(kv[0])))[0]
    return {
        "n_seeds": len(curves),
        "by_dose": reduced,
        "best_mean_dose": None if best_dose is None else int(best_dose),
        "monotone_all_seeds": bool(curves) and all(bool(c.get("monotone_non_decreasing")) for c in curves.values()),
    }
```

- [ ] **Step 4: Run reducer tests**

Run:

```bash
env -u PYTHONPATH pytest tests/test_a7_ablation.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Add ERS/regret reducer tests**

Append this to `tests/test_a7_ablation.py`:

```python
from kino_vla.eval.a7_ablation import ers_regret_from_agent_rows


def test_ers_regret_from_agent_rows_uses_a4_cost_matrix():
    m_cost = {
        "matched_O4_twophase": {"backstep_detour": 1.0, "high_step": 4.0},
        "matched_O2": {"backstep_detour": 1.0, "high_step": 0.0},
    }
    canonical = {"matched_O4_twophase": "backstep_detour", "matched_O2": "high_step"}
    rows = {
        "good": [
            {"scenario": "matched_O4_twophase", "primitive": "backstep_detour"},
            {"scenario": "matched_O2", "primitive": "high_step"},
        ],
        "bad": [
            {"scenario": "matched_O4_twophase", "primitive": "high_step"},
            {"scenario": "matched_O2", "primitive": "backstep_detour"},
        ],
    }
    out = ers_regret_from_agent_rows(rows, m_cost, canonical)
    assert out["good"]["mean_cost"] == 0.5
    assert out["good"]["mean_regret"] == 0.0
    assert out["bad"]["mean_cost"] == 2.5
    assert out["bad"]["mean_regret"] == 2.0
```

- [ ] **Step 6: Run the ERS test and verify it fails**

Run:

```bash
env -u PYTHONPATH pytest tests/test_a7_ablation.py::test_ers_regret_from_agent_rows_uses_a4_cost_matrix -q
```

Expected: FAIL with `ImportError` for `ers_regret_from_agent_rows`.

- [ ] **Step 7: Implement ERS/regret reducer**

Append this near the other reducers in `kino_vla/eval/a7_ablation.py`:

```python
def ers_regret_from_agent_rows(
    agent_rows: dict[str, list[dict[str, Any]]],
    m_cost: dict[str, dict[str, float]],
    canonical_label: dict[str, str],
) -> dict[str, Any]:
    """Project agent primitive choices through the A4 cost matrix.

    Each row must provide a scenario key and a primitive/label. Rows with scenarios or labels not in
    the matrix are counted as missing and excluded from means; this prevents accidental fabrication.
    """
    out: dict[str, Any] = {}
    for agent, rows in sorted(agent_rows.items()):
        costs: list[float] = []
        regrets: list[float] = []
        missing = 0
        for row in rows:
            scenario = str(row.get("scenario") or row.get("a4_scenario") or row.get("scenario_id"))
            label = str(row.get("primitive") or row.get("label") or row.get("pred_label"))
            scenario_costs = m_cost.get(scenario)
            canonical = canonical_label.get(scenario)
            if scenario_costs is None or canonical is None or label not in scenario_costs or canonical not in scenario_costs:
                missing += 1
                continue
            cost = float(scenario_costs[label])
            best_cost = float(scenario_costs[canonical])
            costs.append(cost)
            regrets.append(cost - best_cost)
        out[agent] = {
            "n": len(rows),
            "n_scored": len(costs),
            "n_missing": missing,
            "mean_cost": round(sum(costs) / max(1, len(costs)), 3),
            "mean_regret": round(sum(regrets) / max(1, len(regrets)), 3),
        }
    return out
```

- [ ] **Step 8: Run all A7 reducer tests**

Run:

```bash
env -u PYTHONPATH pytest tests/test_a7_ablation.py -q
```

Expected: PASS.

---

### Task 2: Normalize A7 aggregation and add ERS/regret artifact

**Files:**
- Modify: `scripts/a7_eval.py`
- Test: `tests/test_a7_ablation.py` from Task 1

**Interfaces:**
- Consumes:
  - `status_counts(rows)` from Task 1
  - `aggregate_dose_curves(curves)` from Task 1
  - `ers_regret_from_agent_rows(agent_rows, m_cost, canonical_label)` from Task 1
- Produces:
  - `outputs/eval/a7/ers_regret.json`
  - `outputs/eval/a7/conflict_dose/summary.json` containing `status_counts` and `aggregate`
  - `outputs/eval/a7/a7_eval_manifest.json` with full stage/status/source-hash coverage

- [ ] **Step 1: Update imports in `scripts/a7_eval.py`**

Replace the import block from `kino_vla.eval.a7_ablation` with:

```python
from kino_vla.eval.a7_ablation import (
    aggregate_dose_curves,
    artifact_meta,
    dose_curve_summary,
    ers_regret_from_agent_rows,
    load_json,
    load_yaml,
    repo_path,
    status_counts,
    wilson,
    write_json,
)
```

- [ ] **Step 2: Add scenario mapping helpers**

Add this after `_addons` in `scripts/a7_eval.py`:

```python
OPERATOR_TO_A4_SCENARIO = {
    "O4_tether": "matched_O4_twophase",
    "O2_compliance": "matched_O2",
    "O7_visual_physics": "O7_looks_safe",
    "O8_invisible_collider": "O8_invisible",
    "O5_payload": "O5_payload_B",
    "O10_effort_decay": "O10_decay_B",
    "O1_low_friction": "O1_A_nominal",
}

ATTR_TO_A4_LABEL = {
    "adhesion": "backstep_detour",
    "compliant_terrain": "high_step",
    "low_friction": "slow_low",
    "invisible_obstacle": "detour_replan",
    "overload": "hold_request",
    "effort_decay": "crawl",
    "nominal": "continue",
}


def _row_to_a4_projection(row: dict[str, Any]) -> dict[str, Any] | None:
    op = row.get("operator") or row.get("operator_name") or row.get("true_operator")
    scenario = OPERATOR_TO_A4_SCENARIO.get(str(op))
    pred_attr = row.get("pred_attr") or row.get("attribution") or row.get("predicted_attribution")
    primitive = row.get("primitive") or row.get("pred_primitive") or ATTR_TO_A4_LABEL.get(str(pred_attr))
    if scenario is None or primitive is None:
        return None
    return {"scenario": scenario, "primitive": primitive}
```

- [ ] **Step 3: Add `_ers_regret` function**

Add this after `_row_to_a4_projection` in `scripts/a7_eval.py`:

```python
def _ers_regret(cfg: dict[str, Any], config_path: str) -> dict[str, Any]:
    a4 = load_json(cfg["sources"]["a4_results"])
    agents: dict[str, list[dict[str, Any]]] = {}
    text_dir = repo_path(cfg["output_dir"]) / "text_schema"
    conflict_dir = repo_path(cfg["output_dir"]) / "conflict_dose"

    for name in cfg["text_schemas"]:
        p = text_dir / f"per_item_{name}.json"
        if p.exists():
            rows = load_json(p)
            agents[f"text_schema:{name}"] = [x for r in rows if (x := _row_to_a4_projection(r)) is not None]

    for p in sorted(conflict_dir.glob("per_item_seed*_dose*.json")):
        rows = load_json(p)
        stem = p.stem.replace("per_item_", "")
        agents[f"conflict_dose:{stem}"] = [x for r in rows if (x := _row_to_a4_projection(r)) is not None]

    projected = ers_regret_from_agent_rows(agents, a4["M_mean_cost"], a4["canonical_label"])
    result = {
        **artifact_meta(
            config_path,
            sources={
                "a4_results": cfg["sources"]["a4_results"],
                "text_schema_dir": str(text_dir),
                "conflict_dose_dir": str(conflict_dir),
            },
        ),
        "stage": "ers_regret",
        "status": "available" if projected else "requires_evaluated_per_item",
        "projection_rule": "A7 predictions mapped to A4 scenario/label cost rows; missing mappings excluded and counted",
        "rows": projected,
    }
    write_json(repo_path(cfg["output_dir"]) / "ers_regret.json", result)
    return result
```

- [ ] **Step 4: Enhance conflict-dose summary**

In `_conflict_dose`, after `status_values = ...`, replace the two lines setting status and writing JSON with:

```python
    summary["status_counts"] = status_counts(summary["rows"])
    if summary.get("curves"):
        summary["aggregate"] = aggregate_dose_curves(summary["curves"])
    summary["status"] = "available" if status_values and all(s == "available" for s in status_values) else "requires_runs"
    write_json(out_dir / "summary.json", summary)
    return summary
```

- [ ] **Step 5: Include ERS/regret in `run()`**

In `run()`, after the existing `if stage in ("all", "conflict-dose"):` block, add:

```python
    if stage in ("all", "existing"):
        results["ers_regret"] = _ers_regret(cfg, config_path)
```

- [ ] **Step 6: Improve manifest content**

Replace the `manifest = ...` line in `run()` with:

```python
    manifest = {
        **artifact_meta(
            config_path,
            sources={
                "config": config_path,
                "a0_corpus": cfg["sources"]["a0_corpus"],
                "a3_battery": cfg["sources"]["a3_battery"],
                "a4_results": cfg["sources"]["a4_results"],
                "a5_ood_theta": cfg["sources"]["a5_ood_theta"],
                "a6_theta_sweep": cfg["sources"]["a6_theta_sweep"],
            },
        ),
        "stage": stage,
        "evaluate": evaluate,
        "seeds_filter": None if seeds_filter is None else sorted(seeds_filter),
        "results": {k: v.get("status") for k, v in results.items()},
    }
```

- [ ] **Step 7: Run targeted syntax/test check**

Run:

```bash
env -u PYTHONPATH python -m py_compile scripts/a7_eval.py kino_vla/eval/a7_ablation.py
env -u PYTHONPATH pytest tests/test_a7_ablation.py -q
```

Expected: both commands pass.

---

### Task 3: Make the A7 report publication-grade

**Files:**
- Modify: `scripts/a7_report.py`
- Create/regenerate: `A实验/A7.md`

**Interfaces:**
- Consumes A7 JSON artifacts from Task 2:
  - `text_schema_existing.json`
  - `text_schema/summary.json`
  - `a3_method_slices.json`
  - `conflict_dose/summary.json`
  - `cot_filter_summary.json`
  - `encoder_summary.json`
  - `addons.json`
  - `ers_regret.json`
  - `a7_eval_manifest.json`
- Produces: Markdown report with required sections and honest scope.

- [ ] **Step 1: Add helper formatting functions**

In `scripts/a7_report.py`, after `_table`, add:

```python
def _status_line(obj: dict[str, Any]) -> str:
    status = obj.get("status", "missing")
    if status in {"available", "partial_available"}:
        return f"`{status}`"
    reason = obj.get("reason") or obj.get("scope_note") or obj.get("path") or "not available"
    return f"`{status}` — {reason}"


def _hash_lines(*artifacts: dict[str, Any]) -> str:
    seen: dict[str, str] = {}
    for artifact in artifacts:
        for name, value in artifact.get("source_hashes", {}).items():
            seen[name] = value
    rows = [["source", "sha256(12)"]]
    for name, value in sorted(seen.items()):
        rows.append([name, str(value)])
    return _table(rows)


def _dose_aggregate_text(conflict: dict[str, Any]) -> str:
    aggregate = conflict.get("aggregate") or {}
    if not aggregate:
        return "No fully evaluated multi-seed aggregate is available yet."
    rows = [["dose", "mean O4 attr", "min", "max", "seed cells"]]
    for dose, cell in sorted(aggregate.get("by_dose", {}).items(), key=lambda kv: int(kv[0])):
        rows.append([
            dose,
            f"{cell['mean_rate']:.3f}",
            f"{cell['min_rate']:.3f}",
            f"{cell['max_rate']:.3f}",
            str(cell["n_cells"]),
        ])
    return _table(rows) + f"\n\nBest mean dose: `{aggregate.get('best_mean_dose')}`; monotone all seeds: `{aggregate.get('monotone_all_seeds')}`."
```

- [ ] **Step 2: Load ERS/regret in `build_report()`**

In `build_report`, after `addons = _load_optional(...)`, add:

```python
    ers = _load_optional(out / "ers_regret.json")
```

- [ ] **Step 3: Add ERS/regret table construction**

After the `a3_rows` block, add:

```python
    ers_rows = [["agent", "n scored", "mean cost", "mean regret", "missing"]]
    for agent, row in sorted((ers.get("rows") or {}).items()):
        ers_rows.append([
            agent,
            str(row.get("n_scored", 0)),
            f"{float(row.get('mean_cost', 0.0)):.3f}",
            f"{float(row.get('mean_regret', 0.0)):.3f}",
            str(row.get("n_missing", 0)),
        ])
```

- [ ] **Step 4: Replace the report body with a stricter A-experiment structure**

Replace the returned f-string in `build_report()` with a version that includes these sections exactly:

```markdown
# A7 — Method Ablations

> **Headline:** A7 completes the method-ablation story from real frozen artifacts where available: latent Kino-Tokens are efficient and θ-grounded, conflict-specific data has an empirical dose knee, consequence-aware ERS/regret links method choices back to A4, and unavailable real ApiOracle / full encoder-grid arms are explicitly blocked rather than substituted.

## 1. Why
...
## 2. Claim C1-C5 + logical role
...
## 3. Method
...
## 4. Results with CIs + controls
...
## 5. Claim bridge / falsifier
...
## 6. Honest scope
...
## 7. Reproduction
...
```

Use the existing text as the base, but include:

- Source hash table via `{_hash_lines(text_existing, a3, conflict, cot, enc, addons, ers, manifest)}`.
- Dose aggregate via `{_dose_aggregate_text(conflict)}`.
- ERS/regret table via `{_table(ers_rows)}`.
- Status lines for CoT/encoder via `{_status_line(cot)}` and `{_status_line(enc)}`.
- The exact reproduction commands already in the script.

- [ ] **Step 5: Run report generation after current artifacts**

Run:

```bash
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a7_report.py --config configs/eval/a7.yaml --out A实验/A7.md
```

Expected: `[OK] wrote /home/eureka/KinoVLA/A实验/A7.md`.

- [ ] **Step 6: Inspect report headings**

Run:

```bash
grep -n '^## ' A实验/A7.md
```

Expected headings include:

```text
## 1. Why
## 2. Claim C1-C5 + logical role
## 3. Method
## 4. Results with CIs + controls
## 5. Claim bridge / falsifier
## 6. Honest scope
## 7. Reproduction
```

---

### Task 4: Run A7 build/eval/report pipeline and close available gaps

**Files:**
- Regenerate: `outputs/eval/a7/**`
- Regenerate: `A实验/A7.md`

**Interfaces:**
- Consumes: all code from Tasks 1-3.
- Produces: current-hash A7 artifacts and report.

- [ ] **Step 1: Build A7 dataset cards under the current config**

Run:

```bash
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a7_build_datasets.py --config configs/eval/a7.yaml --stage all
```

Expected: JSON printed with `"stage": "all"` and written keys for `schema`, `conflict_dose`, and `cot_filter`.

- [ ] **Step 2: Aggregate existing/imported A7 evidence without VLA inference**

Run:

```bash
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a7_eval.py --config configs/eval/a7.yaml --stage existing
```

Expected: `a7_eval_manifest.json` has statuses for `text_schema_existing`, `a3_method_slices`, `addons`, `cot_filter`, `encoder`, and `ers_regret`.

- [ ] **Step 3: Aggregate conflict-dose availability without rerunning VLA inference**

Run:

```bash
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a7_eval.py --config configs/eval/a7.yaml --stage conflict-dose
```

Expected: `outputs/eval/a7/conflict_dose/summary.json` reports all configured seeds/doses and `status_counts` indicating which adapters are available vs requires-run.

- [ ] **Step 4: Reuse existing seed-0 conflict-dose results**

Run the non-inference aggregator only:

```bash
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a7_eval.py --config configs/eval/a7.yaml --stage conflict-dose --seeds 0
```

Expected: cached seed0 per-item artifacts and adapters for doses 0/5/10/20/40 are reused to reconstruct `curves.seed0`; no VLA inference or retraining is launched for the already-completed seed0 dose experiments.

- [ ] **Step 5: Evaluate text-schema adapters if all real adapters exist**

Run:

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 KINOVLA_MODEL_ID=/home/eureka/models/Qwen3-VL-4B-Instruct ~/miniconda3/envs/kinovla/bin/python scripts/a7_eval.py --config configs/eval/a7.yaml --stage text-schema --evaluate
```

Expected: if configured adapters exist, `outputs/eval/a7/text_schema/summary.json` has rows for `latent`, `reflect_scalar`, and `rich_stats`. If any adapter is missing, the row remains `requires_run`.

- [ ] **Step 6: Re-run full aggregator to refresh manifest**

Run:

```bash
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a7_eval.py --config configs/eval/a7.yaml --stage all
```

Expected: manifest contains all A7 stages and current config hash.

- [ ] **Step 7: Generate final A7 report**

Run:

```bash
env -u PYTHONPATH ~/miniconda3/envs/kinovla/bin/python scripts/a7_report.py --config configs/eval/a7.yaml --out A实验/A7.md
```

Expected: `A实验/A7.md` is regenerated.

---

### Task 5: Update CLAUDE.md progress records

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: verified `A实验/A7.md` and `outputs/eval/a7/` artifacts.
- Produces: session-end progress record matching project protocol.

- [ ] **Step 1: Read current A7 artifact headline values**

Run:

```bash
env -u PYTHONPATH python - <<'PY'
import json
from pathlib import Path
root = Path('outputs/eval/a7')
for rel in ['text_schema_existing.json','conflict_dose/summary.json','encoder_summary.json','addons.json','ers_regret.json','a7_eval_manifest.json']:
    p = root / rel
    obj = json.loads(p.read_text()) if p.exists() else {'status': 'missing'}
    print(rel, obj.get('status'), obj.get('config_sha256'))
    if rel == 'text_schema_existing.json':
        print('headline', obj.get('headline'))
    if rel == 'conflict_dose/summary.json':
        print('status_counts', obj.get('status_counts'), 'aggregate', obj.get('aggregate'))
    if rel == 'encoder_summary.json':
        print('encoder', obj.get('status'), obj.get('scope_note'))
    if rel == 'addons.json':
        print('theta_star', obj.get('theta_star'), 'B5 AUROC', obj.get('ood_theta_abstention', {}).get('B5-conflict-bi'))
PY
```

Expected: print enough values to write an honest one-line A7 summary.

- [ ] **Step 2: Update CLAUDE §2 A7 status**

Edit `CLAUDE.md` line that currently says pending A7. Replace it with a concise status, for example:

```text
- **A7 DONE/PARTIAL (C2 method ablations).** Real available ablations consolidated under current config: latent/text route evidence, A3 method slices, conflict-dose curve where adapters exist, ERS/regret projection through A4, OOD-θ/θ* add-ons. Raw unfiltered ApiOracle CoT and full encoder grid remain explicitly blocked/requires-run, not substituted. See `A实验/A7.md`, `outputs/eval/a7/`.
```

If all configured real arms were actually run, use `A7 DONE`; otherwise keep `DONE/PARTIAL` and the blocker wording.

- [ ] **Step 3: Update CLAUDE §4 completed log**

Add one line inside the completed-log code block after the A6 line:

```text
2026-07-07 | A7 method ablations consolidated: latent-vs-text, conflict-dose, ERS/regret, OOD-θ/θ* method add-ons; unavailable real ApiOracle/full encoder grid explicitly blocked. | A实验/A7.md; outputs/eval/a7/
```

- [ ] **Step 4: Update CLAUDE §5 live deviations**

Replace the final pending-experiments line with:

```text
A7 SCOPE — no surrogate substitution. Real artifact-backed A7 rows are reportable; raw unfiltered ApiOracle CoT and full contrastive/from-scratch/window/gate encoder grid remain requires-run before those specific ablation claims. See A实验/A7.md.
No post-A7 external benchmark is part of the current paper scope.
```

If the raw CoT or encoder grid was completed during Task 4, remove that part of the A7 scope sentence.

---

### Task 6: Verification and final review

**Files:**
- Verify all modified files and generated artifacts.

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: final verification status and honest blocker list.

- [ ] **Step 1: Run targeted A7 tests**

Run:

```bash
env -u PYTHONPATH pytest tests/test_a7_ablation.py tests/test_a7_datasets.py -q
```

Expected: PASS.

- [ ] **Step 2: Run py_compile on A7 scripts**

Run:

```bash
env -u PYTHONPATH python -m py_compile kino_vla/eval/a7_ablation.py scripts/a7_build_datasets.py scripts/a7_eval.py scripts/a7_report.py scripts/a7_train.py
```

Expected: PASS with no output.

- [ ] **Step 3: Run project fast gate if feasible**

Run:

```bash
env -u PYTHONPATH pytest -m "not slow" -q
```

Expected: PASS. If environment issues unrelated to A7 appear, record the exact failing output and keep A7 targeted verification as the primary evidence.

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git diff -- configs/eval/a7.yaml kino_vla/eval/a7_ablation.py scripts/a7_build_datasets.py scripts/a7_eval.py scripts/a7_report.py tests/test_a7_ablation.py tests/test_a7_datasets.py A实验/A7.md CLAUDE.md docs/superpowers/specs/2026-07-07-a7-method-ablations-design.md docs/superpowers/plans/2026-07-07-a7-method-ablations.md
```

Expected: diff only contains A7-related changes and docs.

- [ ] **Step 5: Inspect git status**

Run:

```bash
git status --short
```

Expected: A7 files and pre-existing user changes are visible. Do not commit unless explicitly asked.

- [ ] **Step 6: Final summary**

Report:

- What A7 artifacts were generated.
- Which tests passed/failed.
- Which real dependencies remain blocked, if any.
- Whether CLAUDE.md §2/§4/§5 was updated.
- That no commit was made unless the user requested one.
