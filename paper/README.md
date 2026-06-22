# Kino-VLA — ICRA paper sources

LaTeX sources for the Kino-VLA conference paper (ICRA format, IEEEtran two-column).

## Status
All prose sections drafted (rigorous, citation-complete; compiles to an **8-page** PDF):
- Abstract
- Section I — Introduction
- Section II — Related Work
- Section III — The Kino-VLA System: overview + latency budget (Table I),
  Kino-Tokens privileged distillation, recovery planner + semantic map, the CBF-QP
  shield with the forward-invariance derivation (Eqs. 1–4), and the training pipeline
- Section IV — Experiments: setup + Kino-Fail benchmark, attribution results
  (Table II), privileged-θ regression (Table III), the CBF shield, the data filter,
  embodied DPO, and the closed-loop demonstration with limitations
- Section V — Conclusion
- Figure 1 — system architecture (self-contained TikZ; the five-layer closed loop,
  dual-rate loops, privileged-θ distillation, μ̂ coupling, semantic map)

The paper is complete: all text sections, three tables (I–III), four equations
(1–4), a TikZ system figure, and 51 verified references.

## Build
The sources build with TeX Live / MiKTeX or on Overleaf out of the box, and have
been compiled successfully here with a user-space **TinyTeX** install
(`~/.TinyTeX`, binaries symlinked into `~/.local/bin`). Beyond the default TinyTeX
set, the build needs the packages `ieeetran`, `cite`, and
`collection-fontsrecommended` (the latter supplies the Courier/Helvetica/Times
metrics IEEEtran uses for `\texttt` and headings), plus `pgf` for the TikZ system
figure; install them with
`tlmgr install ieeetran cite collection-fontsrecommended pgf`. The last full build is
clean: 0 errors, 0 undefined references, 0 overfull boxes, and all 51 `\cite` keys
resolve to `references.bib` with no orphan entries.

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

The paper is set for **double-blind review** (anonymous author block). For a
camera-ready version, fill in the real author block in `main.tex` and remove the
`\def\isanonymous{1}` line near the top.

## Citations
Every entry in `references.bib` was verified to exist via web search (title,
authors, venue/year, and arXiv/DOI confirmed); no entry is fabricated. A few
conventions worth a final check in a reference manager before submission:
- **RT-2** is cited as "Brohan et al." (the field's convention); the CoRL 2023
  camera-ready lists Zitkovich as first author.
- **DoReMi** is given as IROS 2024 (widely cited as such; arXiv:2307.00329).
- **Qwen3-VL** (arXiv:2511.21631) is the planner backbone; its author list is
  abbreviated with `and others`.
- Venues to keep straight: Helpful DoggyBot is IROS 2025, COME-robot is ICRA 2025,
  and `$\pi_0$` is an arXiv technical report (cited as `@misc`, no peer-reviewed venue).

## Scope guardrails (keep the claims honest)
- All results are in **Isaac Lab simulation** on a physically simulated Unitree
  Go2; there is no hardware experiment, and the text states this explicitly.
- The CBF forward-invariance guarantee is a property of the **reduced-order
  (capture-point / LIP) model**. On the full-order policy the shield's
  demonstrated behavior is command clamping, not a zero-fall guarantee; do not
  upgrade the abstract's hedge ("under a reduced-order model") in later sections.
- The training recipe is Kino-SFT followed by embodied DPO, but the headline
  number is the **SFT** attribution result (0.97 vs 0.24 for the rule-based FSM).
  Do **not** claim that DPO improves closed-loop success over SFT; that margin is
  not established.
