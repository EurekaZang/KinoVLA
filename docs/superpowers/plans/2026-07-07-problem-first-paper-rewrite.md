# Problem-first Paper Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `paper/main.tex` so the manuscript reads as a problem-first ICRA robotics paper rather than a technical evidence report.

**Architecture:** The rewrite keeps the current single-file IEEEtran manuscript structure but changes the narrative layer across all sections. The scientific facts, experiment numbers, scope guardrails, references, and artifact traceability remain unchanged; only exposition, section framing, captions, and result organization are revised.

**Tech Stack:** LaTeX (`IEEEtran`), BibTeX (`IEEEtran` bibliography style), existing Paper-A experiment artifacts and reports, local TeX toolchain.

## Global Constraints

- Do not edit `Kino-vla-v2.md`.
- Do not introduce surrogate or synthetic evidence as claim support.
- Do not add unverified references.
- Do not change reported numbers, hashes, seeds, or artifact scope.
- Do not claim latent Kino-Tokens dominate all text routes on greedy accuracy.
- Do not claim A6 shows a literal policy flip at exactly `theta*`.
- Do not claim O8 is solved.
- Do not present A7 as a new closed-loop benchmark.
- Do not present Paper-A as a general navigation-system benchmark.
- Use the problem-first narrative approved in `docs/superpowers/specs/2026-07-07-problem-first-paper-rewrite-design.md`.
- Commit only if the user explicitly asks; otherwise leave changes uncommitted.

---

## File Structure

- Modify: `paper/main.tex`
  - Responsibility: active ICRA manuscript source. All narrative rewrite changes land here.
- Modify if implementation reaches a successful rebuild: `paper/main.pdf`
  - Responsibility: compiled manuscript artifact generated from `main.tex`.
- Modify if BibTeX output changes during rebuild: `paper/main.bbl`
  - Responsibility: generated bibliography output.
- Modify after implementation if session record changes: `CLAUDE.md`
  - Responsibility: project progress log; update §4 with manuscript rewrite evidence and §5 only if a live deviation changes.
- Usually unchanged: `paper/README.md`
  - Responsibility: paper source status. Only edit if the rewrite changes the status description enough to warrant it.

---

### Task 1: Rewrite abstract and introduction around the robotics problem

**Files:**
- Modify: `paper/main.tex:42-74`

**Interfaces:**
- Consumes: Current title, keywords, existing citations in Introduction.
- Produces: Problem-first framing used by Method, Experiments, Discussion, and Conclusion.

- [ ] **Step 1: Replace the abstract with a problem-first abstract**

In `paper/main.tex`, replace the entire content between `\begin{abstract}` and `\end{abstract}` with:

```latex
Legged robots often know that something has gone wrong before they know why.  A slowing foot may indicate compliant terrain that should be crossed with a different gait, adhesion that should be escaped by backing out, or a hidden contact condition that vision alone cannot reveal.  We study this post-failure decision as \emph{cross-modal failure attribution}: inferring the physical cause of a recovery-relevant failure from visual context and recent proprioceptive history.  The problem is not generic multimodal fusion, because either modality can be non-identifying or misleading depending on the failure.  We introduce a deterministic evaluation protocol for quadrupedal recovery attribution using matched failure operators, frozen multimodal snapshots, and forced-label consequence measurements on a simulated Unitree Go2 with real rendering and trained attribution components.  The protocol asks whether one modality can explain every failure, whether multimodal access is sufficient for evidence arbitration, and whether wrong explanations change physical recovery outcome.  Across the Paper-A experiment battery, adhesion and compliant terrain are byte-identical at the proprioceptive decision interface while visually separable, whereas visual-remap and invisible-obstacle cases require proprioception despite misleading or absent visual evidence.  Single-modality and unshaped multimodal baselines fail complementary conflict regimes; conflict-trained attribution recovers the intended evidence direction.  When labels are forced through the recovery registry, correct labels succeed on crux cells while wrong labels fail and incur asymmetric physical costs.  Residual-based abstention further reduces expected cost near out-of-distribution operators and severity boundaries.  These results support a simple robotics principle: safe recovery should explain the failure before selecting the intervention.
```

- [ ] **Step 2: Rewrite the Introduction opening problem**

Replace `paper/main.tex:54-66` with:

```latex
A legged robot that loses progress is not facing a single recovery problem.  The same rise in tracking error or foot effort can mean that the foot is sinking into compliant terrain, sticking to an adhesive surface, slipping on low friction, carrying an unexpected payload, or contacting an obstacle that the camera did not explain.  These causes are not interchangeable.  Mud may be crossed by changing gait.  Adhesion may require backing out before re-planning.  Low friction calls for conservative constraints, while an invisible obstacle may require a topological update.  Detecting an anomaly is therefore only the first step; the robot still has to explain the anomaly well enough to choose a recovery.

This paper studies that explanatory step.  We ask: \emph{when a quadruped fails during navigation, how should it attribute the physical cause when visual and proprioceptive evidence disagree, and how much does that attribution matter for recovery?}  The question sits between two successful lines of robotics.  Proprioceptive adaptation methods infer latent dynamics and compensate continuously for terrain, payload, and actuator changes~\cite{lee2020learning,miki2022learning,kumar2021rma,nahrendra2023dreamwaq}.  Vision-language-action (VLA) and language-conditioned navigation systems bring semantic scene reasoning and task-level planning to robots~\cite{brohan2023rt2,driess2023palme,kim2024openvla,octo2024,cheng2024navila}.  Neither line, by itself, isolates the post-failure question of which evidence source should be trusted before selecting a discrete recovery primitive.

We call this decision \emph{cross-modal failure attribution}.  The input is a frozen failure snapshot containing recent body history and visual context.  The output is an attributed physical cause, such as \texttt{compliant\_terrain}, \texttt{adhesion}, \texttt{low\_friction}, \texttt{invisible\_obstacle}, \texttt{overload}, \texttt{effort\_decay}, or \texttt{nominal}, together with an admissible recovery primitive.  The important distinction is that the label is not an explanation attached after control.  It gates the intervention: a wrong label can select a wrong primitive even if the primitive controller is implemented correctly.

The difficulty is that neither sensing channel dominates.  In one direction, vision is necessary because proprioception can be non-identifying.  Our matched adhesion--mud pair, O4 versus O2, is constructed so the consumed proprioceptive binding window and action/torque trace are byte-identical at the decision interface, while RGB semantics separate the yellow adhesive surface from compliant terrain.  In the other direction, proprioception is necessary because vision can be deceptive or blind.  Visual-remap and invisible-obstacle operators make the image look safe, hazardous, or uninformative while the body reveals the true contact mode.

These cases make recovery attribution different from ordinary multimodal prediction.  A model may receive an image and a body history yet still learn a color shortcut, always trust the camera, always trust the body, or fuse features without learning when the sources conflict.  We therefore evaluate attribution through a block-structured taxonomy: agreement cases, vision-true conflicts, proprioception-true conflicts, proprioceptive fine-structure cases, and nominal or mild cases where intervention can be worse than continuing.  The expected empirical signature is complementary failure: a strong proprioceptive baseline should fail where the body is non-identifying, a vision-only baseline should fail where images are misleading or blind, and a conflict-trained attributor should cover both directions.

To make this question measurable, we combine two evaluation currencies.  Frozen snapshots hold the observation packet fixed across agents, enabling paired comparisons under identical visual and proprioceptive evidence.  Forced-label consequence measurements hold the physical scene fixed while changing only the attributed label passed through a pre-registered recovery registry.  This second measurement asks the robotics question directly: what happens if the robot recovers under the wrong explanation?

The resulting claim is not that a VLA model replaces locomotion control, mapping, or safety filtering.  Those are fixed infrastructure in this paper.  The claim is that safe recovery is attribution-gated: the robot should infer why the failure occurred, decide whether vision or proprioception is trustworthy in that regime, and only then select a recovery or abstain.
```

- [ ] **Step 3: Replace the contribution list**

Replace `paper/main.tex:68-74` with:

```latex
This paper makes four contributions.
\begin{itemize}
  \item We formulate quadrupedal recovery as cross-modal failure attribution: a post-failure decision that maps visual and proprioceptive evidence to a physical cause and an admissible recovery.
  \item We introduce a deterministic evaluation protocol with matched failure operators, frozen multimodal snapshots, paired statistics, and forced-label consequence matrices.
  \item We show bidirectional sensing necessity: some recovery-relevant failures are proprioceptively indistinguishable but visually separable, while others are visually misleading or blind but proprioceptively separable.
  \item We show that conflict-trained attribution changes physical recovery outcomes and supports conservative abstention, whereas single-modality and unshaped multimodal baselines fail complementary regimes.
\end{itemize}
```

- [ ] **Step 4: Search for accidental overclaiming in the rewritten Introduction**

Run:

```bash
grep -n "replace\|benchmark\|dominates\|solved\|policy flip\|closed-loop benchmark" /home/eureka/KinoVLA/paper/main.tex
```

Expected: No new Introduction sentence claims that VLA replaces locomotion, latent tokens dominate text, O8 is solved, or A7 is a closed-loop benchmark. Existing allowed uses such as `does not claim` are acceptable.

---

### Task 2: Smooth Related Work and Method into the problem-first narrative

**Files:**
- Modify: `paper/main.tex:80-190`

**Interfaces:**
- Consumes: Problem framing from Task 1.
- Produces: Method framing used by revised Experiments in Task 3.

- [ ] **Step 1: Replace Related Work paragraphs with gap-oriented prose**

Replace `paper/main.tex:80-93` with:

```latex
\subsection{Learned legged locomotion and proprioceptive adaptation}
Learned quadrupedal locomotion has made substantial progress through large-scale simulation, domain randomization, and teacher-student transfer.  Policies trained in massively parallel simulators~\cite{rudin2022learning,makoviychuk2021isaacgym,mittal2023orbit} and transferred with actuator models~\cite{hwangbo2019learning} can traverse rough terrain and recover from disturbances~\cite{lee2020learning,miki2022learning}.  Rapid Motor Adaptation and related approaches infer latent dynamics from proprioceptive history and condition a low-level policy on that latent~\cite{kumar2021rma,nahrendra2023dreamwaq}.  Recent work further studies specialized load, contact, and fall-recovery regimes~\cite{loadadapt2025,lu2025frnet}.  These methods are strongest when the response is continuous adaptation within the locomotion policy.  Our setting begins when the recovery choice is discrete or semantic: the same body trace may require crossing, backing out, holding, or updating a route depending on the attributed cause.

\subsection{Vision-language-action models and legged semantic navigation}
Robot foundation models and VLA policies connect language, vision, and action across manipulation and navigation tasks~\cite{brohan2023rt1,brohan2023rt2,driess2023palme,kim2024openvla,octo2024,black2024pi0}.  Language planners also sequence skills through affordance grounding or generated programs~\cite{ahn2022saycan,huang2022inner,liang2023codeaspolicies}.  For legged robots, recent systems use vision-language models for open-world navigation, object fetching, or quadruped action generation~\cite{cheng2024navila,mei2024quadrupedgpt,wu2024doggybot,ding2024quarvla}.  Paper-A does not treat the VLA as a replacement for the locomotion stack.  Instead, it isolates a smaller decision that such systems must eventually make: after the body reports a failure, should the recovery explanation come from the image, from proprioception, or from abstaining because the evidence is out of family?

\subsection{Vision--proprioception fusion and traversability}
Traversability estimation and semantic mapping integrate visual appearance with geometry, language, and sometimes proprioceptive feedback.  Open-vocabulary maps ground language in 3D scene representations using vision-language features~\cite{huang2023vlmaps,chen2023nlmap,huang2023voxposer}, while self-supervised traversability systems learn terrain costs from robot experience~\cite{frey2023wvn}.  Proprioceptive and multimodal navigation methods estimate terrain or robot-terrain interaction from body signals and vision~\cite{elnoor2024pronav,elnoor2024amco,elnoor2025vlmgronav}.  These systems predict where a robot can travel or how costly traversal may be.  Our benchmark asks a cause-specific recovery question: when the robot is already failing, which physical explanation should gate the intervention?  Matched operators deliberately create cases where traversability alone is insufficient because the same apparent difficulty can demand opposite recoveries.

\subsection{Failure detection, explanation, and recovery reasoning}
Robotic failure reasoning has increasingly used multimodal language models to explain errors and propose corrections.  REFLECT summarizes robot experience for failure explanation and correction~\cite{liu2023reflect}; AHA trains a vision-language model to detect and reason over manipulation failures~\cite{duan2025aha}; DoReMi grounds language models through detection and recovery from plan-execution misalignment~\cite{guo2024doremi}; and recent failure-CoT datasets scale failure reasoning for manipulation~\cite{pacaud2025guardian}.  In parallel, proprioceptive slip and anomaly detectors identify contact failures in legged systems~\cite{yan2025slip}, and introspective perception predicts when a sensing module is likely to fail~\cite{daftry2016introspective}.  Paper-A follows this explanatory direction but evaluates the explanation before recovery: frozen snapshots fix the evidence, and forced labels measure the physical consequence of choosing one explanation over another.

\subsection{Safety filters, abstention, and conservative recovery}
Safe robot learning often places a certified or conservative layer beneath a learned policy.  Control barrier functions and predictive safety filters provide formal tools for maintaining invariant safe sets or filtering unsafe actions~\cite{ames2017control,ames2019control,wabersich2018linear,brunke2022safe}, and recent quadruped systems use filtering for robust navigation in unknown environments~\cite{grandia2021multi,onefilter2024,tayal2023c3bf}.  Our focus is complementary to proving a new low-level safety certificate.  We study when the semantic recovery layer should intervene, continue, or abstain because attribution is uncertain.  The forced-label matrix supplies the cost model for wrong interventions, while residuals and severity-boundary tests calibrate conservative defaults.
```

- [ ] **Step 2: Replace Method opening and problem formulation prose**

Replace `paper/main.tex:99-120` with:

```latex
The method is an evaluation design for one decision in the recovery stack: the mapping from a failure observation to a cause-specific recovery.  We separate this decision from low-level locomotion, detection operating point, and route execution so that visual--proprioceptive evidence arbitration can be measured directly.

\subsection{Attribution-gated recovery}
\label{subsec:problem}

A failure event is represented by a frozen snapshot
\begin{equation}
    s = (x^{\mathrm{vis}}, x^{\mathrm{prop}}, u_{1:T}, c),
\end{equation}
where $x^{\mathrm{vis}}$ contains the RGB-D frames and appearance context, $x^{\mathrm{prop}}$ contains the recent proprioceptive state history, $u_{1:T}$ contains the commanded action or torque trace consumed by the evaluated agent, and $c$ stores task and scenario metadata.  In the implementation, the proprioceptive binding window is the observable interface used by the attributors: an $\mathrm{obs}_{48}$ history together with a 12-dimensional action/torque summary, cropped to fixed windows $T\in\{25,50,100\}$ when needed.  Privileged simulator parameters are logged for ground truth and analysis, but are never provided to evaluated agents.

The attributor predicts a semantic cause
\begin{equation}
    \hat{\ell} = f_\phi(s), \qquad \ell \in \mathcal{L},
\end{equation}
where $\mathcal{L}$ includes \texttt{low\_friction}, \texttt{compliant\_terrain}, \texttt{adhesion}, \texttt{overload}, \texttt{effort\_decay}, \texttt{invisible\_obstacle}, and \texttt{nominal}.  A pre-registered registry maps each cause to admissible recovery primitives $\mathcal{A}(\ell)$ and to one canonical primitive $a^\star(\ell)$ used for intervention studies.  A prediction is counted as attribution-gated correct only when the cause is correct and the selected primitive lies in the admissible set of the true cause,
\begin{equation}
    \mathbf{1}_{\mathrm{AGCR}}(s)=\mathbf{1}\{\hat{\ell}=\ell^\star(s)\}\,\mathbf{1}\{\hat{a}\in\mathcal{A}(\ell^\star(s))\}.
\end{equation}
This metric prevents a model from receiving credit for a lucky primitive paired with the wrong explanation.

The central contrast is between observing multiple modalities and arbitrating between them.  A model can observe both channels and still fail if it always trusts the camera, always trusts the body, or learns an appearance shortcut.  The benchmark therefore controls the information structure of each operator cell.
```

- [ ] **Step 3: Replace taxonomy and snapshot protocol prose**

Replace `paper/main.tex:125-147` with:

```latex
The benchmark is built from parameterized failure operators applied to the same simulated Go2 substrate.  Each operator has a physical mechanism, a semantic attribution label, and a recovery role.  The main text focuses on the operators that create recovery ambiguity: O2 compliant terrain, O4 adhesion, O7 visual remap, O8 invisible obstacle, O5 payload overload, and O10 effort decay.  Additional operators remain in the registry for coverage and ablations.

Operators are grouped by the evidence available at the moment of recovery:
\begin{itemize}
    \item \textbf{T1, agreement:} vision and proprioception point to the same cause.
    \item \textbf{T2, conflict with vision true:} proprioception is ambiguous or misleading, while visual semantics identify the cause.
    \item \textbf{T3, conflict with proprioception true:} visual evidence is deceptive or blind, while the body identifies the cause.
    \item \textbf{T4, proprioceptive fine structure:} the cause is expressed mainly in temporal or joint-wise dynamics, with little useful terrain-visual signal.
    \item \textbf{T5, nominal or mild:} an anomaly-like signal is benign enough that unnecessary intervention can be worse than continuing.
\end{itemize}

The load-bearing T2 pair is O4 versus O2.  O2 is compliant terrain such as mud: the correct recovery is to change gait or constraints and cross.  O4 is adhesion or tethering: the correct recovery is to backstep and route around.  The pair is configured so that, at the decision interface, the proprioceptive binding window and action/torque trace are byte-identical.  The two scenes remain visually separable.  This creates a constructive impossibility for pure proprioception rather than an ordinary hard classification example.

T3 provides the mirror image.  O7 changes the visual interpretation without changing the underlying contact truth: safe-looking terrain may be low-friction, or a hazard-looking decal may be nominal.  O8 introduces a visually hidden contact obstacle.  These cases are designed so that image evidence is misleading or blind, and the decisive signal is the robot's dynamics.  Together, T2 and T3 test whether an attributor can choose the relevant evidence source rather than use a fixed modality priority.
```

Then replace `paper/main.tex:143-147` if any duplicate old snapshot text remains after the previous replacement with:

```latex
The primary evaluation unit is a frozen failure snapshot rather than a full navigation episode.  A privileged oracle trigger marks the failure onset from simulator state, after which the same multimodal packet is written to disk for every agent.  This protocol has four advantages: all agents consume exactly the same observations, paired tests compare predictions on the same scenes, ground truth is independent of the evaluated agent, and detector operating point is separated from attribution.

Closed-loop quantities are used only when a deterministic reset certificate is available.  The reset protocol reinitializes robot state, operator state, contact buffers, and scenario geometry so that repeated runs of the same lane match across order permutations.  Any closed-loop or interventional table carries that certificate; otherwise the result is reported only as a snapshot-level attribution measurement.

Each corpus item stores the rendered visual frames, depth when used, the proprioceptive binding window, the operator identity, taxonomy cell, appearance identifier, seed, admissible recovery set, and content hash.  Train/test appearance splits are pre-registered.  Multiple appearances per semantic class prevent success on O4/O2 from being explained as a single color-to-action rule.
```

- [ ] **Step 4: Tighten attributor, consequence, and abstention prose**

Replace `paper/main.tex:152-190` with:

```latex
The positive model is a VLA-style attributor with three evidence routes.  The visual route consumes the RGB-D context through the pretrained vision-language backbone.  The proprioceptive route summarizes the high-rate binding window either as text statistics or as latent \kinotokens{}.  The task route supplies the recovery registry, prior decisions, and parser schema.  The model emits a structured answer containing an attributed cause and one recovery primitive.

\kinotokens{} are latent proprioceptive summaries injected into the language-model embedding stream.  A small temporal encoder maps the observable window to a fixed set of soft tokens.  During training, auxiliary heads predict privileged physical parameters such as friction, payload, effort scale, and support/contact ratios; at test time these privileged targets are not available, but the prediction residual becomes an OOD-$\theta$ signal for abstention.  A text-route ablation serializes comparable proprioceptive summaries into language, including scalar and binned variants.  This design allows A7 to test whether body evidence must enter the planner, whether latent and text routes differ in bandwidth and residual grounding, and whether privileged distillation improves boundary calibration.

The baseline roster probes the evidence routes.  B1 is a strong pure-proprioceptive attributor; it should succeed where the body is sufficient and fail on O4/O2 by construction.  B-V masks the proprioceptive route and tests vision-only behavior.  B-T uses text-form proprioceptive injection.  B-F is a closed-set generic fusion classifier.  B5-unshaped receives multimodal input but lacks conflict-specific supervision.  B5-conflict is trained on the O4/O2 vision-true conflict, and B5-conflict-bi includes both vision-true and proprioception-true conflict directions.  The important comparison is the block structure across cells, not a single leaderboard rank.

\subsection{Recovery registry and structured outputs}
\label{subsec:registry}

Attribution labels are tied to recoveries through a pre-registered registry.  The primitive vocabulary includes \texttt{continue}, \texttt{backstep\_detour}, \texttt{high\_step}, \texttt{slow\_low}, \texttt{crawl}, \texttt{hold\_request}, and \texttt{detour\_replan}.  These names correspond to executable recovery controllers or conservative defaults rather than free-form language.  All model outputs pass through a schema-validating parser that admits only registered labels and primitives, records malformed outputs separately, and prevents unsupported commands from entering consequence measurements.

\subsection{Interventional consequence matrix}
\label{subsec:consequence-method}

Frozen-snapshot attribution establishes whether a model names the right cause.  It does not by itself show that the label matters physically.  We therefore define a forced-label intervention matrix
\begin{equation}
    M(s,\ell) = \big( y(s,\ell),\; C(s,\ell) \big),
\end{equation}
where the scene $s$ is fixed, the attributed label $\ell$ is forced, the registry maps $\ell$ to its canonical recovery, and the simulator records outcome $y$ and cost $C$.  Because only the label is intervened on, $M$ measures the causal role of explanation in recovery selection.

The cost function is scenario-registered before evaluation.  It penalizes falls, immobilization, unsafe energy release, failure to escape the hazard, unnecessary intervention on nominal rows, and time or path inefficiency.  From $M$ we compute expected recovery score and regret for an attributor,
\begin{equation}
    \mathrm{Regret}(f_\phi) = \mathbb{E}_{s}\left[C(s,\hat{\ell}) - \min_{\ell'\in\mathcal{L}} C(s,\ell')\right].
\end{equation}
This converts semantic mistakes into physical units.  It also yields a safe-default rule: when posterior mass is spread across labels with asymmetric costs, the expected-cost-minimizing primitive can be conservative even if the most likely label is not catastrophic.

\subsection{Abstention and boundary calibration}
\label{subsec:abstention-method}

Some failures should not be forced into a confident known label.  The method therefore includes abstention as a first-class recovery decision.  An attributor may abstain when the OOD-$\theta$ residual is high, when appearance or operator family is held out, or when the scene lies near a physical severity boundary.  Abstention maps to a conservative default such as \texttt{hold\_request} or a registry-defined safe action, and is evaluated by risk--coverage using the same cost matrix $M$.

The O10 effort-decay axis supplies a boundary calibration example.  A privileged sweep locates a physical threshold $\theta^\star$ at which the base controller transitions from successful traversal to failure.  The learned model is not trained to know $\theta^\star$ directly.  Instead, we test whether the residual grows near and beyond the boundary and whether abstention reduces expected cost.  O2 mild-compliance rows provide the complementary point: even when a proprio-only detector fires, vision can indicate benign mud and support continuing, so the decision of \emph{when} to intervene is itself cross-modal.

\subsection{Scope of the method}
\label{subsec:scope-method}

The locomotion policy, simulator, trigger infrastructure, and safety wrappers are fixed substrate for the study.  Paper-A does not claim a new SLAM system, a new locomotion controller, or a new formal safety filter.  Those components make the attribution experiment executable.  The methodological contribution is the isolation of cross-modal failure attribution through matched operators, frozen snapshots, structured recovery labels, interventional consequence, and abstention scored in physical-cost units.
```

- [ ] **Step 5: Run a LaTeX syntax smoke check**

Run:

```bash
cd /home/eureka/KinoVLA/paper && pdflatex -interaction=nonstopmode main.tex
```

Expected: command completes without a fatal LaTeX error. Warnings about references may remain until the final full rebuild.

---

### Task 3: Reorganize Experiments and Results around reader-facing questions

**Files:**
- Modify: `paper/main.tex:193-353`

**Interfaces:**
- Consumes: Method definitions from Task 2.
- Produces: Problem-first empirical narrative used by Discussion and Conclusion.

- [ ] **Step 1: Replace the Experiments opening and protocol figure/table framing**

Replace `paper/main.tex:196-240` with:

```latex
The experiments are organized around four robotics questions.  Can one sensing modality explain every recovery-relevant failure?  Does giving a model both channels make it resolve conflicts?  Do wrong explanations change physical recovery?  When should the robot abstain instead of forcing a known label?  A final ablation group asks which design choices carry these behaviors.  Fig.~\ref{fig:protocol} summarizes the evaluation loop, and Table~\ref{tab:question-map} maps each question to its measurement.

\begin{figure*}[t]
\centering
\setlength{\fboxsep}{5pt}
\begin{minipage}{0.98\textwidth}
\centering
\fbox{\begin{minipage}{0.18\textwidth}\centering\textbf{1. Failure event}\\Go2 enters an operator region; oracle trigger freezes the decision packet.\end{minipage}}
\hfill
\fbox{\begin{minipage}{0.18\textwidth}\centering\textbf{2. Frozen snapshot}\\RGB-D, appearance ID, $\mathrm{obs}_{48}{+}\tau_{12}$ binding window, seed, label.\end{minipage}}
\hfill
\fbox{\begin{minipage}{0.18\textwidth}\centering\textbf{3. Attribution}\\B1, B-V, B-F, B-T, and B5 variants predict cause and primitive.\end{minipage}}
\hfill
\fbox{\begin{minipage}{0.18\textwidth}\centering\textbf{4. Label intervention}\\Force $\ell$ through the registry while holding the same scene fixed.\end{minipage}}
\hfill
\fbox{\begin{minipage}{0.18\textwidth}\centering\textbf{5. Consequence}\\Measure success, cost, regret, and safe-default value.\end{minipage}}
\end{minipage}
\caption{\textbf{Evaluation loop.}  Frozen snapshots make attribution comparisons paired and repeatable.  Forced-label interventions then ask what physical recovery follows from each explanation while holding the scene fixed.}
\label{fig:protocol}
\end{figure*}

\begin{table*}[t]
\centering
\caption{Experimental questions and measurements.  The main text reports distilled results; artifact hashes and reproduction commands are listed in the appendix and experiment reports.}
\label{tab:question-map}
\scriptsize
\begin{tabular}{p{0.23\textwidth} p{0.24\textwidth} p{0.18\textwidth} p{0.25\textwidth}}
\toprule
Question & Measurement & Experiments & Main readout \\
\midrule
Can one modality explain every failure? & Matched and mirrored conflict operators & A1, A3 & O4/O2 blocks pure proprioception; O7/O8 blocks vision-only. \\
Does multimodal input imply conflict resolution? & Baseline and conflict-trained attributor comparisons & A2, A3, A7 & Unshaped or single-route models fail complementary cells; conflict supervision changes the pattern. \\
Do wrong explanations change recovery? & Forced-label consequence matrix & A4 & Canonical labels succeed on crux cells while off-label interventions fail and incur asymmetric cost. \\
When should the robot abstain? & OOD residuals, risk--coverage, and severity boundaries & A5, A6 & Abstention lowers expected cost and tracks known physical boundaries. \\
Which design choices matter? & Latent/text route, conflict dose, truth filter, encoder grid & A7 & Body evidence, conflict data, truth filtering, and privileged grounding each explain part of the behavior. \\
\bottomrule
\end{tabular}
\end{table*}

\subsection{Protocol, artifacts, and statistics}
\label{subsec:exp-protocol}

All headline numbers use the real Paper-A stack: Isaac physical Go2 simulation, live RTX rendering, real CLIP features where vision certificates are reported, trained VLA/monitor components where agents are evaluated, and real ApiOracle annotations for hindsight-rationale filtering.  CPU surrogates and synthetic-render smoke tests are excluded from claim evidence.  Closed-loop and consequence results use the A0.1 deterministic reset certificate: \texttt{deep\_reset} with fixed geometry gives cross-order max $|\Delta|=0.0$, gold-match $7/7$, and same-operator C2ST $0.5$; controls fail with naive reset $\Delta=19.8$ and deep-reset without fixed order $\Delta=10.6$.  The frozen A0 corpus contains 548 snapshots, and the A3 T3 extension adds 144 snapshots, giving 692 total real frozen snapshots for later ablations.

Proportions are reported with Wilson 95\% intervals.  Paired agent comparisons on shared snapshots use McNemar tests.  C2ST certificates use lane-grouped bootstrap or permutation tests and must pass the same-operator validity gate.  Every quantitative table is tied to a config hash, seed list, and commit in the experiment artifacts; hand-edited tables are not used for claim evidence.
```

- [ ] **Step 2: Rename and revise the main results table**

Replace `paper/main.tex:242-263` with:

```latex
\begin{table*}[t]
\centering
\caption{Key empirical findings.  CIs are Wilson intervals where reported by the experiment artifact.}
\label{tab:main-results}
\scriptsize
\begin{tabular}{p{0.20\textwidth} p{0.16\textwidth} p{0.20\textwidth} p{0.34\textwidth}}
\toprule
Phenomenon & Metric & Result & Takeaway \\
\midrule
Deterministic consequence evaluation & reset certificate & max $|\Delta|=0.0$; same-op C2ST $0.5$ & Closed-loop and consequence numbers are not order-residue artifacts. \\
Proprioceptive non-identifiability & O4/O2 proprio difference & max $|\Delta(\mathrm{obs}_{48}{+}\tau)|=0.0$ & Adhesion and compliant terrain can be identical at the body interface. \\
Visual separability of the matched pair & real-CLIP separation & $1.0$ & The same pair remains visually distinguishable. \\
Mirror conflict direction & B5-conflict-bi T3 & $0.917$--$0.92$ & Proprioception-true conflicts require learning the reverse trust direction. \\
Conflict attribution & B1 vs B5-conflict & $0.50 \rightarrow 0.90$; McNemar $p<10^{-5}$ & Conflict supervision improves over strong pure proprioception and generic multimodal baselines. \\
Physical consequence of labels & canonical vs off-label & $1.00$ $[0.72,1.00]$ vs $0.00$ $[0.00,0.28]$ on crux cells & The explanation selected by the robot changes recovery outcome. \\
Asymmetric recovery cost & safe default & $p^\star(\mathrm{adhesion})=0.25$ & A conservative primitive can minimize expected cost under label uncertainty. \\
Selective prediction & risk--coverage & $1.63$ vs always $2.80$ and never $1.71$ & Calibrated abstention lowers expected physical cost. \\
Physical severity boundary & O10 $\theta^\star$ & $0.2754$ & Residuals track a boundary in the underlying dynamics. \\
Rationale grounding & truth filter & grounded-correct $0.604\rightarrow0.765$ & Real ApiOracle filtering improves physical consistency of rationales. \\
\bottomrule
\end{tabular}
\end{table*}
```

- [ ] **Step 3: Replace C1/C5 and C2 subsections with question-oriented subsections**

Replace `paper/main.tex:265-278` with:

```latex
\subsection{Can one modality explain every failure?}
\label{subsec:results-modality}

The O4/O2 matched pair is the cleanest case where the body alone is insufficient.  O4 adhesion and O2 compliant terrain are placed so that the consumed proprioceptive binding representation is byte-identical: max $|\Delta(\mathrm{obs}_{48}{+}\tau)|=0.0$.  The same-operator C2ST validity gate is at chance ($0.5$), while real-CLIP visual separation is $1.0$.  A pure-proprioceptive attributor therefore has no information with which to separate the two labels at that interface.  B1's failure on this pair is a fair-opponent result because B1 remains strong on proprio-sufficient cases.

The mirror direction appears in T3.  In O7 visual-remap and O8 invisible-obstacle cases, the image is misleading or blind while the body signal is decisive.  A vision-only baseline fails this block (O7 $0.19$, O8 $0.00$ in the A3 slices), while B1 is complementary on the O8 proprioceptive case and B5-conflict-bi reaches $0.917$--$0.92$ T3 accuracy across the bidirectional battery.  These two directions rule out a fixed answer such as always trust vision or always trust proprioception.

\subsection{Does multimodal input imply conflict resolution?}
\label{subsec:results-conflict}

A2 tests whether access to both modalities is enough.  On the balanced O4/O2 conflict corpus, B1 is at $0.50$; generic multimodal or single-route baselines such as B-F, B-T, B-V, and B5-unshaped cluster at $0.80$; and the conflict-trained B5-conflict reaches $0.90$.  The comparison against the adjacent row is paired and significant (McNemar $b=0,c=24,p<10^{-5}$), and the three evaluated seeds all reproduce the $0.900$ headline.  The conservative reading is that conflict supervision is load-bearing beyond merely exposing a model to an image and a body channel.

A3 asks whether the learned behavior is evidence arbitration or a shortcut such as ``trust vision in conflicts.''  B5-conflict-bi, trained with both conflict directions, solves the T3 block at $0.917$ while the original vision-true specialist does not become a universal model.  The bidirectional battery therefore turns a scalar accuracy result into a behavioral pattern: competence depends on learning which evidence source is reliable in the current failure family.
```

- [ ] **Step 4: Replace the result-story figure caption and C3/C4/A7/Summary prose**

Replace `paper/main.tex:279-353` with:

```latex
\begin{figure*}[t]
\centering
\setlength{\fboxsep}{5pt}
\begin{minipage}{0.98\textwidth}
\fbox{\begin{minipage}{0.30\textwidth}\centering
\textbf{Modality ambiguity}\\
O4/O2: max $|\Delta|=0.0$ in proprioception; CLIP vision $=1.0$.\\
O7/O8: vision misleading/blind; B5-bi T3 $\approx0.92$.
\end{minipage}}
\hfill
\fbox{\begin{minipage}{0.30\textwidth}\centering
\textbf{Evidence arbitration}\\
A2 headline: B1 $0.50$ $<$ generic rows $0.80$ $<$ B5-conflict $0.90$.\\
Dose knee: best mean O4 attr at 10 matched samples.
\end{minipage}}
\hfill
\fbox{\begin{minipage}{0.30\textwidth}\centering
\textbf{Recovery consequence}\\
A4 crux cells: canonical $1.00$ vs off-label $0.00$.\\
Risk--coverage: abstention cost $1.63$ vs always $2.80$.
\end{minipage}}
\end{minipage}
\caption{\textbf{Empirical story.}  The experiments move from sensing ambiguity, to learned evidence arbitration, to physical recovery consequence and abstention.  Full numeric summaries are in Tables~\ref{tab:main-results} and~\ref{tab:ablation-results}.}
\label{fig:result-story}
\end{figure*}

\subsection{Do wrong explanations change physical recovery?}
\label{subsec:results-consequence}

A4 turns attribution into a causal intervention.  The same scene $s$ is held fixed while the forced label $\ell$ changes the recovery selected by the registry.  The matrix covers 630 episodes: 9 scenarios, 7 labels, and $N=10$ deterministic lanes per cell.  On crux cells, canonical labels achieve success $1.00$ $[0.72,1.00]$, while off-label interventions are $0.00$ $[0.00,0.28]$.  The label is therefore not a cosmetic explanation; it selects a physically different recovery.

The costs are asymmetric.  In T2, treating adhesion as compliant terrain can immobilize or catapult the robot, whereas treating compliant terrain as adhesion is often slow but safe.  This yields a safe-default crossover at $p^\star(\mathrm{adhesion})=0.25$: once adhesion has non-negligible posterior probability, the expected-cost-minimizing action becomes conservative.  Composing agent predictions through the matrix converts snapshot attribution into physical units.  The conflict agent's projected regret is $0.11$, compared with $0.44$ for baseline alternatives, and A7's conflict-dose predictions inherit the same regret interpretation.

\subsection{When should the robot abstain?}
\label{subsec:results-abstention}

A5 tests whether the attribution behavior survives beyond the exact appearance and operator seen during training.  Appearance OOD degradation scales with modality reliance: the vision-only row drops the most ($-0.21$), while the pure-proprioceptive row drops least ($-0.03$).  Composition tests pass $16/16$ on the registered combinations.  Leave-one-out O5 is rescued by the OOD-$\theta$ residual (16.09), which triggers abstain/Hold and yields 100\% safe outcome under the registered criterion.

Selective prediction converts uncertainty into a recovery decision.  The deployed residual has attribution-error AUROC $0.871$ over the 692-snapshot merged corpus, and the A5 risk--coverage curve reaches expected cost $1.633$, lower than always intervening ($2.796$) and lower than never intervening ($1.714$).  A6 supplies a physical boundary: the privileged O10 sweep locates $\theta^\star=0.2754$, with reach $1.0$ at effort floors $0.4$ and $0.3$ but $0.0$ at $0.25$, $0.2$, and $0.15$.  The OOD-$\theta$ residual rises across this boundary (reported range $0.629\rightarrow0.949$ and crossing near $0.85$ in the artifact).  We do not claim a literal policy flip at $\theta^\star$; the supported result is that the residual tracks the boundary and supports abstention calibration.

O2 and O8 mark useful boundaries of the current passive-snapshot setting.  Mild compliant terrain can trigger proprioceptive anomaly signatures, yet visual context can indicate benign mud and support continuing.  Thus on O2, even the decision of \emph{when} to intervene is cross-modal.  O8 likewise remains an active-probing boundary: the body reveals that contact is wrong, but a visually invisible obstacle can require probing or topology update beyond passive snapshot classification.

\subsection{Which design choices matter?}
\label{subsec:results-a7}

A7 explains which method choices carry the behavior.  Table~\ref{tab:ablation-results} reports the main ablations.  Vision-only remains near chance in the ambiguous regime ($0.458$ greedy, $0.392\pm0.062$ sampled), while proprioceptive injection through either text or latent routes reaches $1.000$ greedy ambiguous attribution.  The claim is therefore not that latent \kinotokens{} beat all text routes on greedy accuracy.  The supported claim is that latent \kinotokens{} match the strongest text route while adding a $\theta$ head, better sampled ambiguity behavior than scalar text, and lower prompt cost than binned text (1.27$\times$ fewer prompt tokens).

The conflict-dose curve is small-data and non-monotone.  Across three seeds, the best mean O4-conflict attribution is at dose 10 ($0.733$, range $0.600$--$0.800$), while doses 20 and 40 plateau at $0.667$ mean.  Through the A4 matrix, dose 10 is also the cleanest joint point because it ties for the lowest projected regret while maximizing mean O4 attribution.  Truth filtering improves grounded-correct rationales on the real ApiOracle kept+dropped stream from $0.604$ to $0.765$ over 413 kept and 110 rejected judged records, excluding 10 transport-error rows from the denominator.

The full encoder grid is available over real frozen binding windows: 18/18 variant $\times$ window $\times$ gate cells.  The best attribution cell is privileged distillation with $T=50$ and gate on ($0.805$).  The best $\theta$ MAE is privileged distillation with $T=25$ and gate on ($0.0279$).  Contrastive-only achieves the best residual-error ranking ($0.738$ AUROC) but has far worse $\theta$ MAE because it receives no privileged $\theta$ loss.  From-scratch supervised attribution ties the attribution ceiling on this offline split but does not recover the $\theta$ grounding; this narrows the method claim correctly.
```

- [ ] **Step 5: Keep the existing ablation table unless labels need minor consistency edits**

Inspect the table beginning at `\begin{table*}[t]` after `\label{subsec:results-a7}`. It can remain as written if the caption is still:

```latex
\caption{Method ablations from A7.  ``Greedy amb.'' is greedy attribution accuracy on the ambiguity regime; sampled values report mean and standard deviation where available.}
```

If the table was accidentally removed during Step 4, restore the original table from `paper/main.tex` lines 330--348 in the pre-rewrite file.

- [ ] **Step 6: Run a LaTeX syntax smoke check**

Run:

```bash
cd /home/eureka/KinoVLA/paper && pdflatex -interaction=nonstopmode main.tex
```

Expected: command completes without a fatal LaTeX error.

---

### Task 4: Rewrite Discussion, Conclusion, and appendix wording

**Files:**
- Modify: `paper/main.tex:356-410`

**Interfaces:**
- Consumes: Result narrative from Task 3.
- Produces: Final manuscript message and scoped limitations.

- [ ] **Step 1: Replace Discussion with robotics implications and limitations**

Replace `paper/main.tex:359-375` with:

```latex
\subsection{Recovery as explanation before intervention}

The experiments suggest a different view of legged recovery from the usual anomaly-detection pipeline.  Detecting that the robot is stuck, slipping, or overexerting is not enough when different causes require different primitives.  The O4/O2 and O7/O8 pairs show that the reliable evidence source can change by failure family.  The A4 intervention matrix then closes the loop: when the explanation changes, the executed recovery and physical outcome change as well.

This framing is compatible with robust locomotion and adaptation rather than a replacement for them.  A-class disturbances remain exactly where adaptive controllers should be strong.  Paper-A studies the residual decision where the response is discrete, semantic, or topological.  The result also does not imply that every VLA architecture is intrinsically better than a compact fusion model.  Any system that solves the same matched T2 and T3 cases, passes the same consequence matrix, and abstains at the same boundaries has learned the relevant operation.

\subsection{Implications for recovery systems}

Matched operators are useful because they reveal when a sensing channel is non-identifying, not merely noisy.  If two causes are identical at the consumed body interface, a stronger proprioceptive classifier cannot separate them without additional information.  Conversely, visual deception and invisible contact show why semantic appearance cannot be the only recovery signal.  This suggests that future recovery stacks should expose not only anomaly scores, but also the evidence basis for the proposed intervention.

The consequence matrix is equally important.  It converts semantic mistakes into physical cost and shows why safe defaults are asymmetric.  In the adhesion--mud pair, one wrong direction can be catastrophic while the other is slow but safe.  That asymmetry makes abstention and conservative recovery part of the policy design rather than an afterthought.

\subsection{Limitations}

All reported Paper-A evidence is from the Isaac physical-Go2 stack, not from outdoor hardware deployment.  A8 real-world snapshots remain optional future evidence.  The operators are constructed to isolate scientific questions, so they simplify real terrain physics even when they are physically motivated.  The forced-label matrix measures causal recovery consequence under registered scripted primitives; it does not claim that a fully autonomous navigation stack has been optimized end-to-end.

The detector is also not the contribution.  Oracle triggering is primary because the paper studies attribution given a failure event.  Hardened realistic-trigger checks are useful robustness evidence, but changing the trigger cannot replace the frozen-snapshot attribution and forced-label consequence protocol.  Finally, the safety wrappers and locomotion policy are fixed substrate.  Paper-A uses them to execute the attribution study; it does not introduce a new formal safety certificate or a new low-level locomotion controller.
```

- [ ] **Step 2: Replace Conclusion with concise problem-first conclusion**

Replace `paper/main.tex:381-383` with:

```latex
This paper studied safe quadrupedal recovery as cross-modal failure attribution.  The central result is that recovery cannot be reduced to anomaly detection followed by a fixed reflex.  Some failures are proprioceptively identical but visually separable; others are visually misleading or blind but proprioceptively separable.  A robot must therefore decide why it failed, which modality to trust, and whether uncertainty calls for abstention before selecting a recovery.

Matched operators and deterministic frozen snapshots isolate the information structure of this decision.  Forced-label interventions show that attribution errors change physical recovery outcome.  Residual-based abstention lowers expected cost near out-of-distribution operators and severity boundaries.  Together, these results support a problem-first principle for legged recovery: explain the failure before intervening.
```

- [ ] **Step 3: Revise appendix falsifier wording to avoid claim-ledger tone**

Replace `paper/main.tex:408-410` with:

```latex
The paper's conclusions remain falsifiable.  The modality-necessity result would fail if a same-interface proprioceptive discriminator separated O4/O2 while passing the same-operator gate, or if T3 images alone solved O7/O8.  The conflict-resolution result would fail if generic fusion or unshaped multimodal models matched conflict-trained agents across both T2 and T3.  The recovery-consequence result would fail if forced labels did not change the consequence matrix.  The abstention result would fail if residuals were near chance for attribution errors or did not reduce risk--coverage cost.  The fair-opponent interpretation would fail if B1 were weak on proprio-sufficient cells.  The current artifacts do not support these falsifiers.
```

- [ ] **Step 4: Search for old claim-ledger headings and wording**

Run:

```bash
grep -nE "C1|C2|C3|C4|C5|claim map|evidence block|support(s|ed)? the full chain|Serves" /home/eureka/KinoVLA/paper/main.tex
```

Expected: No visible main-text headings such as `C1/C5:` or table captions such as `Claims, evidence`. Acceptable matches include appendix falsifier wording only if kept minimal, or citations/labels unrelated to claim-ledger prose.

---

### Task 5: Final build, QA, and project record updates

**Files:**
- Modify: `paper/main.pdf` if generated by successful build
- Modify: `paper/main.bbl` if BibTeX output changes
- Modify: `CLAUDE.md`
- Optionally modify: `paper/README.md` only if the status text needs to mention the problem-first rewrite

**Interfaces:**
- Consumes: Complete rewritten manuscript from Tasks 1--4.
- Produces: Build-verified manuscript and accurate session record.

- [ ] **Step 1: Run the full LaTeX build**

Run:

```bash
cd /home/eureka/KinoVLA/paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

Expected: build exits 0 and regenerates `main.pdf`. Warnings are acceptable only if they are non-fatal and do not include unresolved references or missing citations.

- [ ] **Step 2: Check unresolved references and citations**

Run:

```bash
grep -nE "undefined references|Citation .* undefined|There were undefined|Rerun to get cross-references right" /home/eureka/KinoVLA/paper/main.log || true
```

Expected: no lines mentioning undefined references or undefined citations after the final build. A stale `Rerun to get cross-references right` line is not acceptable after the fourth build pass; rerun `pdflatex -interaction=nonstopmode main.tex` once if it appears.

- [ ] **Step 3: Check scope guardrails in the manuscript**

Run:

```bash
grep -nE "dominates all text|policy flip|O8 is solved|closed-loop benchmark|general navigation-system benchmark|replaces locomotion|new formal safety certificate" /home/eureka/KinoVLA/paper/main.tex || true
```

Expected: no overclaiming. Acceptable matches are explicit negations, such as `does not claim` or `not a new formal safety certificate`.

- [ ] **Step 4: Check that the visible result organization is problem-first**

Run:

```bash
grep -nE "Can one modality|Does multimodal input|Do wrong explanations|When should the robot abstain|Which design choices matter|Experimental questions" /home/eureka/KinoVLA/paper/main.tex
```

Expected: matches for the question-oriented result subsections and the experimental-question table.

- [ ] **Step 5: Update `CLAUDE.md` completed log**

In `CLAUDE.md`, add one line inside the §4 Completed Log block after the 2026-07-07 A7 line:

```text
2026-07-07 | Paper-A manuscript rewritten from claim-ledger/technical-report style into problem-first ICRA narrative while preserving A0--A7 evidence scope. | paper/main.tex; paper/main.pdf; docs/superpowers/specs/2026-07-07-problem-first-paper-rewrite-design.md
```

Do not change §2 experiment status unless the rewrite changed experiment completion state. Do not change §5 unless a live deviation or sign-off item changed.

- [ ] **Step 6: Inspect git diff summary**

Run:

```bash
cd /home/eureka/KinoVLA && git diff --stat
```

Expected: `paper/main.tex`, likely `paper/main.pdf`, the new spec and plan files, and `CLAUDE.md` appear. `Kino-vla-v2.md` must not appear.

- [ ] **Step 7: Final review pass for manuscript tone**

Read the rewritten `paper/main.tex` sections and confirm:

```text
- Abstract starts from the robotics recovery problem, not A0--A7.
- Introduction does not list the experiment battery as a proof plan.
- Experiment section is organized by reader-facing questions.
- Tables/captions no longer call themselves claim maps or evidence blocks.
- Discussion focuses on implications and limitations, not C1--C5 restatement.
- All numeric results remain unchanged from the approved evidence record.
```

If any item fails, revise `paper/main.tex` before reporting completion.

---

## Self-Review

**Spec coverage:** The plan covers abstract, introduction, related work, method, experiments/results, discussion, limitations, conclusion, appendix wording, build verification, overclaiming checks, and required `CLAUDE.md` session record update. The non-negotiable constraints from the spec are copied into Global Constraints.

**Placeholder scan:** The plan contains no `TBD`, `TODO`, `implement later`, or unspecified test steps. Each edit step includes concrete replacement text or an exact command and expected result.

**Type consistency:** Not applicable to executable code. LaTeX labels are consistent: `tab:question-map`, `fig:protocol`, `tab:main-results`, `fig:result-story`, and `tab:ablation-results` are referenced where defined or already exist.
