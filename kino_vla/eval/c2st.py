"""Classifier-two-sample test (C2ST) over temporal proprioceptive windows (E1).

E1 establishes the load-bearing claim of the recoverability dichotomy: that an ambiguity
pair (O4↔O2, O5↔O10, O3↔O1) is *proprioceptively indistinguishable* over the windowed
observation+history vector the robot consumes — so visual/semantic attribution is
necessary, not merely convenient. The proof is a classifier-two-sample test: train the
strongest practical discriminator to separate operator-A windows from operator-B windows;
if held-out ROC-AUC ≈ 0.5 (bootstrap CI covers 0.5, permutation p ≥ α) the two
distributions are indistinguishable to *any* proprioceptive function — the C2ST AUC
upper-bounds the separability of any such discriminator (including an RMA/DreamWaQ history
encoder, the strongest pure-proprioception baseline).

Crucially we match the JOINT distribution over the temporal window — not summary statistics
(cf. ``ambiguity.match_ambiguity_pair``, the spec-§8.1-P4 curve matcher, kept as a separate
artifact) — because a history encoder infers terrain from temporal *correlation structure*,
which marginal/summary matching does not constrain.

Two correctness points, both load-bearing for an honest result:

- **Lane-grouped resampling.** Windows overlap (stride 1) and all windows of one rollout
  lane share its operator label, so they are correlated. Bootstrap CIs and the permutation
  null resample whole *lanes*, never individual windows; naive per-window resampling yields
  anti-conservatively tight CIs / tiny p — a false "indistinguishable" verdict.
- **Held-out evaluation.** AUC is always measured on lanes the discriminator never trained
  on (a disjoint test seed-base, or a grouped hold-out split), so it is a genuine
  generalization number, not a memorization artifact.

Pure analysis: numpy for the statistic; torch (lazy-imported) only for the MLP / 1D-CNN
discriminators, so the module and the ``logreg`` path import on a torch-free machine. No
Isaac import.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from kino_vla.tokens.features import Standardizer

Predict = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------------------
# Statistic primitives (numpy only)
# --------------------------------------------------------------------------------------
def _rank_average(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (numpy-only ``scipy.stats.rankdata`` equivalent)."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    n = a.shape[0]
    r = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        r[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = r
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via the rank (Mann-Whitney U) statistic; NaN if a class is empty."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rank_average(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def covers_half(ci: tuple[float, float]) -> bool:
    """True iff the bootstrap CI brackets chance (0.5)."""
    return bool(ci[0] <= 0.5 <= ci[1])


# --------------------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------------------
def windows_from_lanes(
    lane_traces: Sequence[np.ndarray], length: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Slice each per-lane ``(S_i, F)`` trace into ``(length, F)`` windows.

    Returns ``(X, lane_ids)`` of shape ``(N, length, F)`` and ``(N,)``; a lane shorter than
    ``length`` contributes no windows. ``lane_ids`` indexes back into ``lane_traces`` so
    resampling can stay lane-grouped.
    """
    xs: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for li, trace in enumerate(lane_traces):
        trace = np.asarray(trace, dtype=np.float64)
        s = trace.shape[0]
        if s < length:
            continue
        starts = range(0, s - length + 1, max(1, stride))
        win = np.stack([trace[k : k + length] for k in starts], axis=0)
        xs.append(win)
        ids.append(np.full(win.shape[0], li, dtype=np.int64))
    if not xs:
        f = int(np.asarray(lane_traces[0]).shape[-1]) if len(lane_traces) else 0
        return np.empty((0, length, f), dtype=np.float64), np.empty((0,), dtype=np.int64)
    return np.concatenate(xs, axis=0), np.concatenate(ids, axis=0)


# --------------------------------------------------------------------------------------
# Discriminators — each returns a predict(raw (N,T,F)) -> scores closure (standardizes
# internally with train statistics). logreg = numpy; mlp/cnn1d = torch (lazy import).
# --------------------------------------------------------------------------------------
def _fit_standardizer(x_train: np.ndarray) -> Standardizer:
    return Standardizer.fit(x_train.reshape(-1, x_train.shape[-1]), eps=1e-6)


def _fit_logreg(
    x_train: np.ndarray, y_train: np.ndarray, *, seed: int, l2: float = 1e-3,
    lr: float = 0.5, epochs: int = 400,
) -> Predict:
    std = _fit_standardizer(x_train)
    n = x_train.shape[0]
    x = std.transform(x_train).reshape(n, -1)
    y = y_train.astype(np.float64)
    d = x.shape[1]
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(epochs):
        p = _sigmoid(x @ w + b)
        g = p - y
        w -= lr * (x.T @ g / n + l2 * w)
        b -= lr * float(g.mean())

    def predict(raw: np.ndarray) -> np.ndarray:
        xs = std.transform(raw).reshape(raw.shape[0], -1)
        return _sigmoid(xs @ w + b)

    return predict


def _fit_torch(
    x_train: np.ndarray, y_train: np.ndarray, kind: str, *, seed: int,
    epochs: int = 150, lr: float = 1e-3, weight_decay: float = 1e-4,
    batch_size: int = 128, hidden: int = 64,
) -> Predict:
    import torch  # lazy: keeps the module + logreg path torch-free
    from torch import nn

    torch.manual_seed(seed)
    device = torch.device("cpu")  # CPU for determinism in the gate
    std = _fit_standardizer(x_train)
    n, t, f = x_train.shape

    class _MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(t * f, hidden), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)

    class _TemporalCNN(nn.Module):
        """1D-CNN over (T, F); the power-matched analogue of an RMA history encoder
        (mirrors the MonitorNet conv stack, kernels 5/3, GELU, global-mean pool)."""

        def __init__(self) -> None:
            super().__init__()
            conv: list[nn.Module] = []
            in_ch = f
            for out_ch, k in ((48, 5), (48, 3)):
                conv += [nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2), nn.GELU()]
                in_ch = out_ch
            self.conv = nn.Sequential(*conv)
            self.head = nn.Sequential(
                nn.Linear(in_ch, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, C)
            return self.head(h.mean(dim=1)).squeeze(-1)

    net = (_TemporalCNN() if kind == "cnn1d" else _MLP()).to(device)
    xt = torch.tensor(std.transform(x_train), dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()

    @torch.no_grad()
    def predict(raw: np.ndarray) -> np.ndarray:
        net.eval()
        x = torch.tensor(std.transform(raw), dtype=torch.float32, device=device)
        return torch.sigmoid(net(x)).cpu().numpy()

    return predict


_FITTERS: dict[str, str] = {"logreg": "numpy", "mlp": "torch", "cnn1d": "torch"}


def _fit(x_train: np.ndarray, y_train: np.ndarray, classifier: str, *, seed: int) -> Predict:
    if classifier == "logreg":
        return _fit_logreg(x_train, y_train, seed=seed)
    if classifier in ("mlp", "cnn1d"):
        return _fit_torch(x_train, y_train, classifier, seed=seed)
    raise ValueError(f"unknown classifier {classifier!r} (logreg|mlp|cnn1d)")


# --------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class C2STResult:
    """One C2ST verdict. ``indistinguishable`` (CI covers 0.5 AND perm_p ≥ α) is the
    headline: True ⇒ the discriminator *cannot* separate the pair (proprioception is
    insufficient). For a control we WANT indistinguishable=False (AUC ≥ auc_high)."""

    auc: float
    auc_ci: tuple[float, float]
    perm_p: float
    n_perm: int
    classifier: str
    window_len: int
    feature_set: str
    n_per_class: tuple[int, int]
    feature_importance: dict[str, float]
    indistinguishable: bool
    alpha: float

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["auc_ci"] = [self.auc_ci[0], self.auc_ci[1]]
        d["n_per_class"] = [self.n_per_class[0], self.n_per_class[1]]
        return d


def _balance(
    xa: np.ndarray, xb: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample the larger class so the discriminator is not biased by class ratio."""
    m = min(xa.shape[0], xb.shape[0])
    if m == 0:
        return xa, xb
    ia = rng.choice(xa.shape[0], size=m, replace=False) if xa.shape[0] > m else np.arange(m)
    ib = rng.choice(xb.shape[0], size=m, replace=False) if xb.shape[0] > m else np.arange(m)
    return xa[ia], xb[ib]


def _lane_bootstrap_ci(
    scores: np.ndarray, labels: np.ndarray, lanes: np.ndarray, *, n_boot: int,
    rng: np.random.Generator, q: tuple[float, float] = (2.5, 97.5),
) -> tuple[float, float]:
    """Lane-grouped bootstrap CI: resample whole lanes (stratified by class) with replacement."""
    uniq = np.unique(lanes)
    cls = np.array([int(labels[lanes == ln][0]) for ln in uniq])
    pos_lanes, neg_lanes = uniq[cls == 1], uniq[cls == 0]
    by_lane = {int(ln): np.flatnonzero(lanes == ln) for ln in uniq}
    aucs: list[float] = []
    for _ in range(n_boot):
        chosen = np.concatenate(
            [
                rng.choice(pos_lanes, size=pos_lanes.shape[0], replace=True),
                rng.choice(neg_lanes, size=neg_lanes.shape[0], replace=True),
            ]
        )
        idx = np.concatenate([by_lane[int(ln)] for ln in chosen])
        a = roc_auc(labels[idx], scores[idx])
        if not np.isnan(a):
            aucs.append(a)
    if not aucs:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(aucs, q)
    return (float(lo), float(hi))


def _lane_perm_p(
    scores: np.ndarray, labels: np.ndarray, lanes: np.ndarray, observed: float, *,
    n_perm: int, rng: np.random.Generator,
) -> float:
    """Two-sided lane-grouped permutation p: shuffle the lane→label map (each lane is one
    class), recompute AUC on the held-out scores. p = P(|nullAUC−.5| ≥ |obsAUC−.5|)."""
    uniq = np.unique(lanes)
    cls = np.array([int(labels[lanes == ln][0]) for ln in uniq])
    by_lane = {int(ln): np.flatnonzero(lanes == ln) for ln in uniq}
    obs_dev = abs(observed - 0.5)
    count = 0
    perm_labels = np.empty_like(labels)
    for _ in range(n_perm):
        shuffled = rng.permutation(cls)
        for ln, c in zip(uniq, shuffled, strict=True):
            perm_labels[by_lane[int(ln)]] = c
        a = roc_auc(perm_labels, scores)
        if not np.isnan(a) and abs(a - 0.5) >= obs_dev:
            count += 1
    return (count + 1) / (n_perm + 1)


def _permutation_importance(
    predict: Predict, x_test: np.ndarray, labels: np.ndarray, base_auc: float,
    feature_names: Sequence[str] | None, rng: np.random.Generator,
) -> dict[str, float]:
    """Per-channel permutation importance on the held-out set: AUC drop when channel c is
    shuffled across windows+time. Names the leaking channel for the calibration loop."""
    f = x_test.shape[-1]
    names = list(feature_names) if feature_names is not None else [f"ch{c}" for c in range(f)]
    imp: dict[str, float] = {}
    flat_n = x_test.shape[0] * x_test.shape[1]
    for c in range(f):
        xp = x_test.copy()
        col = xp[:, :, c].reshape(flat_n)
        xp[:, :, c] = col[rng.permutation(flat_n)].reshape(x_test.shape[0], x_test.shape[1])
        imp[names[c]] = base_auc - roc_auc(labels, predict(xp))
    return imp


def c2st(
    xa_train: np.ndarray,
    xb_train: np.ndarray,
    xa_test: np.ndarray,
    xb_test: np.ndarray,
    lane_a_test: np.ndarray,
    lane_b_test: np.ndarray,
    *,
    classifier: str = "cnn1d",
    n_boot: int = 1000,
    n_perm: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
    feature_names: Sequence[str] | None = None,
    feature_set: str = "",
    window_len: int = 0,
    importance: bool = True,
) -> C2STResult:
    """Classifier-two-sample test: fit ``classifier`` on the train windows, evaluate on the
    disjoint test windows. ``xa_*``/``xb_*`` are ``(N, T, F)`` for the two operators;
    ``lane_*_test`` give the test windows' lane ids (made globally unique internally) for
    lane-grouped resampling. Returns AUC + lane-grouped bootstrap CI + permutation p.
    """
    rng = np.random.default_rng(seed)
    xa_b, xb_b = _balance(xa_train, xb_train, rng)
    x_train = np.concatenate([xa_b, xb_b], axis=0)
    y_train = np.concatenate([np.zeros(xa_b.shape[0]), np.ones(xb_b.shape[0])]).astype(np.int64)
    predict = _fit(x_train, y_train, classifier, seed=seed)

    x_test = np.concatenate([xa_test, xb_test], axis=0)
    labels = np.concatenate(
        [np.zeros(xa_test.shape[0]), np.ones(xb_test.shape[0])]
    ).astype(np.int64)
    # Globally-unique lane ids: offset op-b's past op-a's max so a and b never collide,
    # robust to any (possibly overlapping / negative) input lane-id ranges.
    la, lb = np.asarray(lane_a_test), np.asarray(lane_b_test)
    off = int(la.max(initial=-1)) - int(lb.min(initial=0)) + 1
    lanes = np.concatenate([la, lb + off]).astype(np.int64)

    scores = predict(x_test)
    auc = roc_auc(labels, scores)
    ci = _lane_bootstrap_ci(scores, labels, lanes, n_boot=n_boot, rng=rng)
    p = _lane_perm_p(scores, labels, lanes, auc, n_perm=n_perm, rng=rng)
    imp = (
        _permutation_importance(predict, x_test, labels, auc, feature_names, rng)
        if importance
        else {}
    )
    return C2STResult(
        auc=auc,
        auc_ci=ci,
        perm_p=float(p),
        n_perm=n_perm,
        classifier=classifier,
        window_len=int(window_len or (x_test.shape[1] if x_test.ndim == 3 else 0)),
        feature_set=feature_set,
        n_per_class=(int(xa_test.shape[0]), int(xb_test.shape[0])),
        feature_importance=imp,
        indistinguishable=bool(covers_half(ci) and p >= alpha),
        alpha=alpha,
    )


def _split_lanes(
    n_lanes: int, test_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(n_lanes)
    n_test = max(1, int(round(test_frac * n_lanes)))
    return perm[n_test:], perm[:n_test]  # (train_lanes, test_lanes)


def c2st_holdout(
    xa: np.ndarray,
    xb: np.ndarray,
    lane_a: np.ndarray,
    lane_b: np.ndarray,
    *,
    test_frac: float = 0.3,
    seed: int = 0,
    **kw: object,
) -> C2STResult:
    """Convenience C2ST with an internal lane-grouped train/test split (for cases without a
    pre-collected disjoint test set, e.g. the vision control or unit tests)."""
    rng = np.random.default_rng(seed)
    lane_a = np.asarray(lane_a)
    lane_b = np.asarray(lane_b)
    ua, ub = np.unique(lane_a), np.unique(lane_b)
    tr_a, te_a = _split_lanes(ua.shape[0], test_frac, rng)
    tr_b, te_b = _split_lanes(ub.shape[0], test_frac, rng)
    mtr_a = np.isin(lane_a, ua[tr_a])
    mte_a = np.isin(lane_a, ua[te_a])
    mtr_b = np.isin(lane_b, ub[tr_b])
    mte_b = np.isin(lane_b, ub[te_b])
    return c2st(
        xa[mtr_a], xb[mtr_b], xa[mte_a], xb[mte_b], lane_a[mte_a], lane_b[mte_b],
        seed=seed, **kw,  # type: ignore[arg-type]
    )


def c2st_sweep(
    train_lanes_a: Sequence[np.ndarray],
    train_lanes_b: Sequence[np.ndarray],
    test_lanes_a: Sequence[np.ndarray],
    test_lanes_b: Sequence[np.ndarray],
    *,
    window_lens: Sequence[int] = (25, 50, 100),
    classifiers: Sequence[str] = ("mlp", "cnn1d"),
    stride: int = 1,
    feature_names: Sequence[str] | None = None,
    feature_set: str = "",
    **kw: object,
) -> dict[tuple[int, str], C2STResult]:
    """Run the C2ST at every (window length T × classifier). Each ``*_lanes_*`` is a list of
    per-lane ``(S_i, F)`` per-step traces; windows are re-sliced at each T. The
    indistinguishability claim must hold at ALL T (longer history is the RMA adversary)."""
    out: dict[tuple[int, str], C2STResult] = {}
    for t in window_lens:
        xa_tr, _ = windows_from_lanes(train_lanes_a, t, stride)
        xb_tr, _ = windows_from_lanes(train_lanes_b, t, stride)
        xa_te, la_te = windows_from_lanes(test_lanes_a, t, stride)
        xb_te, lb_te = windows_from_lanes(test_lanes_b, t, stride)
        for clf in classifiers:
            out[(t, clf)] = c2st(
                xa_tr, xb_tr, xa_te, xb_te, la_te, lb_te,
                classifier=clf, feature_names=feature_names, feature_set=feature_set,
                window_len=t, **kw,  # type: ignore[arg-type]
            )
    return out


def c2st_vision(emb_a: np.ndarray, emb_b: np.ndarray, **kw: object) -> C2STResult:
    """C2ST on CLIP appearance embeddings (the disambiguator control). Each ``(D,)``
    embedding is an independent sample (its own lane); a logistic discriminator over the
    raw embedding. For O4↔O2 this MUST be distinguishable (AUC ≥ auc_high) — the
    information proprioception lacks is present in vision."""
    a = np.asarray(emb_a, dtype=np.float64)[:, None, :]  # (Na, 1, D)
    b = np.asarray(emb_b, dtype=np.float64)[:, None, :]
    lane_a = np.arange(a.shape[0])
    lane_b = np.arange(b.shape[0])
    kw.setdefault("classifier", "logreg")
    kw.setdefault("feature_set", "clip")
    kw.setdefault("importance", False)
    return c2st_holdout(a, b, lane_a, lane_b, **kw)  # type: ignore[arg-type]
