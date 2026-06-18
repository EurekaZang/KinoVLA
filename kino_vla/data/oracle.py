"""Oracle annotation client (spec §10 PHASE 3).

PHASE 3 has an external Oracle LLM (the spec names Gemini 3.1 Pro) write a reflection chain
over the failure snapshot. This module provides:

- :func:`build_prompt` — the English prompt template encoding the spec's PHASE 3 constraints
  (one atomic primitive per round, the §5 primitive library, the category vocabulary, the 2D
  pixel-coordinate format for ``Replan_Waypoint``, the safety iron-rule, and "prefer physical
  evidence over deceptive appearance"). This is the paper artifact reviewed on any change
  (QA 5.2); all prompt text sent to the model is English.
- :class:`ApiOracle` — the deployment client: it builds the prompt and delegates the actual
  chat completion to an injected ``complete`` callable (wired to a real provider via
  ``from_config``; injected with a stub in tests). The Oracle's raw text is parsed + filtered
  downstream — the client itself stays a thin transport.
- :class:`ScriptedOracle` — a deterministic offline annotator (the CI / throughput path, like
  the M4/M5 surrogates for unavailable real dependencies). It perceives the snapshot
  (appearance + proprioception), emits the canonical recovery for the perceived category, and
  confabulates at a configured rate — so the pipeline and the truth-consistency filter can be
  exercised end-to-end without an API key, with a realistic, reproducible reject mix.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Protocol

import numpy as np

from kino_vla.data.schema import Snapshot
from kino_vla.data.taxonomy import FailureTaxonomy
from kino_vla.tokens.features import FEATURE_SCHEMA
from kino_vla.utils.config import Config

_F = {name: i for i, name in enumerate(FEATURE_SCHEMA)}

# Canonical parameters the ScriptedOracle emits per primitive (kept minimal but schema-valid).
_DEFAULT_PARAMS: dict[str, dict] = {
    "Backstep": {"distance_m": 0.5},
    "Replan_Waypoint": {"point_px": [480, 360]},
    "Switch_Gait": {"mode": "high_step"},
    "Adjust_Posture": {"body_height_m": 0.25, "pitch_deg": 0.0},
    "Set_Constraint": {"max_speed": 0.4, "stiffness": 0.5},
    "Update_Topology": {"region_xy": [0.0, 0.0], "radius_m": 0.6, "status": "untraversable"},
    "Hold_and_Request": {"reason": "actuator torque saturated; cannot proceed safely"},
}
# A crawl gait is the canonical limp for effort-decay (distinct from mud's high_step).
_GAIT_FOR_CATEGORY = {"effort_decay": "crawl", "compliant_terrain": "high_step"}

# Visual appearance class → the category a vision-led reader infers from it.
_VISION_CATEGORY: dict[str, str] = {
    "ice_sheet": "low_friction",
    "ice": "low_friction",
    "brown_mud": "compliant_terrain",
    "mud": "compliant_terrain",
    "yellow_adhesive": "adhesion",
    "adhesive": "adhesion",
}


class OracleClient(Protocol):
    """Anything that annotates a failure snapshot with a raw CoT string (spec §10 PHASE 3)."""

    def annotate(self, snapshot: Snapshot) -> str: ...


# ------------------------------------------------------------------- prompt template
def build_prompt(
    snapshot: Snapshot, cfg: Config, *, reveal_appearance: bool = True
) -> list[dict[str, str]]:
    """Build the English Oracle prompt (system + user) for one snapshot (spec §10 PHASE 3).

    ``reveal_appearance=False`` (the multimodal API path) omits the material-class name from the
    text so the Oracle must read the surface from the attached RGB frames and reason — naming the
    class would hand it the visual half of the attribution for free.

    Returns ``[{"role": "system", "content": ...}, {"role": "user", "content": ...}]``. A
    multimodal client attaches the snapshot's RGB/depth frames alongside the user message; the
    textual context here mirrors what the model is told. All text is English (paper artifact).
    """
    p = cfg.oracle.prompt
    primitives = list(p.primitive_library)
    categories = list(p.category_vocabulary)
    system = (
        "You are a privileged oracle annotating a quadruped robot's failure-recovery dataset.\n"
        "You are shown a multimodal snapshot captured the instant an on-board monitor detected "
        "a kinodynamic anomaly: the last 5 RGB frames, the last 5 depth frames, a 500 ms window "
        "of proprioceptive Kino-Tokens (motor-current/joint/IMU/contact-derived physical "
        "features), the robot's previous recovery outputs, and scene context.\n"
        "Reflect on the physical cause of the anomaly and choose exactly ONE recovery "
        "primitive.\n\n"
        "Output a SINGLE JSON object and nothing else, with this schema:\n"
        '{"thought": <string>, "attribution": <category>, '
        '"action": {"primitive": <name>, "params": <object>}}\n\n'
        f"'attribution' must be exactly one of: {', '.join(categories)}.\n"
        "Category meanings (the physical failure each names; infer which one from the evidence):\n"
        "- low_friction: a slippery surface (ice, oil) where the feet lose traction and slide.\n"
        "- compliant_terrain: soft, deformable ground (e.g. mud) the feet sink into and that "
        "resists forward motion.\n"
        "- region_collapse: the ground gave way underneath (e.g. thin ice breaking); a sudden "
        "loss of support, not merely a slippery patch.\n"
        "- adhesion: a sticky or elastic surface (e.g. a glue board or a tether) that grips or "
        "pulls the body back.\n"
        "- overload: excessive EXTERNAL weight that crouches the trunk and bogs the motion (the "
        "healthy motors strain under the load but the feet keep their grip).\n"
        "- effort_decay: the robot's OWN actuators weaken (overheating) so they hit their reduced "
        "torque cap, the gait sags, and the feet start to slip.\n"
        "- external_push: a sudden external impulse/shove.\n"
        "- invisible_obstacle: an unseen rigid barrier blocks the path.\n"
        "- high_centering: the body is beached on a ridge with the feet partly unloaded.\n"
        "- obs_bias: a sensor/observation bias, not a real terrain interaction.\n"
        "- nominal: no physical failure.\n"
        "Calibration examples (reason about THIS snapshot on its own; do not copy blindly):\n"
        "- slip sustained high, no effort, plain surface => low_friction => Set_Constraint.\n"
        "- slip steps up partway through the window (low then high) => region_collapse => "
        "Update_Topology.\n"
        "- low slip, sustained tracking deficit, brown soft-looking surface => compliant_terrain "
        "=> Switch_Gait(high_step).\n"
        "- low slip, a brief resistance then it eases, a yellow/sticky-looking surface => "
        "adhesion => Backstep.\n"
        "- the trunk height SAGS (a crouch) with the feet SLIPPING and the effort trace SPIKING "
        "toward saturation => effort_decay (the actuators are capped and can no longer hold the "
        "gait) => Switch_Gait(crawl).\n"
        "- the trunk height SAGS (a crouch) but the feet keep GRIP (low slip) and effort stays "
        "~0, with a speed deficit on plain ground => overload (a heavy external load the healthy "
        "motors strain under) => Hold_and_Request.\n"
        "- a speed deficit with NO slip, NO effort, and trunk height NORMAL (no sag) => "
        "invisible_obstacle => Update_Topology.\n"
        f"'action.primitive' must be exactly one of: {', '.join(primitives)} "
        "(one atomic action per round).\n\n"
        "Primitive parameter schema:\n"
        "- Backstep: {distance_m: number > 0}\n"
        "- Replan_Waypoint: {point_px: [u, v]}  (a 2D PIXEL coordinate in the RGB frame; it is "
        "depth-unprojected to a 3D waypoint downstream)\n"
        "- Switch_Gait: {mode: high_step | trot | crawl}\n"
        "- Adjust_Posture: {body_height_m: number > 0, pitch_deg: number}\n"
        "- Set_Constraint: {max_speed: number > 0, stiffness: number}\n"
        "- Update_Topology: {region_xy: [x, y], radius_m: number > 0, status: string}\n"
        "- Hold_and_Request: {reason: string}\n\n"
        "Safety iron-rule: on a sudden trap (elastic entanglement/adhesion, or a collapsing "
        "surface) prioritize escaping first (Backstep) and marking the region "
        "(Update_Topology); do NOT replan a detour (Replan_Waypoint) in the same round.\n"
        "Integrate BOTH the visible material and the proprioception. The proprioception is given "
        "as a short per-channel TIME-SERIES (a few bins, oldest to newest): read its SHAPE and "
        "sustained level, where a single-bin spike is a gait artifact but a sustained level or a "
        "monotone step/ramp is a real signature. A SAG in the 'base_height_trace' (the trunk "
        "crouching to a lower level) means the legs are overwhelmed: by a heavy external load "
        "(overload) when the feet still GRIP (low slip) and the 'effort_trace' stays ~0, or by "
        "weakened actuators (effort_decay) when the feet SLIP and the effort trace SPIKES toward "
        "its cap. A give-way (region_collapse) instead leaves the trunk at NORMAL height and the "
        "motors UNLOADED (effort ~0) while the feet slip in a STEP (low then high). "
        "Appearance can be deceptive (a reflective ice sheet can look like solid ground), so when "
        "the sustained physics and the appearance conflict, trust the physics; but when the "
        "physics is ambiguous (e.g. a velocity deficit with low slip that could be mud or an "
        "adhesive board), use the visible material to decide."
    )
    proprio = _proprio_summary(snapshot)
    appearance_line = (
        f"- visible surface appearance ahead: {snapshot.appearance_class}\n"
        if reveal_appearance
        else "- the RGB frames of the surface ahead are attached (most recent last); read it\n"
    )
    user = (
        "Failure snapshot:\n"
        f"- monitor channel that fired: {snapshot.monitor_channel}\n"
        f"{appearance_line}"
        f"- proprioceptive window time-series (bins oldest->newest): {json.dumps(proprio)}\n"
        f"- robot pose (odometry x, y): [{snapshot.pose_xy[0]:.2f}, {snapshot.pose_xy[1]:.2f}]\n"
        f"- previous recovery outputs this episode: {snapshot.prior_outputs or 'none'}\n"
        "Attribute the physical cause and choose one recovery primitive as JSON."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _proprio_summary(snapshot: Snapshot) -> dict[str, object]:
    """Proprioceptive window as a short TIME-SERIES (Method A: richer, less-lossy conditioning).

    A 4-scalar mean discards the temporal SHAPE — which is exactly what separates the proprio-
    matched pairs: O3 thin-ice shows a slip STEP (intact -> collapsed) where O1 uniform ice is
    flat-high; both O10 effort-decay and O5 overload CROUCH the trunk (a base_height sag), split by
    the feet+effort — O10 SLIPS with the effort trace spiking to its derated cap, O5 keeps grip
    (low slip) with effort ~0. So we report each channel down-sampled to a few time-bins (oldest ->
    newest): the Oracle reads the shape itself (we do NOT tell it which shape means which category).
    A one-bin blip is a gait artifact; a sustained level or a monotone step is a real signature.
    base_height is the crouch discriminator — Observable Sport-Client signals only, never truth.
    """
    w = snapshot.proprio_window
    if w.size == 0:
        return {}

    mean = w.mean(axis=0)

    def trace(name: str, bins: int = 8) -> list[float]:
        col = w[:, _F[name]]
        return [round(float(b.mean()), 2) for b in np.array_split(col, min(bins, len(col)))]

    # Give BOTH the time-series (shape) and the sustained mean (a stable anchor): the trace alone
    # reads "noisy" to the Oracle and it over-attributes obs_bias; the mean grounds it (Method A+).
    return {
        "slip_trace": trace("slip_ratio"),  # 0..1; sustained high ⇒ slipping; a step ⇒ a give-way
        "slip_mean": round(float(mean[_F["slip_ratio"]]), 3),
        "effort_trace": trace("effort_ratio"),  # actuator-torque saturation
        "effort_mean": round(float(mean[_F["effort_ratio"]]), 3),
        "tracking_trace": trace("tracking_err"),  # velocity deficit (resistance/blocking)
        "tracking_mean": round(float(mean[_F["tracking_err"]]), 3),
        # trunk height: a sustained DROP (a crouch) ⇒ the legs are overwhelmed (overload OR
        # effort_decay); normal ⇒ a give-way/obstacle. Pairs with effort+slip to split O5↔O10.
        "base_height_trace": trace("base_height"),
        "base_height_mean": round(float(mean[_F["base_height"]]), 3),
        "tilt_peak": round(float(w[:, _F["tilt"]].max()), 3),
    }


# ------------------------------------------------------------------- external-LLM client
class ApiOracle:
    """External-LLM Oracle (spec §10 PHASE 3): build the English prompt, delegate completion.

    ``complete`` maps the prompt messages to the model's raw text reply; ``from_config`` wires
    it to a real provider via an API key in the environment. Tests inject a stub ``complete``,
    so the client's logic (prompt build + transport contract) is fully covered without network.
    """

    def __init__(
        self,
        complete: Callable[[list[dict[str, str]], list], str],
        cfg: Config,
        *,
        multimodal: bool = True,
        n_images: int = 2,
        reveal_appearance: bool = False,
    ) -> None:
        self._complete = complete
        self._cfg = cfg
        self._multimodal = bool(multimodal)
        self._n_images = int(n_images)
        self._reveal = bool(reveal_appearance)

    def annotate(self, snapshot: Snapshot) -> str:
        messages = build_prompt(snapshot, self._cfg, reveal_appearance=self._reveal)
        images: list = []
        if self._multimodal and snapshot.rgb.size:
            images = [snapshot.rgb[i] for i in range(-min(self._n_images, len(snapshot.rgb)), 0)]
        return self._complete(messages, images)

    @classmethod
    def from_config(cls, cfg: Config) -> ApiOracle:
        """Wire ``complete`` to a real provider from env API keys (OpenAI, then Anthropic, Google).

        Not exercised in CI (no key / no network); raises a clear error if no provider is
        configured so a misconfigured deployment fails loudly rather than silently faking data.
        """
        api = cfg.oracle.get("api")
        if os.environ.get("OPENAI_API_KEY"):
            n_images = int(api.get("n_images", 2)) if api else 2
            return cls(_openai_responses_complete_factory(cfg), cfg, n_images=n_images)
        model = str(cfg.oracle.prompt.get("model", "gemini-3.1-pro"))
        if os.environ.get("ANTHROPIC_API_KEY"):
            return cls(_anthropic_complete_factory(), cfg)
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            return cls(_gemini_complete_factory(model), cfg)
        raise RuntimeError(
            "ApiOracle.from_config: no LLM provider configured. Set OPENAI_API_KEY (gateway, "
            "spec §10 Oracle) / ANTHROPIC_API_KEY / GOOGLE_API_KEY, or use ScriptedOracle offline."
        )


def _png_data_url(rgb: np.ndarray) -> str:
    """Encode an ``(H, W, 3)`` float-[0,1] frame to a base64 PNG data URL for vision input."""
    import base64
    import io

    arr = (np.clip(np.asarray(rgb), 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    try:
        from PIL import Image

        Image.fromarray(arr).save(buf, format="PNG")
    except ImportError:  # PIL absent — fall back to imageio (used elsewhere for video)
        import imageio.v2 as imageio

        imageio.imwrite(buf, arr, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _openai_responses_complete_factory(cfg: Config) -> Callable[[list[dict[str, str]], list], str]:
    """OpenAI-compatible **Responses API** client (the gateway in cfg.oracle.api), multimodal.

    System prompt → ``instructions``; the user text + the attached RGB frames → ``input``; honours
    the configured model, reasoning effort, ``store`` flag, and token budget. Small retry on
    429/5xx/timeout. Raw urllib (no SDK dependency); ``OPENAI_API_KEY`` from the env.
    """
    import time
    import urllib.error
    import urllib.request

    api = cfg.oracle.api
    url = str(api.base_url).rstrip("/") + str(api.path)
    model = str(api.model)
    store = bool(api.get("store", False))
    effort = api.get("reasoning_effort")
    max_out = api.get("max_output_tokens")  # optional — some gateways reject the param
    timeout = float(api.get("timeout_s", 180.0))
    retries = int(api.get("retries", 3))
    key = os.environ["OPENAI_API_KEY"]

    def complete(messages: list[dict[str, str]], images: list) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
        content: list[dict] = [{"type": "input_text", "text": user_text}]
        for img in images:
            content.append({"type": "input_image", "image_url": _png_data_url(img)})
        body: dict = {
            "model": model,
            "instructions": system,
            "input": [{"role": "user", "content": content}],
            "store": store,
        }
        if max_out:
            body["max_output_tokens"] = int(max_out)
        if effort:
            body["reasoning"] = {"effort": str(effort)}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        last = ""
        for attempt in range(retries):
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(), headers=headers, method="POST"
            )
            try:
                with opener.open(req, timeout=timeout) as r:
                    return _extract_output_text(json.load(r))
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace')}"
                if e.code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"Oracle API error: {last}") from e
            except Exception as e:  # noqa: BLE001 - network transport, retry then surface
                last = f"{type(e).__name__}: {e}"
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"Oracle API failed after {retries} attempts: {last}")

    return complete


def _extract_output_text(resp: dict) -> str:
    """Pull the assistant text out of a Responses-API reply (``output_text`` or message items)."""
    text = resp.get("output_text")
    if isinstance(text, str) and text.strip():
        return text
    parts = [
        c.get("text", "")
        for item in resp.get("output", [])
        if item.get("type") == "message"
        for c in item.get("content", [])
        if c.get("type") == "output_text"
    ]
    return "".join(parts)


def _anthropic_complete_factory() -> Callable[[list[dict[str, str]], list], str]:
    def complete(messages: list[dict[str, str]], images: list) -> str:  # noqa: ARG001
        import anthropic  # optional dep; only imported when a key is present

        client = anthropic.Anthropic()
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = [{"role": "user", "content": m["content"]} for m in messages if m["role"] == "user"]
        resp = client.messages.create(
            model="claude-opus-4-8", max_tokens=1024, system=system, messages=user
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    return complete


def _gemini_complete_factory(model: str) -> Callable[[list[dict[str, str]], list], str]:
    def complete(messages: list[dict[str, str]], images: list) -> str:  # noqa: ARG001
        from google import genai  # optional dep; only imported when a key is present

        client = genai.Client()
        prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text or ""

    return complete


# ------------------------------------------------------------------- offline surrogate
class ScriptedOracle:
    """Deterministic offline Oracle surrogate (the CI / throughput path).

    Perceives the snapshot (visual appearance first, proprioceptive symptom as fallback), emits
    the canonical recovery for the perceived category, and confabulates at ``confab_rate``. The
    perception is *honestly* wrong on deceptive scenes (a shuffled appearance) and on the
    proprio-only ambiguity pair (O5 overload vs O10 effort-decay share a torque-saturation
    symptom), which is where the truth-consistency filter earns its keep. Determinism: an
    internal RNG seeded at construction advances once per :meth:`annotate` call (the pipeline
    drives maze cells in a fixed, seeded order).
    """

    def __init__(self, cfg: Config, taxonomy: FailureTaxonomy, seed: int = 0) -> None:
        self._cfg = cfg
        self._tax = taxonomy
        self._confab = float(cfg.oracle.confab_rate)
        self._sibling = _sibling_categories(cfg, taxonomy)
        # Each operator's honest appearance (from the maze spec): the surrogate is fooled only
        # when the snapshot's appearance differs from it (a genuine physics-visual decoupling),
        # not when an honest-but-ambiguous appearance is shown (e.g. ice for O1 *and* O3 — there
        # it competently disambiguates from the Kino-Tokens, as a real all-modality oracle would).
        self._canonical_appearance: dict[str, str] = {
            str(c["op"]): str(c["appearance"]) for c in cfg.maze.cells
        }
        self._rng = np.random.default_rng(int(seed))

    def annotate(self, snapshot: Snapshot) -> str:
        """Annotate one snapshot (a controllable surrogate for the external Oracle LLM).

        It derives the true failure class from the snapshot (it is *generating* training data,
        not being evaluated — like the M4/M5 surrogates for unavailable real dependencies),
        then: follows the visual story when the appearance contradicts the physics (a decoupled
        / O7-style deception fools a vision-biased reader — a *natural* confabulation), else
        emits the correct attribution and confabulates at ``confab_rate`` (cycling the filter's
        distinct failure modes). The truth-consistency filter — the component under test —
        judges the result regardless of how it was produced. The real ``ApiOracle`` instead
        reflects from the prompt alone and is genuinely uncertain.
        """
        true_cat = self._tax.category_of(snapshot.operator_name)
        canonical_appr = self._canonical_appearance.get(snapshot.operator_name)
        is_decoupled = canonical_appr is not None and snapshot.appearance_class != canonical_appr
        vision_cat = _VISION_CATEGORY.get(snapshot.appearance_class)
        if is_decoupled and vision_cat is not None and vision_cat != true_cat:
            # Appearance was shuffled away from the physics (decoupled scene): a vision-biased
            # reader is fooled and follows the (wrong) visual story.
            category, primitive = vision_cat, self._tax_canonical(vision_cat)
        else:
            category, primitive = true_cat, self._tax_canonical(true_cat)
            if true_cat != "nominal" and self._rng.random() < self._confab:
                category, primitive = self._confabulate(true_cat)
        thought = (
            f"The {snapshot.monitor_channel} channel fired over surface "
            f"'{snapshot.appearance_class}'. I attribute this to {category} and recover with "
            f"{primitive}."
        )
        params = self._params_for(primitive, category)
        return json.dumps(
            {
                "thought": thought,
                "attribution": category,
                "action": {"primitive": primitive, "params": params},
            }
        )

    # ---------------------------------------------------------------- confabulation
    def _confabulate(self, category: str) -> tuple[str, str]:
        """Inject a plausible-but-wrong reflection, cycling the distinct filter failure modes."""
        modes = []
        sibling = self._sibling.get(category)
        if sibling is not None:
            modes.append("wrong_attribution")  # tell the sibling's story
            modes.append("wrong_primitive")  # right story, sibling's strategy
        if self._tax.is_sudden_trap(category):
            modes.append("unsafe_detour")  # same-round detour on a sudden trap
        if not modes:
            return category, self._tax_canonical(category)
        mode = modes[int(self._rng.integers(len(modes)))]
        if mode == "wrong_attribution":
            return sibling, self._tax_canonical(sibling)
        if mode == "wrong_primitive":
            return category, self._tax_canonical(sibling)
        return category, "Replan_Waypoint"

    # ---------------------------------------------------------------- helpers
    def _tax_canonical(self, category: str) -> str:
        gt = self._cfg.recovery.canonical
        return (
            str(gt.get(category, "Set_Constraint")) if category != "nominal" else "Set_Constraint"
        )

    def _params_for(self, primitive: str, category: str) -> dict:
        params = dict(_DEFAULT_PARAMS.get(primitive, {}))
        if primitive == "Switch_Gait":
            params["mode"] = _GAIT_FOR_CATEGORY.get(category, "high_step")
        return params


def _sibling_categories(cfg: Config, taxonomy: FailureTaxonomy) -> dict[str, str]:
    """Map each ambiguity-pair member's category to its sibling's category (for confabulation)."""
    op_cat = cfg.attribution.operator_category
    out: dict[str, str] = {}
    for pair in cfg.maze.ambiguity_pairs:
        a, b = str(pair[0]), str(pair[1])
        ca, cb = str(op_cat.get(a)), str(op_cat.get(b))
        if ca and cb and ca != cb:
            out[ca], out[cb] = cb, ca
    return out
