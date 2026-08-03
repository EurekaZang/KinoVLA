#!/usr/bin/env python3
"""Build one deterministic 11-operator × 3-view Kino-Fail gallery."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper/figures"
OUT_PNG = OUT_DIR / "kinofail_11_operator_multiview.png"
OUT_JPG = OUT_DIR / "kinofail_11_operator_multiview_preview.jpg"
OUT_MANIFEST = OUT_DIR / "kinofail_11_operator_multiview_manifest.json"

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def f(path: str) -> Path:
    return ROOT / path


CARDS = [
    (
        "O1  Low friction",
        "Surface friction changes while appearance is controlled",
        [
            ("outputs/kinofail_realistic/terrain_pbr_gate/01_life_train_tiles141_nominal.png", "Nominal physics"),
            ("outputs/kinofail_realistic/terrain_pbr_gate/01_life_train_tiles141_anomaly.png", "Low-friction pair"),
            ("outputs/kinofail_realistic/terrain_pbr_gate/03_wild_train_ground054_anomaly.png", "Cross-material view"),
        ],
    ),
    (
        "O2  Deformable terrain",
        "Contact-driven sinkage and resistance on disturbed soil",
        [
            ("outputs/kinofail_realistic/o2_o3_visual_evidence/O2_disturbed_soil_overview.png", "Overview"),
            ("outputs/kinofail_realistic/o2_o3_visual_evidence/O2_disturbed_soil_side_close.png", "Side close-up"),
            ("outputs/kinofail_realistic/o2_o3_visual_evidence/O2_disturbed_soil_top_oblique.png", "Top oblique"),
        ],
    ),
    (
        "O3  Triggered collapse",
        "Load-triggered fracture changes support topology",
        [
            ("outputs/kinofail_realistic/o2_o3_visual_evidence/O3_fractured_surface_overview.png", "Overview"),
            ("outputs/kinofail_realistic/o2_o3_visual_evidence/O3_fractured_surface_side_close.png", "Side close-up"),
            ("outputs/kinofail_realistic/o2_o3_visual_evidence/O3_fractured_surface_top_oblique.png", "Top oblique"),
        ],
    ),
    (
        "O4  Foot adhesion",
        "Foot attachment resists commanded locomotion",
        [
            ("outputs/demos/indoor_adhesion_icra/multiview_peak/01_room_overview_left.png", "Room overview"),
            ("outputs/demos/indoor_adhesion_icra/multiview_peak/03_left_side_low.png", "Side close-up"),
            ("outputs/demos/indoor_adhesion_icra/multiview_peak/07_top_down.png", "Top-down"),
        ],
    ),
    (
        "O5  Rigid payload",
        "Visible overload modifies mass, CoM, and load transfer",
        [
            ("outputs/kinofail_realistic/operator_gallery/source/O5/01_overview_left.png", "Overview"),
            ("outputs/kinofail_realistic/operator_gallery/source/O5/02_side_low.png", "Side close-up"),
            ("outputs/kinofail_realistic/operator_gallery/source/O5/03_front_oblique.png", "Front oblique"),
        ],
    ),
    (
        "O6  Lateral impulse",
        "Finite body-frame wrench perturbs the walking state",
        [
            ("outputs/kinofail_realistic/operator_gallery/source/O6/01_overview_left.png", "Overview"),
            ("outputs/kinofail_realistic/operator_gallery/source/O6/02_side_low.png", "Side close-up"),
            ("outputs/kinofail_realistic/operator_gallery/source/O6/03_front_oblique.png", "Front oblique"),
        ],
    ),
    (
        "O7  Visual-physics remap",
        "Appearance is decoupled from depth/geometry evidence",
        [
            ("outputs/kinofail_realistic/o7_rtx_depth_gate/seed_42_val_tiles139_nominal.png", "Nominal mapping"),
            ("outputs/kinofail_realistic/o7_rtx_depth_gate/seed_42_val_tiles139_severe.png", "Severe remap"),
            ("outputs/kinofail_realistic/o7_rtx_depth_gate/seed_59_test_concrete039_severe.png", "Cross-material view"),
        ],
    ),
    (
        "O8  Transparent obstacle",
        "High-transmission geometry blocks forward progress",
        [
            ("outputs/kinofail_realistic/operator_gallery/source/O8/01_overview_left.png", "Overview"),
            ("outputs/kinofail_realistic/operator_gallery/source/O8/02_side_low.png", "Side close-up"),
            ("outputs/kinofail_realistic/operator_gallery/source/O8/03_front_oblique.png", "Front oblique"),
        ],
    ),
    (
        "O9  High-centering",
        "Belly contact on a central runner removes foot authority",
        [
            ("outputs/kinofail_realistic/operator_gallery/source/O9/01_overview_left.png", "Overview"),
            ("outputs/kinofail_realistic/operator_gallery/source/O9/02_side_low.png", "Side close-up"),
            ("outputs/kinofail_realistic/operator_gallery/source/O9/03_front_oblique.png", "Front oblique"),
        ],
    ),
    (
        "O10  Actuator derating",
        "Reduced torque limits cause saturation and loss of mobility",
        [
            ("outputs/kinofail_realistic/operator_gallery/source/O10/01_overview_left.png", "Overview"),
            ("outputs/kinofail_realistic/operator_gallery/source/O10/02_side_low.png", "Side close-up"),
            ("outputs/kinofail_realistic/operator_gallery/source/O10/03_front_oblique.png", "Front oblique"),
        ],
    ),
    (
        "O11  Sensor bias",
        "Bias, random walk, and latency corrupt proprioception",
        [
            ("outputs/kinofail_realistic/operator_gallery/source/O11/01_overview_left.png", "Overview"),
            ("outputs/kinofail_realistic/operator_gallery/source/O11/02_side_low.png", "Side close-up"),
            ("outputs/kinofail_realistic/operator_gallery/source/O11/03_front_oblique.png", "Front oblique"),
        ],
    ),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for _, _, views in CARDS:
        for rel, _ in views:
            if not f(rel).is_file():
                raise FileNotFoundError(f(rel))

    # One operator per row keeps every view large.  Labels are placed directly on
    # the first view so there are no title bands, cards, captions, or dead zones.
    view_w, view_h = 1920, 1080
    margin = 12
    view_gap = 8
    row_gap = 12
    width = 2 * margin + 3 * view_w + 2 * view_gap
    height = 2 * margin + len(CARDS) * view_h + (len(CARDS) - 1) * row_gap
    canvas = Image.new("RGB", (width, height), "white")
    label_font = ImageFont.truetype(FONT_BOLD, 54)

    records = []
    for index, (name, description, views) in enumerate(CARDS):
        y = margin + index * (view_h + row_gap)

        source_rows = []
        for view_index, (rel, caption) in enumerate(views):
            x = margin + view_index * (view_w + view_gap)
            image = Image.open(f(rel)).convert("RGB")
            image = ImageOps.fit(
                image,
                (view_w, view_h),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            if view_index == 0:
                overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                bbox = overlay_draw.textbbox((0, 0), name, font=label_font)
                label_w = bbox[2] - bbox[0] + 48
                label_h = bbox[3] - bbox[1] + 34
                overlay_draw.rounded_rectangle(
                    (18, 18, 18 + label_w, 18 + label_h),
                    radius=12,
                    fill=(10, 15, 22, 205),
                )
                overlay_draw.text((42, 27 - bbox[1]), name, font=label_font, fill="white")
                image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
            canvas.paste(image, (x, y))
            source_rows.append({"path": rel, "sha256": sha256(f(rel)), "caption": caption})
        records.append(
            {
                "operator": name.split()[0],
                "title": name,
                "description": description,
                "views": source_rows,
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PNG, optimize=True)
    preview_w = width // 2
    preview_h = round(height * preview_w / width)
    canvas.resize((preview_w, preview_h), Image.Resampling.LANCZOS).save(
        OUT_JPG,
        quality=94,
        optimize=True,
    )
    manifest = {
        "schema_version": "kinofail.11-operator-multiview-gallery.v4",
        "created_utc": datetime.now(UTC).isoformat(),
        "layout": {
            "operators": 11,
            "views_per_operator": 3,
            "canvas_px": [width, height],
            "view_px": [view_w, view_h],
            "style": "full_width_rows_in_image_labels_only",
        },
        "generated_imagery": False,
        "deterministic_compositing_only": True,
        "cards": records,
        "outputs": {
            str(OUT_PNG.relative_to(ROOT)): sha256(OUT_PNG),
            str(OUT_JPG.relative_to(ROOT)): sha256(OUT_JPG),
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT_PNG)
    print(OUT_JPG)
    print(OUT_MANIFEST)


if __name__ == "__main__":
    main()
