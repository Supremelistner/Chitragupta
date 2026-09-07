"""Generate synthetic test documents for Chitragupta.

Real user documents (with real Aadhaar/PAN/etc.) live in ``data/Sample/`` and
are gitignored. Tests, demos, and CI must use the synthetic documents this
script writes to ``data/Synthetic/`` — the values are obviously fake (e.g.
Aadhaar ``9999 8888 7777``) so any accidental commit is harmless and the
fields round-trip cleanly through the existing extraction pipeline.

Usage:
    python scripts/generate_synthetic_docs.py

Output (under data/Synthetic/):
    aadhaar.png
    pan.png
    marksheet.png
    vehicle.png

Each image is a simple white-background PNG with the document's title and
field labels in English. Numbers and names are deliberately unrealistic so
they're never mistaken for real PII.

Style notes:
- Aadhaar layout mimics the standard "front" half (photo box, name, number,
  DOB, gender, address).
- PAN card layout mimics the standard form (name, father's name, DOB, PAN).
- Marksheet layout mimics a single-subject school marksheet (name, roll,
  DOB, school, subject marks).
- Vehicle RC layout mimics the standard Form 23 (reg no, owner, vehicle,
  chassis, engine).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "Synthetic"


# ---------------------------------------------------------------------------
# Fake but recognisable data. All numbers repeat the same digit twice and
# trivially obvious patterns (1234 5678 9012, AB123456CD) so a human or a
# regex can immediately tell these aren't real.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AadhaarData:
    name: str = "TEST USER SAMPLE"
    aadhaar_number: str = "9999 8888 7777"
    dob: str = "01/01/1990"
    gender: str = "MALE"
    address: str = "HOUSE NO 123, TEST STREET, SAMPLE CITY 000000"


@dataclass(frozen=True)
class PanData:
    name: str = "TEST USER SAMPLE"
    father_name: str = "TEST FATHER SAMPLE"
    dob: str = "01/01/1990"
    pan_number: str = "ABCDE1234F"


@dataclass(frozen=True)
class MarksheetData:
    name: str = "TEST USER SAMPLE"
    roll_number: str = "9999999"
    dob: str = "01/01/1990"
    school: str = "SAMPLE TEST SCHOOL"
    subject: str = "TEST SUBJECT"
    marks_obtained: str = "099"
    max_marks: str = "100"


@dataclass(frozen=True)
class VehicleData:
    registration_no: str = "TEST 0001"
    owner_name: str = "TEST USER SAMPLE"
    vehicle_class: str = "TEST-CLASS"
    chassis_no: str = "TESTCHASSIS0001"
    engine_no: str = "TESTENGINE0001"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _load_font(size: int) -> ImageFont.ImageFont:
    """Load a monospaced font that exists on most systems.

    Falls back to PIL's default bitmap font if nothing else is found, so the
    script works in CI containers without any TTF bundled.
    """
    candidates: Sequence[str] = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str = "black",
) -> None:
    draw.text(xy, text, fill=fill, font=font)


# ---------------------------------------------------------------------------
# Document renderers — each one writes a single PNG to OUT_DIR.
# Keep the layouts simple: title, photo box (where applicable), field list.
# The goal is "looks enough like a real document that the LLM extractor can
# practice on it", not pixel-perfect reproduction.
# ---------------------------------------------------------------------------
def render_aadhaar(out: Path) -> None:
    data = AadhaarData()
    img = Image.new("RGB", (800, 500), "white")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(28)
    label_font = _load_font(18)
    value_font = _load_font(22)
    _draw_text(draw, (40, 30), "Aadhaar Card (Synthetic Test Image)", font=title_font)
    _draw_text(draw, (40, 80), "(Not a real document)", font=label_font, fill="red")
    # Photo placeholder box
    draw.rectangle([(540, 100), (740, 320)], outline="black", width=2)
    _draw_text(draw, (560, 200), "PHOTO", font=label_font)
    fields = [
        ("Name", data.name),
        ("Aadhaar No", data.aadhaar_number),
        ("DOB", data.dob),
        ("Gender", data.gender),
        ("Address", data.address),
    ]
    y = 130
    for label, value in fields:
        _draw_text(draw, (40, y), f"{label}:", font=label_font)
        _draw_text(draw, (200, y), value, font=value_font)
        y += 50
    img.save(out, "PNG")


def render_pan(out: Path) -> None:
    data = PanData()
    img = Image.new("RGB", (700, 450), "white")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(28)
    label_font = _load_font(18)
    value_font = _load_font(22)
    _draw_text(draw, (40, 30), "PAN Card (Synthetic Test Image)", font=title_font)
    _draw_text(draw, (40, 80), "(Not a real document)", font=label_font, fill="red")
    fields = [
        ("Name", data.name),
        ("Father's Name", data.father_name),
        ("DOB", data.dob),
        ("PAN", data.pan_number),
    ]
    y = 130
    for label, value in fields:
        _draw_text(draw, (40, y), f"{label}:", font=label_font)
        _draw_text(draw, (220, y), value, font=value_font)
        y += 50
    img.save(out, "PNG")


def render_marksheet(out: Path) -> None:
    data = MarksheetData()
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(28)
    label_font = _load_font(18)
    value_font = _load_font(22)
    _draw_text(draw, (40, 30), "Marksheet (Synthetic Test Image)", font=title_font)
    _draw_text(draw, (40, 80), "(Not a real document)", font=label_font, fill="red")
    fields = [
        ("Name", data.name),
        ("Roll No", data.roll_number),
        ("DOB", data.dob),
        ("School", data.school),
        ("Subject", data.subject),
        ("Marks", f"{data.marks_obtained} / {data.max_marks}"),
    ]
    y = 130
    for label, value in fields:
        _draw_text(draw, (40, y), f"{label}:", font=label_font)
        _draw_text(draw, (200, y), value, font=value_font)
        y += 50
    img.save(out, "PNG")


def render_vehicle(out: Path) -> None:
    data = VehicleData()
    img = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(28)
    label_font = _load_font(18)
    value_font = _load_font(22)
    _draw_text(
        draw, (40, 30), "Vehicle Registration (Synthetic Test Image)", font=title_font
    )
    _draw_text(draw, (40, 80), "(Not a real document)", font=label_font, fill="red")
    fields = [
        ("Registration No", data.registration_no),
        ("Owner", data.owner_name),
        ("Vehicle Class", data.vehicle_class),
        ("Chassis No", data.chassis_no),
        ("Engine No", data.engine_no),
    ]
    y = 130
    for label, value in fields:
        _draw_text(draw, (40, y), f"{label}:", font=label_font)
        _draw_text(draw, (260, y), value, font=value_font)
        y += 50
    img.save(out, "PNG")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
RENDERERS = {
    "aadhaar.png": render_aadhaar,
    "pan.png": render_pan,
    "marksheet.png": render_marksheet,
    "vehicle.png": render_vehicle,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help="Output directory (default: data/Synthetic/)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Render only this filename (e.g. aadhaar.png). Default: all four.",
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for filename, renderer in RENDERERS.items():
        if args.only and filename != args.only:
            continue
        target = args.out / filename
        renderer(target)
        rendered.append(target)
        print(f"wrote {target}")
    if not rendered:
        print("nothing rendered (--only filter matched no files)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())