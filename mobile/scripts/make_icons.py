"""Draws the PPB app icon (a safety-yellow hard hat with a green tick badge on the app's blue)
and writes every file mobile/app.json points at, plus a preview sheet.

Usage (from the repo root, with the backend's environment, which has Pillow):
    uv run python mobile/scripts/make_icons.py mobile/assets icon-preview.png
Drawn at 4x and scaled down, for smooth edges.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

BLUE = (29, 111, 214, 255)          # #1d6fd6, the app's accent
BLUE_DARK = (21, 86, 168, 255)      # a shade for the background's lower half
YELLOW = (255, 194, 14, 255)        # safety yellow
YELLOW_SHADE = (224, 160, 0, 255)   # the hat's ridge and brim edge
GREEN = (26, 127, 55, 255)          # #1a7f37, the app's PASS green
WHITE = (255, 255, 255, 255)
SS = 4                              # supersampling


def glyph(size: int, scale: float, mono: bool = False) -> Image.Image:
    """The hat and badge on a transparent square `size` px wide, filling `scale` of it."""
    s = size * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    u = s * scale / 1000          # 1 unit = 1/1000 of the glyph's box
    ox = oy = (s - 1000 * u) / 2  # the glyph's box, centred

    def box(x0, y0, x1, y1):
        return [ox + x0 * u, oy + y0 * u, ox + x1 * u, oy + y1 * u]

    hat = WHITE if mono else YELLOW
    shade = WHITE if mono else YELLOW_SHADE
    # Dome: the top half of an ellipse.
    d.pieslice(box(150, 130, 850, 830), 180, 360, fill=hat)
    # Ridge down the middle of the dome.
    if not mono:
        d.rounded_rectangle(box(462, 140, 538, 480), radius=38 * u, fill=shade)
    # Brim, wider than the dome.
    d.rounded_rectangle(box(60, 470, 940, 590), radius=60 * u, fill=hat)
    if not mono:
        d.rounded_rectangle(box(60, 548, 940, 590), radius=21 * u, fill=shade)
        d.rectangle(box(90, 548, 910, 560), fill=hat)

    # Tick badge, lower right, overlapping the brim.
    cx, cy, r, ring = 715, 700, 215, 34
    if mono:
        # Monochrome icons are one colour: cut the badge out of the hat, and the tick out of the badge.
        d.ellipse(box(cx - r - ring, cy - r - ring, cx + r + ring, cy + r + ring), fill=(0, 0, 0, 0))
        d.ellipse(box(cx - r, cy - r, cx + r, cy + r), fill=WHITE)
        tick_colour = (0, 0, 0, 0)
    else:
        d.ellipse(box(cx - r - ring, cy - r - ring, cx + r + ring, cy + r + ring), fill=WHITE)
        d.ellipse(box(cx - r, cy - r, cx + r, cy + r), fill=GREEN)
        tick_colour = WHITE
    pts = [(cx - 110, cy + 5), (cx - 30, cy + 90), (cx + 120, cy - 85)]
    d.line([(ox + x * u, oy + y * u) for x, y in pts], fill=tick_colour, width=int(58 * u), joint="curve")
    for x, y in (pts[0], pts[2]):  # round the tick's ends
        d.ellipse(box(x - 29, y - 29, x + 29, y + 29), fill=tick_colour)

    return img.resize((size, size), Image.LANCZOS)


def background(size: int) -> Image.Image:
    """The app's blue, a shade darker towards the bottom."""
    img = Image.new("RGBA", (size, size), BLUE)
    d = ImageDraw.Draw(img)
    for y in range(size):
        t = y / (size - 1)
        c = tuple(round(BLUE[i] + (BLUE_DARK[i] - BLUE[i]) * t) for i in range(3)) + (255,)
        d.line([(0, y), (size, y)], fill=c)
    return img


def full_icon(size: int) -> Image.Image:
    """Opaque (iOS refuses transparency in an app icon): background plus the glyph at 80%."""
    img = background(size)
    img.alpha_composite(glyph(size, 0.80))
    return img.convert("RGB")


def main(assets: Path, preview: Path) -> None:
    full_icon(1024).save(assets / "icon.png")
    full_icon(48).save(assets / "favicon.png")
    # Android adaptive icon: the launcher masks a 108-unit square to a shape and shows the middle
    # 66 units, so the glyph stays within about 60% of the canvas.
    background(512).convert("RGB").save(assets / "android-icon-background.png")
    glyph(512, 0.58).save(assets / "android-icon-foreground.png")
    glyph(432, 0.58, mono=True).save(assets / "android-icon-monochrome.png")

    # Preview sheet: iOS-style rounded square at three sizes, Android circle mask, monochrome.
    sheet = Image.new("RGB", (1240, 400), (245, 246, 248))
    x = 20
    for px in (300, 120, 48):
        icon = full_icon(px).convert("RGBA")
        mask = Image.new("L", (px, px), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, px - 1, px - 1], radius=int(px * 0.22), fill=255)
        sheet.paste(icon, (x, 50 + (300 - px) // 2), mask)
        x += px + 40
    adaptive = background(300)
    adaptive.alpha_composite(glyph(300, 0.58))
    circle = Image.new("L", (300, 300), 0)
    ImageDraw.Draw(circle).ellipse([0, 0, 299, 299], fill=255)
    sheet.paste(adaptive, (x, 50), circle)
    x += 340
    mono = Image.new("RGBA", (300, 300), (60, 64, 67, 255))
    mono.alpha_composite(glyph(300, 0.58, mono=True))
    sheet.paste(mono, (x, 50), circle)
    sheet.save(preview)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
