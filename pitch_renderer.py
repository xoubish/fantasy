"""
FPL Pitch Formation Figure Renderer (pitch_renderer.py)

Generates a high-resolution FPL-style pitch visualization (PNG) showing:
- Starting XI in their exact tactical formation (GKP, DEF, MID, FWD rows)
- Authentic club kit colors & sleeve/stripe patterns
- Captain (C) & Vice-Captain (V) armbands, Dream Team stars, prices, fixtures, and Expected Points (xP)
- 4-player Substitutes bench box at the bottom
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from predictor import OptimizedSquadResult, PlayerPrediction

# Club kit colors: (body_color, sleeve_color, stripe_color_or_None, text_color)
CLUB_KITS: Dict[str, Tuple[str, str, str | None, str]] = {
    "ARS": ("#EF0107", "#FFFFFF", None, "#FFFFFF"),
    "CHE": ("#034694", "#034694", None, "#FFFFFF"),
    "MCI": ("#6CABDD", "#FFFFFF", None, "#1C2C5B"),
    "LIV": ("#C8102E", "#C8102E", None, "#FFFFFF"),
    "BHA": ("#0057B8", "#0057B8", "#FFFFFF", "#FFFFFF"),
    "BRE": ("#E30613", "#FFFFFF", "#FFFFFF", "#FFFFFF"),
    "TOT": ("#F5F7FA", "#132257", None, "#132257"),
    "EVE": ("#003399", "#FFFFFF", None, "#FFFFFF"),
    "LEE": ("#FFFFFF", "#FFCD00", None, "#1D428A"),
    "SUN": ("#EB172B", "#FFFFFF", "#FFFFFF", "#FFFFFF"),
    "HUL": ("#F5A122", "#111111", "#111111", "#FFFFFF"),
    "AVL": ("#670E36", "#95BFE5", None, "#FFFFFF"),
    "BOU": ("#DA291C", "#111111", "#111111", "#FFFFFF"),
    "COV": ("#59CBE8", "#1D2951", None, "#FFFFFF"),
    "CRY": ("#1B458F", "#C4122E", "#C4122E", "#FFFFFF"),
    "IPS": ("#3A64A3", "#FFFFFF", None, "#FFFFFF"),
    "FUL": ("#FFFFFF", "#111111", None, "#111111"),
    "NEW": ("#111111", "#FFFFFF", "#FFFFFF", "#FFFFFF"),
    "MUN": ("#DA291C", "#FFFFFF", None, "#FFFFFF"),
    "NFO": ("#DD0000", "#FFFFFF", None, "#FFFFFF"),
}


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_jersey(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    team_short: str,
    is_gkp: bool = False,
) -> None:
    """Draws a stylized football shirt centered at (cx, cy)."""
    if is_gkp:
        body_col = "#FF5992" if team_short == "BHA" else ("#2ECC71" if team_short == "EVE" else "#1E5631")
        sleeve_col = body_col
        stripe_col = None
    else:
        body_col, sleeve_col, stripe_col, _ = CLUB_KITS.get(
            team_short, ("#37003C", "#FFFFFF", None, "#FFFFFF")
        )

    # Left and right sleeves (polygon)
    left_sleeve = [
        (cx - 22, cy - 28),
        (cx - 42, cy - 10),
        (cx - 34, cy + 4),
        (cx - 20, cy - 8),
    ]
    right_sleeve = [
        (cx + 22, cy - 28),
        (cx + 42, cy - 10),
        (cx + 34, cy + 4),
        (cx + 20, cy - 8),
    ]
    draw.polygon(left_sleeve, fill=sleeve_col, outline="#222222")
    draw.polygon(right_sleeve, fill=sleeve_col, outline="#222222")

    # Main torso
    torso_box = [cx - 22, cy - 28, cx + 22, cy + 28]
    draw.rounded_rectangle(torso_box, radius=4, fill=body_col, outline="#222222", width=2)

    # Optional vertical stripes
    if stripe_col:
        for sx in (-12, 0, 12):
            draw.rectangle(
                [cx + sx - 3, cy - 26, cx + sx + 3, cy + 26],
                fill=stripe_col,
            )

    # Collar cutout
    draw.pieslice(
        [cx - 10, cy - 34, cx + 10, cy - 20],
        start=0,
        end=180,
        fill="#1D1128",
        outline="#222222",
    )


def _draw_player_card(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    pred: PlayerPrediction,
    is_captain: bool,
    is_vice: bool,
    sub_label: str | None = None,
) -> None:
    card_w, card_h = 156, 152
    x0, y0 = cx - card_w // 2, cy - card_h // 2
    x1, y1 = x0 + card_w, y0 + card_h

    # Sub label above card if on bench
    if sub_label:
        font_sub = _load_font(14, bold=True)
        draw.text(
            (cx, y0 - 14),
            sub_label,
            fill="#FFFFFF",
            font=font_sub,
            anchor="mm",
        )

    # Card background
    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=10,
        fill="#1E6B40" if not sub_label else "#2B583E",
        outline="#52B788",
        width=2,
    )

    # Draw jersey
    _draw_jersey(
        draw,
        cx,
        y0 + 50,
        pred.player.team_short,
        is_gkp=(pred.player.position == "GKP"),
    )

    # Expected Points (xP) badge top-right
    xp_val = pred.expected_points * 2 if is_captain else pred.expected_points
    xp_bg = "#00FF87" if xp_val >= 6.0 else "#E9F5EC"
    draw.rounded_rectangle(
        [x1 - 56, y0 + 6, x1 - 6, y0 + 28],
        radius=6,
        fill=xp_bg,
        outline="#1B4332",
    )
    font_xp = _load_font(13, bold=True)
    draw.text(
        (x1 - 31, y0 + 17),
        f"{xp_val:.1f}p",
        fill="#111111",
        font=font_xp,
        anchor="mm",
    )

    # Price tag top-left
    font_price = _load_font(12, bold=True)
    draw.rounded_rectangle(
        [x0 + 6, y0 + 6, x0 + 52, y0 + 26],
        radius=5,
        fill="#123724",
    )
    draw.text(
        (x0 + 29, y0 + 16),
        f"£{pred.player.price:.1f}m",
        fill="#A3E4D7",
        font=font_price,
        anchor="mm",
    )

    # Captain / Vice-Captain circle badge on left
    if is_captain or is_vice:
        badge_y = y0 + 44
        badge_col = "#37003C" if is_captain else "#2D3142"
        draw.ellipse(
            [x0 + 8, badge_y - 11, x0 + 30, badge_y + 11],
            fill=badge_col,
            outline="#00FF87" if is_captain else "#FFFFFF",
            width=2,
        )
        font_cap = _load_font(13, bold=True)
        draw.text(
            (x0 + 19, badge_y),
            "C" if is_captain else "V",
            fill="#00FF87" if is_captain else "#FFFFFF",
            font=font_cap,
            anchor="mm",
        )

    # Dream Team Star badge
    if pred.player.in_dreamteam:
        star_cx, star_cy = x0 + 19, y0 + 72
        draw.ellipse(
            [star_cx - 10, star_cy - 10, star_cx + 10, star_cy + 10],
            fill="#37003C",
            outline="#FFD166",
            width=1,
        )
        star_pts = []
        for i in range(10):
            angle = math.radians(-90 + i * 36)
            r = 6.5 if i % 2 == 0 else 2.8
            star_pts.append(
                (star_cx + r * math.cos(angle), star_cy + r * math.sin(angle))
            )
        draw.polygon(star_pts, fill="#FFD166")

    # Name bar
    name_y0 = y1 - 52
    draw.rectangle([x0 + 2, name_y0, x1 - 2, name_y0 + 26], fill="#FFFFFF")
    font_name = _load_font(15, bold=True)
    draw.text(
        (cx, name_y0 + 13),
        pred.player.web_name,
        fill="#240028",
        font=font_name,
        anchor="mm",
    )

    # Fixture bar at bottom
    fix_y0 = name_y0 + 26
    draw.rounded_rectangle(
        [x0 + 2, fix_y0, x1 - 2, y1 - 2],
        radius=8,
        fill="#E8E4F0",
    )
    font_fix = _load_font(13, bold=True)
    draw.text(
        (cx, fix_y0 + 12),
        f"{pred.player.team_short} • {pred.player.next_opponent}",
        fill="#37003C",
        font=font_fix,
        anchor="mm",
    )


def render_pitch_figure(
    squad_result: OptimizedSquadResult,
    total_cost: float,
    budget: float,
    output_path: Path,
    title: str = "FPL AI Optimizer — Best Squad of the Week",
) -> Path:
    """Renders the Starting XI and Bench onto a football pitch graphic and saves as PNG."""
    width, height = 1160, 1280
    img = Image.new("RGB", (width, height), "#28002E")
    draw = ImageDraw.Draw(img)

    # 1. Header Banner
    draw.rectangle([0, 0, width, 108], fill="#37003C")
    draw.rectangle([0, 104, width, 108], fill="#00FF87")

    font_title = _load_font(28, bold=True)
    font_sub = _load_font(18, bold=True)

    draw.text((40, 34), title, fill="#FFFFFF", font=font_title, anchor="lm")
    subtitle = (
        f"Formation: {squad_result.formation}   |   "
        f"Captain: {squad_result.captain.player.web_name} (2x)   |   "
        f"Vice: {squad_result.vice_captain.player.web_name}   |   "
        f"Squad Cost: £{total_cost:.1f}m / £{budget:.1f}m"
    )
    draw.text((40, 74), subtitle, fill="#00FF87", font=font_sub, anchor="lm")

    # Score badge in top-right header
    draw.rounded_rectangle(
        [width - 255, 18, width - 35, 90],
        radius=12,
        fill="#00FF87",
    )
    font_score_lbl = _load_font(13, bold=True)
    font_score_val = _load_font(28, bold=True)
    draw.text(
        (width - 145, 36),
        "PROJECTED SCORE",
        fill="#240028",
        font=font_score_lbl,
        anchor="mm",
    )
    draw.text(
        (width - 145, 67),
        f"{squad_result.total_expected_points:.1f} pts",
        fill="#240028",
        font=font_score_val,
        anchor="mm",
    )

    # 2. Football Pitch Background (y=125 to y=995)
    pitch_top, pitch_bot = 125, 995
    pitch_left, pitch_right = 50, width - 50
    stripe_h = (pitch_bot - pitch_top) // 8
    for i in range(8):
        sy0 = pitch_top + i * stripe_h
        sy1 = pitch_bot if i == 7 else sy0 + stripe_h
        turf_col = "#1E7E48" if i % 2 == 0 else "#248F52"
        draw.rectangle([pitch_left, sy0, pitch_right, sy1], fill=turf_col)

    # Pitch markings (white lines)
    draw.rectangle(
        [pitch_left, pitch_top, pitch_right, pitch_bot],
        outline="#FFFFFF",
        width=3,
    )
    # Penalty box at top
    cx_mid = width // 2
    draw.rectangle(
        [cx_mid - 230, pitch_top, cx_mid + 230, pitch_top + 135],
        outline="#FFFFFF",
        width=3,
    )
    draw.rectangle(
        [cx_mid - 105, pitch_top, cx_mid + 105, pitch_top + 55],
        outline="#FFFFFF",
        width=3,
    )
    # Halfway line & center circle near bottom of starting area
    half_y = 820
    draw.line([pitch_left, half_y, pitch_right, half_y], fill="#FFFFFF", width=3)
    draw.ellipse(
        [cx_mid - 85, half_y - 85, cx_mid + 85, half_y + 85],
        outline="#FFFFFF",
        width=3,
    )

    # 3. Group Starting XI by Position Rows
    gkps = [p for p in squad_result.starting_xi if p.player.position == "GKP"]
    defs = [p for p in squad_result.starting_xi if p.player.position == "DEF"]
    mids = [p for p in squad_result.starting_xi if p.player.position == "MID"]
    fwds = [p for p in squad_result.starting_xi if p.player.position == "FWD"]

    row_y = {
        "GKP": 220,
        "DEF": 420,
        "MID": 635,
        "FWD": 855,
    }

    for pos_key, group in [("GKP", gkps), ("DEF", defs), ("MID", mids), ("FWD", fwds)]:
        n = len(group)
        if n == 0:
            continue
        span = width - 160
        step = span // n
        start_x = 80 + step // 2
        for idx, pred in enumerate(group):
            px = start_x + idx * step
            py = row_y[pos_key]
            _draw_player_card(
                draw,
                px,
                py,
                pred,
                is_captain=(pred.player.id == squad_result.captain.player.id),
                is_vice=(pred.player.id == squad_result.vice_captain.player.id),
            )

    # 4. Substitutes Bench Box at Bottom (y=1015 to y=1255)
    bench_box = [130, 1015, width - 130, 1255]
    draw.rounded_rectangle(
        bench_box,
        radius=16,
        fill="#1F4E38",
        outline="#52B788",
        width=2,
    )
    font_bench_title = _load_font(20, bold=True)
    draw.text(
        (cx_mid, 1235),
        "Substitutes",
        fill="#FFFFFF",
        font=font_bench_title,
        anchor="mm",
    )

    bench_all = [squad_result.bench_gkp] + squad_result.bench_outfield
    bench_labels = ["GKP"] + [
        f"{i}. {bp.player.position}" for i, bp in enumerate(squad_result.bench_outfield, 1)
    ]
    b_span = (width - 300) // max(len(bench_all), 1)
    b_start_x = 150 + b_span // 2
    for idx, (bp, lbl) in enumerate(zip(bench_all, bench_labels)):
        bx = b_start_x + idx * b_span
        _draw_player_card(
            draw,
            bx,
            1125,
            bp,
            is_captain=False,
            is_vice=False,
            sub_label=lbl,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return output_path
