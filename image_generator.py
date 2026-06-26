"""
image_generator.py
Генерация финальной инфографики PSL Rating (1000x1500, тёмная тема, неон).

Использует только Pillow + OpenCV. Никаких внешних сервисов.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from analyzer import (
    FaceAnalysisResult,
    FACE_OVAL,
    PROFILE_JAW,
    get_e_line_points,
)

# ---------------------------------------------------------------------------
# Шрифты (нужны кириллические TTF, иначе текст рисуется как "????")
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FONT_REGULAR_PATH = BASE_DIR / "fonts" / "DejaVuSans.ttf"
FONT_BOLD_PATH = BASE_DIR / "fonts" / "DejaVuSans-Bold.ttf"
_FONT_CACHE: dict = {}

# ---------------------------------------------------------------------------
# Константы холста
# ---------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 1000, 1500
LEFT_W = 600
RIGHT_W = 400
PHOTO_SIZE = 600
BOTTOM_H = 300

BG_COLOR = (8, 10, 18)
PANEL_COLOR = (16, 19, 32)
NEON_BLUE = (0, 220, 255)
NEON_CYAN_SOFT = (0, 255, 200)
NEON_PINK = (255, 0, 140)
TEXT_COLOR = (230, 235, 245)
SUBTEXT_COLOR = (140, 150, 170)

TIER_COLORS = [
    ("sub-3", 0.0, 2.9, (120, 30, 30)),
    ("sub-5", 3.0, 4.9, (170, 60, 30)),
    ("ltn", 5.0, 5.4, (190, 110, 20)),
    ("lmtn", 5.5, 5.9, (200, 150, 20)),
    ("mtn", 6.0, 6.5, (190, 190, 30)),
    ("mhtn", 6.6, 6.7, (140, 200, 40)),
    ("htn", 6.8, 7.3, (60, 200, 90)),
    ("chadlite", 7.4, 8.0, (30, 200, 170)),
    ("chad", 8.1, 8.9, (30, 150, 230)),
    ("true adam", 9.0, 10.0, (190, 60, 230)),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Грузит TTF-шрифт с поддержкой кириллицы (PIL.load_default не умеет кириллицу)."""
    cache_key = (size, bold)
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]

    path = FONT_BOLD_PATH if bold else FONT_REGULAR_PATH
    try:
        font = ImageFont.truetype(str(path), size=size)
    except OSError:
        # Запасной вариант, если файл шрифта не найден рядом со скриптом.
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()

    _FONT_CACHE[cache_key] = font
    return font


def _text(draw: ImageDraw.ImageDraw, xy, text, size, color=TEXT_COLOR, anchor=None, bold=False):
    draw.text(xy, text, font=_font(size, bold=bold), fill=color, anchor=anchor)


def _scale_points(points: np.ndarray, orig_w: int, orig_h: int,
                   crop_box: tuple, target_size: int) -> np.ndarray:
    """Пересчитывает координаты landmark-точек после crop+resize изображения."""
    x0, y0, x1, y1 = crop_box
    crop_w, crop_h = x1 - x0, y1 - y0
    scale = target_size / crop_w
    pts = points.copy()
    pts[:, 0] = (pts[:, 0] - x0) * scale
    pts[:, 1] = (pts[:, 1] - y0) * scale
    return pts


def _center_crop_box(w: int, h: int) -> tuple:
    """Возвращает координаты квадратного crop по центру изображения."""
    side = min(w, h)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return x0, y0, x0 + side, y0 + side


def _load_square_resized(path: str, size: int) -> tuple:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    box = _center_crop_box(w, h)
    x0, y0, x1, y1 = box
    cropped = img[y0:y1, x0:x1]
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    return resized, box


def _cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Отрисовка front фото с face mesh
# ---------------------------------------------------------------------------
def _draw_front_photo(front_path: str, landmarks: np.ndarray) -> Image.Image:
    img, box = _load_square_resized(front_path, PHOTO_SIZE)
    pil_img = _cv2_to_pil(img).convert("RGBA")

    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    h, w = img.shape[:2]
    pts = _scale_points(landmarks, w, h, box, PHOTO_SIZE)

    # Тонкие точки по всем landmark
    for p in pts:
        x, y = p[0], p[1]
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(0, 220, 255, 160))

    # Контур овала лица тонкой неоновой линией
    oval_pts = [(pts[i][0], pts[i][1]) for i in FACE_OVAL]
    draw.line(oval_pts + [oval_pts[0]], fill=(0, 255, 230, 200), width=2)

    blurred_glow = overlay.filter(ImageFilter.GaussianBlur(1))
    pil_img = Image.alpha_composite(pil_img, blurred_glow)
    pil_img = Image.alpha_composite(pil_img, overlay)
    return pil_img.convert("RGB")


def _draw_profile_photo(profile_path: str, landmarks: np.ndarray) -> Image.Image:
    img, box = _load_square_resized(profile_path, PHOTO_SIZE)
    pil_img = _cv2_to_pil(img).convert("RGBA")

    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    h, w = img.shape[:2]
    pts = _scale_points(landmarks, w, h, box, PHOTO_SIZE)

    # Контур челюсти
    jaw_pts = [(pts[i][0], pts[i][1]) for i in PROFILE_JAW if i < len(pts)]
    if len(jaw_pts) > 1:
        draw.line(jaw_pts, fill=(0, 220, 255, 220), width=3)

    # E-line (линия Риккетса): от кончика носа до кончика подбородка
    nose_tip, chin = get_e_line_points(landmarks)
    nose_tip_s = _scale_points(np.array([[nose_tip[0], nose_tip[1], 0]]), w, h, box, PHOTO_SIZE)[0]
    chin_s = _scale_points(np.array([[chin[0], chin[1], 0]]), w, h, box, PHOTO_SIZE)[0]
    draw.line([(nose_tip_s[0], nose_tip_s[1]), (chin_s[0], chin_s[1])],
              fill=(255, 0, 140, 230), width=2)
    _text(draw, (chin_s[0] + 8, chin_s[1] - 20), "E-line", 14, (255, 0, 140, 255))

    pil_img = Image.alpha_composite(pil_img, overlay)
    return pil_img.convert("RGB")


# ---------------------------------------------------------------------------
# Вертикальная шкала тиров + стрелка PSL
# ---------------------------------------------------------------------------
def _draw_vertical_scale(draw: ImageDraw.ImageDraw, x: int, y_top: int, y_bottom: int,
                          width: int, psl_score: float):
    height = y_bottom - y_top

    def score_to_y(score: float) -> int:
        frac = score / 10.0
        return int(y_bottom - frac * height)

    # Градиентный фон шкалы (сегменты по тирам)
    for name, lo, hi, color in TIER_COLORS:
        y1 = score_to_y(lo)
        y2 = score_to_y(hi)
        draw.rectangle([x, y2, x + width, y1], fill=color)
        label_y = (y1 + y2) // 2
        _text(draw, (x + width + 10, label_y), name, 13, TEXT_COLOR, anchor="lm")

    draw.rectangle([x, y_top, x + width, y_bottom], outline=(60, 65, 80), width=2)

    # числовые засечки
    for val in range(0, 11):
        ty = score_to_y(val)
        draw.line([(x - 6, ty), (x, ty)], fill=SUBTEXT_COLOR, width=1)
        _text(draw, (x - 10, ty), str(val), 11, SUBTEXT_COLOR, anchor="rm")

    # Неоновая стрелка к итоговому баллу
    arrow_y = score_to_y(psl_score)
    arrow_x0 = x - 55
    draw.line([(arrow_x0, arrow_y), (x - 4, arrow_y)], fill=NEON_BLUE, width=4)
    draw.polygon(
        [(x - 4, arrow_y - 8), (x + 6, arrow_y), (x - 4, arrow_y + 8)],
        fill=NEON_BLUE,
    )
    _text(draw, (arrow_x0 - 6, arrow_y), f"{psl_score:.1f}", 22, NEON_BLUE, anchor="rm", bold=True)


def _draw_potential_box(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
                         potential_score: float):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=PANEL_COLOR,
                            outline=NEON_CYAN_SOFT, width=2)
    _text(draw, (x + w // 2, y + 18), "POTENTIAL SCORE", 13, SUBTEXT_COLOR, anchor="mm")
    _text(draw, (x + w // 2, y + h // 2 + 8), f"{potential_score:.1f}", 34, NEON_CYAN_SOFT, anchor="mm", bold=True)


# ---------------------------------------------------------------------------
# Прогресс-бар с градиентом
# ---------------------------------------------------------------------------
def _draw_progress_bar(canvas: Image.Image, x: int, y: int, w: int, h: int,
                        label: str, value_pct: float):
    draw = ImageDraw.Draw(canvas)
    _text(draw, (x, y), f"{label}", 16, TEXT_COLOR)
    _text(draw, (x + w, y), f"{value_pct:.0f}%", 16, NEON_BLUE, anchor="ra")

    bar_y0 = y + 26
    bar_y1 = bar_y0 + h
    draw.rounded_rectangle([x, bar_y0, x + w, bar_y1], radius=h // 2, fill=PANEL_COLOR,
                            outline=(50, 55, 70), width=1)

    fill_w = int(w * max(0.0, min(100.0, value_pct)) / 100.0)
    if fill_w > 2:
        gradient = Image.new("RGB", (fill_w, h), color=0)
        for i in range(fill_w):
            t = i / max(1, fill_w - 1)
            r = int(NEON_PINK[0] * (1 - t) + NEON_BLUE[0] * t)
            g = int(NEON_PINK[1] * (1 - t) + NEON_BLUE[1] * t)
            b = int(NEON_PINK[2] * (1 - t) + NEON_BLUE[2] * t)
            for yy in range(h):
                gradient.putpixel((i, yy), (r, g, b))
        mask = Image.new("L", (fill_w, h), 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.rounded_rectangle([0, 0, fill_w, h], radius=h // 2, fill=255)
        canvas.paste(gradient, (x, bar_y0), mask)


# ---------------------------------------------------------------------------
# Главная функция генерации
# ---------------------------------------------------------------------------
def generate_infographic(front_path: str, profile_path: str,
                          result: FaceAnalysisResult, output_path: str) -> str:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # Заголовок
    _text(draw, (CANVAS_W // 2, 14), "P S L   R A T I N G", 26, NEON_BLUE, anchor="mt", bold=True)

    top_offset = 50

    # --- Левая колонка: фото анфас + профиль ---
    front_photo = _draw_front_photo(front_path, result.front_landmarks_raw)
    profile_photo = _draw_profile_photo(profile_path, result.profile_landmarks_raw)
    canvas.paste(front_photo, (0, top_offset))
    canvas.paste(profile_photo, (0, top_offset + PHOTO_SIZE))

    # --- Правая колонка: шкала тиров + potential score ---
    right_x = LEFT_W
    draw.rectangle([right_x, 0, CANVAS_W, top_offset + 2 * PHOTO_SIZE], fill=BG_COLOR)

    scale_x = right_x + 90
    scale_top = top_offset + 30
    scale_bottom = top_offset + 2 * PHOTO_SIZE - 140
    _draw_vertical_scale(draw, scale_x, scale_top, scale_bottom, 26, result.psl_score)

    _text(draw, (right_x + 200, scale_top - 22), "ТИР:", 14, SUBTEXT_COLOR, anchor="mm")
    _text(draw, (right_x + 200, scale_top - 2), result.tier, 16, NEON_PINK, anchor="mm")

    _draw_potential_box(draw, right_x + 30, scale_bottom + 40, RIGHT_W - 60, 100,
                         result.potential_score)

    # --- Нижняя часть: 4 прогресс-бара ---
    bars_top = top_offset + 2 * PHOTO_SIZE + 20
    draw.rectangle([0, bars_top - 10, CANVAS_W, bars_top + BOTTOM_H], fill=(12, 14, 24))

    bar_x = 40
    bar_w = CANVAS_W - 80
    bar_h = 22
    gap = 56

    metrics = [
        ("Симметрия лица", result.symmetry_score),
        ("Структура костей", result.bone_score),
        ("Состояние кожи", result.skin_score),
        ("Гармония черт лица", result.harmony_score),
    ]
    for i, (label, value) in enumerate(metrics):
        _draw_progress_bar(canvas, bar_x, bars_top + i * gap, bar_w, bar_h, label, value)

    draw = ImageDraw.Draw(canvas)

    # Сноска о компенсации наклона
    if result.tilt_detected:
        _text(
            draw,
            (CANVAS_W // 2, CANVAS_H - 18),
            "*Обнаружен наклон головы, произведена ИИ-калибровка осей",
            11,
            SUBTEXT_COLOR,
            anchor="mm",
        )

    canvas.save(output_path, format="PNG")
    return output_path
