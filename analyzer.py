"""
analyzer.py
Локальный математический движок анализа лица для PSL Rating.

Все вычисления выполняются ИСКЛЮЧИТЕЛЬНО локально:
- OpenCV  -> загрузка изображений, анализ текстуры кожи (Laplacian variance)
- MediaPipe Face Mesh -> 468 (+ ирисы) 3D-landmarks лица
- numpy   -> геометрия, повороты, метрики

Никаких внешних платных API / нейросетевых сервисов не используется.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# ---------------------------------------------------------------------------
# Индексы ключевых точек MediaPipe Face Mesh (468 точек + 10 точек ирисов)
# ---------------------------------------------------------------------------
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
GLABELLA = 9          # точка между бровями (верх средней зоны лица)
SUBNASALE = 2          # подносовая точка
UPPER_LIP_TOP = 0      # верхняя точка верхней губы
CHIN_BOTTOM = 152      # низ подбородка
NOSE_TIP = 1           # кончик носа (центр для компенсации наклона)
LEFT_IRIS_CENTER = 468   # требует refine_landmarks=True
RIGHT_IRIS_CENTER = 473  # требует refine_landmarks=True

# Примерные пары симметричных точек (левая/правая половина лица)
SYMMETRY_PAIRS = [
    (33, 263),    # внешние уголки глаз
    (133, 362),   # внутренние уголки глаз
    (159, 386),   # верхние веки
    (145, 374),   # нижние веки
    (70, 300),    # внешние края бровей
    (105, 334),   # брови (середина)
    (234, 454),   # скулы
    (61, 291),    # уголки губ
    (48, 278),    # крылья носа
    (172, 397),   # линия челюсти (середина)
    (58, 288),    # линия челюсти (нижняя)
    (215, 435),   # скулы/щёки
]

# Контур овала лица (для отрисовки и оценки структуры костей)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]

# Точки щёк для анализа состояния кожи (по 1 региону на каждую сторону)
LEFT_CHEEK_REGION = [50, 101, 36, 205, 187, 123]
RIGHT_CHEEK_REGION = [280, 330, 266, 425, 411, 352]

# Точки для отрисовки/анализа в профиль
PROFILE_JAW = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172,
               136, 150, 149, 176, 148, 152]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1[:2] - p2[:2]))


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _closeness_score(actual: float, ideal: float, tolerance: float) -> float:
    """Возвращает % близости значения к идеалу (100% = точное совпадение)."""
    diff = abs(actual - ideal)
    score = 100.0 * math.exp(-(diff ** 2) / (2 * (tolerance ** 2)))
    return _clip(score)


def _extract_landmarks(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Возвращает массив (478, 3) пиксельных координат (x, y, z) или None."""
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as fm:
        results = fm.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0]
    pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm.landmark], dtype=np.float64)
    return pts


def _rotate_points(points: np.ndarray, center: np.ndarray, angle_deg: float) -> np.ndarray:
    """Поворачивает 2D-координаты точек вокруг center на angle_deg (по часовой)."""
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    shifted = points[:, :2] - center[:2]
    rotated = shifted @ rot.T
    out = points.copy()
    out[:, :2] = rotated + center[:2]
    return out


# ---------------------------------------------------------------------------
# Результат анализа
# ---------------------------------------------------------------------------
@dataclass
class FaceAnalysisResult:
    success: bool = False
    error: Optional[str] = None

    # landmarks (для отрисовки используем ОРИГИНАЛЬНЫЕ, не повёрнутые точки)
    front_landmarks_raw: Optional[np.ndarray] = None
    front_landmarks_corrected: Optional[np.ndarray] = None
    profile_landmarks_raw: Optional[np.ndarray] = None

    tilt_detected: bool = False
    tilt_angle: float = 0.0

    symmetry_score: float = 0.0
    fwhr: float = 0.0
    fwhr_score: float = 0.0
    lower_third_ratio: float = 0.0
    lower_third_score: float = 0.0
    ipd_ratio: float = 0.0
    ipd_score: float = 0.0
    skin_score: float = 0.0
    bone_score: float = 0.0
    harmony_score: float = 0.0

    psl_score: float = 0.0
    potential_score: float = 0.0
    tier: str = ""

    advice_soft: list = field(default_factory=list)
    advice_hard: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Шкала тиров
# ---------------------------------------------------------------------------
def score_to_tier(score: float) -> str:
    if score <= 2.9:
        return "sub-3"
    if score <= 4.9:
        return "sub-5"
    if score <= 5.4:
        return "ltn (Low-tier Normie)"
    if score <= 5.9:
        return "lmtn (Low-mid-tier Normie)"
    if score <= 6.5:
        return "mtn (Mid-tier Normie)"
    if score <= 6.7:
        return "mhtn (Mid-high-tier Normie)"
    if score <= 7.3:
        return "htn (High-tier Normie)"
    if score <= 8.0:
        return "chadlite"
    if score <= 8.9:
        return "chad"
    return "true adam"


# ---------------------------------------------------------------------------
# Основная функция анализа
# ---------------------------------------------------------------------------
def analyze_face(front_image_path: str, profile_image_path: str) -> FaceAnalysisResult:
    result = FaceAnalysisResult()

    front_img = cv2.imread(front_image_path)
    profile_img = cv2.imread(profile_image_path)

    if front_img is None or profile_img is None:
        result.error = "Не удалось прочитать одно из изображений."
        return result

    front_pts = _extract_landmarks(front_img)
    if front_pts is None:
        result.error = (
            "Лицо не обнаружено на фото АНФАС. Убедитесь, что лицо хорошо "
            "освещено и полностью видно в кадре."
        )
        return result

    profile_pts = _extract_landmarks(profile_img)
    if profile_pts is None:
        result.error = (
            "Лицо не обнаружено на фото ПРОФИЛЬ. Убедитесь, что лицо хорошо "
            "освещено и полностью видно в кадре."
        )
        return result

    result.front_landmarks_raw = front_pts
    result.profile_landmarks_raw = profile_pts

    # ------------------------------------------------------------------
    # 1. Компенсация наклона головы (Tilt Compensation, Roll)
    # ------------------------------------------------------------------
    left_eye = front_pts[LEFT_EYE_INNER]
    right_eye = front_pts[RIGHT_EYE_INNER]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    tilt_angle = math.degrees(math.atan2(dy, dx))

    corrected_pts = front_pts.copy()
    if abs(tilt_angle) > 1.0:
        result.tilt_detected = True
        result.tilt_angle = tilt_angle
        nose_center = front_pts[NOSE_TIP]
        corrected_pts = _rotate_points(front_pts, nose_center, -tilt_angle)

    result.front_landmarks_corrected = corrected_pts

    # ------------------------------------------------------------------
    # 2. Симметрия лица
    # ------------------------------------------------------------------
    face_width = _dist(corrected_pts[LEFT_CHEEK], corrected_pts[RIGHT_CHEEK])
    axis_x = float(np.mean([
        corrected_pts[NOSE_TIP][0],
        corrected_pts[GLABELLA][0],
        corrected_pts[UPPER_LIP_TOP][0],
        corrected_pts[CHIN_BOTTOM][0],
    ]))

    deviations = []
    for left_idx, right_idx in SYMMETRY_PAIRS:
        lp = corrected_pts[left_idx]
        rp = corrected_pts[right_idx]
        left_dist = abs(lp[0] - axis_x)
        right_dist = abs(rp[0] - axis_x)
        # учитываем и расхождение по высоте (Y) точек одной пары
        y_diff = abs(lp[1] - rp[1])
        x_diff = abs(left_dist - right_dist)
        total_diff = math.hypot(x_diff, y_diff)
        deviations.append(total_diff / face_width * 100.0)

    avg_deviation_pct = float(np.mean(deviations))
    result.symmetry_score = _clip(100.0 - avg_deviation_pct * 2.2)

    # ------------------------------------------------------------------
    # 3. Индекс FWHR (facial width-to-height ratio)
    # ------------------------------------------------------------------
    midface_height = _dist(corrected_pts[GLABELLA], corrected_pts[UPPER_LIP_TOP])
    fwhr = face_width / midface_height if midface_height else 0.0
    result.fwhr = fwhr
    result.fwhr_score = _closeness_score(fwhr, ideal=1.9, tolerance=0.35)

    # ------------------------------------------------------------------
    # 4. Нижняя треть лица (Lower Third Ratio), идеал 1:2 (0.5)
    # ------------------------------------------------------------------
    upper_part = _dist(corrected_pts[SUBNASALE], corrected_pts[UPPER_LIP_TOP])
    lower_part = _dist(corrected_pts[UPPER_LIP_TOP], corrected_pts[CHIN_BOTTOM])
    lower_third_ratio = upper_part / lower_part if lower_part else 0.0
    result.lower_third_ratio = lower_third_ratio
    result.lower_third_score = _closeness_score(lower_third_ratio, ideal=0.5, tolerance=0.18)

    # ------------------------------------------------------------------
    # 5. Расстояние между глазами (IPD), идеал ~0.45 от ширины лица
    # ------------------------------------------------------------------
    try:
        ipd = _dist(corrected_pts[LEFT_IRIS_CENTER], corrected_pts[RIGHT_IRIS_CENTER])
    except IndexError:
        ipd = _dist(corrected_pts[LEFT_EYE_INNER], corrected_pts[RIGHT_EYE_INNER]) * 1.6
    ipd_ratio = ipd / face_width if face_width else 0.0
    result.ipd_ratio = ipd_ratio
    result.ipd_score = _closeness_score(ipd_ratio, ideal=0.45, tolerance=0.06)

    # ------------------------------------------------------------------
    # 6. Состояние кожи (Skin Condition), Laplacian variance
    # ------------------------------------------------------------------
    result.skin_score = _analyze_skin(front_img, front_pts)

    # ------------------------------------------------------------------
    # 7. Структура костей (на основе FWHR + симметрии овала лица + профиль)
    # ------------------------------------------------------------------
    jaw_angle_score = _analyze_jaw_structure(profile_pts)
    result.bone_score = _clip(0.5 * result.fwhr_score + 0.5 * jaw_angle_score)

    # ------------------------------------------------------------------
    # 8. Гармония черт (среднее по золотому сечению пропорций)
    # ------------------------------------------------------------------
    result.harmony_score = _clip(
        0.30 * result.fwhr_score
        + 0.30 * result.lower_third_score
        + 0.20 * result.ipd_score
        + 0.20 * result.symmetry_score
    )

    # ------------------------------------------------------------------
    # 9. Итоговый балл PSL (0.0 - 10.0), взвешенная формула
    # ------------------------------------------------------------------
    weighted = (
        0.25 * result.symmetry_score
        + 0.20 * result.bone_score
        + 0.15 * result.fwhr_score
        + 0.10 * result.lower_third_score
        + 0.10 * result.ipd_score
        + 0.10 * result.skin_score
        + 0.10 * result.harmony_score
    )
    psl_score = weighted / 10.0
    result.psl_score = round(_clip(psl_score, 0.0, 10.0), 1)

    # ------------------------------------------------------------------
    # 10. Potential Score (идеальная кожа, минимальный % жира => harmony +)
    # ------------------------------------------------------------------
    potential_skin = 96.0
    potential_harmony = _clip(result.harmony_score * 1.08)
    potential_weighted = (
        0.25 * _clip(result.symmetry_score * 1.05)
        + 0.20 * result.bone_score
        + 0.15 * result.fwhr_score
        + 0.10 * result.lower_third_score
        + 0.10 * result.ipd_score
        + 0.10 * potential_skin
        + 0.10 * potential_harmony
    )
    result.potential_score = round(_clip(potential_weighted / 10.0, 0.0, 10.0), 1)
    if result.potential_score < result.psl_score:
        result.potential_score = result.psl_score

    result.tier = score_to_tier(result.psl_score)
    result.advice_soft, result.advice_hard = _generate_advice(result)
    result.success = True
    return result


# ---------------------------------------------------------------------------
# Анализ кожи
# ---------------------------------------------------------------------------
def _region_mask(points: np.ndarray, indices: list[int], shape) -> np.ndarray:
    h, w = shape[:2]
    poly = np.array([[points[i][0], points[i][1]] for i in indices], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    return mask


def _analyze_skin(image_bgr: np.ndarray, points: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variances = []
    for region in (LEFT_CHEEK_REGION, RIGHT_CHEEK_REGION):
        mask = _region_mask(points, region, image_bgr.shape)
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0 or h == 0:
            continue
        roi_gray = gray[y:y + h, x:x + w]
        roi_mask = mask[y:y + h, x:x + w]
        if roi_gray.size == 0:
            continue
        roi_gray = cv2.bitwise_and(roi_gray, roi_gray, mask=roi_mask)
        roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
        lap = cv2.Laplacian(roi_gray, cv2.CV_64F)
        variances.append(float(lap.var()))

    if not variances:
        return 70.0

    avg_var = float(np.mean(variances))
    # Чем выше дисперсия Лапласиана, тем больше неровностей/текстуры/акне.
    # Эмпирическая нормализация: var ~ 0 -> идеально гладкая кожа (100%),
    # var >= 350 -> сильно неровная кожа (минимум ~30%).
    smoothness = 100.0 - _clip(avg_var / 350.0 * 70.0, 0.0, 70.0)
    return _clip(smoothness)


# ---------------------------------------------------------------------------
# Анализ структуры челюсти в профиль
# ---------------------------------------------------------------------------
def _analyze_jaw_structure(profile_pts: np.ndarray) -> float:
    try:
        nose_tip = profile_pts[NOSE_TIP]
        chin = profile_pts[CHIN_BOTTOM]
        jaw_mid = profile_pts[172] if profile_pts.shape[0] > 172 else profile_pts[CHIN_BOTTOM]
        ear_approx = profile_pts[234]
    except IndexError:
        return 70.0

    # Угол челюсти: чем ближе к выраженному (более вертикальному) углу
    # между ухом/скулой и подбородком, тем выше оценка ("чёткая челюсть").
    v1 = jaw_mid[:2] - ear_approx[:2]
    v2 = chin[:2] - jaw_mid[:2]
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 70.0
    cos_angle = float(np.dot(v1, v2) / (norm1 * norm2))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_deg = math.degrees(math.acos(cos_angle))
    # Идеал около 120-130 градусов угла нижней челюсти (gonial angle)
    return _closeness_score(angle_deg, ideal=125.0, tolerance=25.0)


# ---------------------------------------------------------------------------
# E-line (линия Риккетса) для профиля
# ---------------------------------------------------------------------------
def get_e_line_points(profile_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (точка кончика носа, точка кончика подбородка) для E-line."""
    return profile_pts[NOSE_TIP][:2], profile_pts[CHIN_BOTTOM][:2]


# ---------------------------------------------------------------------------
# Советы softmaxxing / hardmaxxing
# ---------------------------------------------------------------------------
def _generate_advice(r: FaceAnalysisResult) -> tuple[list, list]:
    soft, hard = [], []

    if r.skin_score < 75:
        soft.append("Уход за кожей: умывание 2 раза в день, увлажняющий крем, SPF днём, ретинол/ниацинамид на ночь.")
        soft.append("Рассмотрите консультацию дерматолога при акне/постакне.")
    else:
        soft.append("Кожа в хорошем состоянии — поддерживайте текущий уход (очищение + SPF).")

    if r.symmetry_score < 80:
        soft.append("Симметрию визуально улучшит правильно подобранная стрижка и форма бровей.")

    if r.fwhr_score < 70:
        soft.append("Снижение % жира на лице (спорт, дефицит калорий) может визуально увеличить FWHR и чёткость скул.")
        hard.append("При выраженной диспропорции скул/нижней челюсти — консультация с челюстно-лицевым хирургом (имплантация скул/genioplasty).")

    if r.lower_third_score < 70:
        hard.append("При сильном дисбалансе нижней трети лица рассматривается ортодонтия или ортогнатическая хирургия (после консультации специалиста).")

    if r.bone_score < 70:
        soft.append("Чёткость челюсти улучшают: mewing (с осторожностью, без научного консенсуса), снижение % жира, силовые тренировки шеи.")
        hard.append("Контурная пластика подбородка/челюсти (chin filler, jaw implants) — только после консультации хирурга.")

    soft.append("Спорт (особенно силовые/осанка) и качественный сон напрямую влияют на восприятие внешности.")
    hard.append("Любые хирургические/инвазивные процедуры — только после очной консультации с квалифицированным врачом.")

    return soft, hard
