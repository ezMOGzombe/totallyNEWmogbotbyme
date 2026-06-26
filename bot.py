"""
bot.py
Обработчик aiogram 3.x: middleware белого списка, FSM, команды /start,
/admin, /settings, /psl, /me, обработка фото и отправка результата PSL Rating.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    TelegramObject,
)

from analyzer import analyze_face, FaceAnalysisResult, score_to_tier
from image_generator import generate_infographic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
ADMIN_ID = 1178284542
BASE_DIR = Path(__file__).resolve().parent
ALLOWED_USERS_FILE = BASE_DIR / "allowed_users.json"
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

BACK_BUTTON_TEXT = "◀️ Назад"
USERS_BUTTON_TEXT = "Пользователи"
QUOTA_BUTTON_TEXT = "Настроить кол-во запросов"
EDIT_QUOTA_BUTTON_TEXT = "Настроить"
REMOVE_BUTTON_TEXT = "Удалить пользователя"

DEFAULT_GRANTED_REQUESTS = 2
ADMIN_GRANTED_REQUESTS = 500

STRICTNESS_MIN = -90.0
STRICTNESS_MAX = 90.0

# Варианты срока действия доступа: (подпись, код, дней | None = навсегда)
DURATION_OPTIONS = [
    ("3 дня", "3d", 3),
    ("Неделя", "7d", 7),
    ("2 недели", "14d", 14),
    ("Месяц", "30d", 30),
    ("3 месяца", "90d", 90),
    ("Навсегда", "perm", None),
]


# ---------------------------------------------------------------------------
# Белый список пользователей + лимиты запросов + настройки строгости
# ---------------------------------------------------------------------------
def _today_str() -> str:
    return date.today().isoformat()


def _default_user_entry(user_id: int) -> dict:
    granted = ADMIN_GRANTED_REQUESTS if user_id == ADMIN_ID else DEFAULT_GRANTED_REQUESTS
    return {
        "id": user_id,
        "granted": granted,
        "used_today": 0,
        "last_reset": _today_str(),
        "expires_at": None,
    }


def _read_db() -> dict:
    """Читает базу пользователей, автоматически мигрируя старый формат
    {"allowed_users": [id, id, ...]} в новый {"users": [{...}, ...], "settings": {...}}."""
    if not ALLOWED_USERS_FILE.exists():
        data = {"users": [_default_user_entry(ADMIN_ID)], "settings": {"strictness_pct": 0.0}}
        _write_db(data)
        logger.info("Создан allowed_users.json с ADMIN_ID=%s", ADMIN_ID)
        return data

    try:
        raw = json.loads(ALLOWED_USERS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        raw = {}

    needs_write = False

    if "users" not in raw and "allowed_users" in raw:
        # Миграция со старого простого списка ID.
        migrated_users = [_default_user_entry(uid) for uid in raw.get("allowed_users", [])]
        raw = {"users": migrated_users}
        needs_write = True
        logger.info("allowed_users.json мигрирован в новый формат с лимитами запросов.")

    raw.setdefault("users", [])

    if not any(u.get("id") == ADMIN_ID for u in raw["users"]):
        raw["users"].append(_default_user_entry(ADMIN_ID))
        needs_write = True

    for u in raw["users"]:
        if "expires_at" not in u:
            u["expires_at"] = None
            needs_write = True

    if "settings" not in raw:
        raw["settings"] = {"strictness_pct": 0.0}
        needs_write = True
    elif "strictness_pct" not in raw["settings"]:
        raw["settings"]["strictness_pct"] = 0.0
        needs_write = True

    if needs_write:
        _write_db(raw)

    return raw


def _write_db(data: dict) -> None:
    ALLOWED_USERS_FILE.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def ensure_allowed_users_file() -> None:
    """Создаёт allowed_users.json при отсутствии / мигрирует старый формат."""
    _read_db()


def get_all_users() -> list[dict]:
    return _read_db().get("users", [])


def find_user(user_id: int) -> Optional[dict]:
    for u in get_all_users():
        if u.get("id") == user_id:
            return u
    return None


def _is_active(user: dict) -> bool:
    expires_at = user.get("expires_at")
    if not expires_at:
        return True
    try:
        return date.fromisoformat(expires_at) >= date.today()
    except ValueError:
        return True


def is_allowed(user_id: int) -> bool:
    user = find_user(user_id)
    return user is not None and _is_active(user)


def add_allowed_user(user_id: int, expires_at: Optional[str] = None) -> None:
    """Добавляет пользователя в белый список (или обновляет срок, если уже есть)."""
    data = _read_db()
    existing = next((u for u in data["users"] if u.get("id") == user_id), None)
    if existing is not None:
        existing["expires_at"] = expires_at
    else:
        entry = _default_user_entry(user_id)
        entry["expires_at"] = expires_at
        data["users"].append(entry)
    _write_db(data)


def remove_allowed_user(user_id: int) -> bool:
    """Удаляет пользователя из белого списка. Возвращает True, если был найден."""
    data = _read_db()
    before = len(data["users"])
    data["users"] = [u for u in data["users"] if u.get("id") != user_id]
    changed = len(data["users"]) < before
    if changed:
        _write_db(data)
    return changed


def set_user_quota(user_id: int, granted: int) -> bool:
    """Устанавливает выданное количество запросов. Возвращает False, если
    пользователя нет в белом списке."""
    data = _read_db()
    for u in data["users"]:
        if u.get("id") == user_id:
            u["granted"] = granted
            _write_db(data)
            return True
    return False


def _apply_daily_reset(user: dict) -> dict:
    if user.get("last_reset") != _today_str():
        user["last_reset"] = _today_str()
        user["used_today"] = 0
    return user


def get_remaining_requests(user_id: int) -> int:
    user = find_user(user_id)
    if user is None:
        return 0
    user = _apply_daily_reset(user)
    remaining = user.get("granted", 0) - user.get("used_today", 0)
    return max(0, remaining)


def consume_request(user_id: int) -> bool:
    """Списывает один запрос у пользователя. Возвращает True при успехе."""
    data = _read_db()
    for u in data["users"]:
        if u.get("id") == user_id:
            _apply_daily_reset(u)
            if u.get("granted", 0) - u.get("used_today", 0) <= 0:
                _write_db(data)
                return False
            u["used_today"] = u.get("used_today", 0) + 1
            _write_db(data)
            return True
    return False


def get_strictness_pct() -> float:
    return float(_read_db().get("settings", {}).get("strictness_pct", 0.0))


def set_strictness_pct(value: float) -> float:
    value = max(STRICTNESS_MIN, min(STRICTNESS_MAX, value))
    data = _read_db()
    data.setdefault("settings", {})["strictness_pct"] = value
    _write_db(data)
    return value


def adjust_strictness_pct(delta: float) -> float:
    return set_strictness_pct(get_strictness_pct() + delta)


# ---------------------------------------------------------------------------
# Middleware: глобальная проверка белого списка
# ---------------------------------------------------------------------------
class WhitelistMiddleware(BaseMiddleware):
    """Полностью игнорирует любые сообщения/callback'и не из белого списка."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        if user is None:
            return None
        if not is_allowed(user.id):
            # Полностью игнорируем — никакого ответа.
            return None
        return await handler(event, data)


# ---------------------------------------------------------------------------
# FSM состояния
# ---------------------------------------------------------------------------
class AdminStates(StatesGroup):
    waiting_for_user_id = State()
    waiting_for_quota_user_id = State()
    waiting_for_quota_value = State()
    waiting_for_remove_user_id = State()


class PSLStates(StatesGroup):
    waiting_for_front = State()
    waiting_for_profile = State()


# ---------------------------------------------------------------------------
# Клавиатуры
# ---------------------------------------------------------------------------
def back_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=BACK_BUTTON_TEXT)]],
        resize_keyboard=True,
    )


def admin_main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=USERS_BUTTON_TEXT), KeyboardButton(text=QUOTA_BUTTON_TEXT)],
            [KeyboardButton(text=REMOVE_BUTTON_TEXT)],
            [KeyboardButton(text=BACK_BUTTON_TEXT)],
        ],
        resize_keyboard=True,
    )


def quota_edit_inline_keyboard(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=EDIT_QUOTA_BUTTON_TEXT, callback_data=f"quota_edit:{user_id}")]
        ]
    )


def duration_keyboard(user_id: int) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for label, code, _days in DURATION_OPTIONS:
        row.append(InlineKeyboardButton(text=label, callback_data=f"adddur:{user_id}:{code}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _strictness_description(pct: float) -> str:
    if abs(pct) < 0.05:
        return "стандартная (0%)"
    if pct > 0:
        return f"строже стандартной на {pct:.0f}% (итоговые оценки ниже)"
    return f"мягче стандартной на {abs(pct):.0f}% (итоговые оценки выше)"


def settings_text(pct: float) -> str:
    return (
        "⚖️ <b>Настройка строгости оценки внешности</b>\n\n"
        f"Текущий режим: <b>{_strictness_description(pct)}</b>\n\n"
        "Кнопки ниже изменяют итоговый PSL-балл, который получают все пользователи."
    )


def settings_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="➖ Облегчить на 10%", callback_data="strict:-10"),
                InlineKeyboardButton(text="➖ Облегчить на 1%", callback_data="strict:-1"),
            ],
            [InlineKeyboardButton(text="↺ Сбросить к стандартной", callback_data="strict:reset")],
            [
                InlineKeyboardButton(text="➕ Усложнить на 1%", callback_data="strict:1"),
                InlineKeyboardButton(text="➕ Усложнить на 10%", callback_data="strict:10"),
            ],
        ]
    )


ADMIN_WAITING_ID_TEXT = "Ожидаю Telegram ID человека, которого нужно добавить в список доступа."


WELCOME_TEXT = (
    "👋 Добро пожаловать в <b>PSL Rating</b> — локальный анализ внешности по шкале looksmaxxing.\n\n"
    "Доступные команды:\n"
    "/psl — оценка внешности\n"
    "/me — доступные запросы"
)

ADMIN_RIGHTS_TEXT = "Вы имеете права администратора."


# ---------------------------------------------------------------------------
# Роутер
# ---------------------------------------------------------------------------
router = Router(name="psl_router")


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(WELCOME_TEXT, reply_markup=ReplyKeyboardRemove())
    if message.from_user.id == ADMIN_ID:
        await message.answer(ADMIN_RIGHTS_TEXT)


# ---------------------------------------------------------------------------
# /me — доступные запросы пользователя
# ---------------------------------------------------------------------------
@router.message(Command("me"))
async def cmd_me(message: Message) -> None:
    user = find_user(message.from_user.id)
    if user is None:
        return

    remaining = get_remaining_requests(message.from_user.id)
    granted = user.get("granted", 0)
    expires_at = user.get("expires_at")
    expiry_text = "без ограничения по сроку" if not expires_at else f"до {expires_at}"

    await message.answer(
        "📊 <b>Ваш профиль</b>\n\n"
        f"Доступно запросов сегодня: <b>{remaining}</b> из {granted}\n"
        f"Срок доступа: {expiry_text}"
    )


# ---------------------------------------------------------------------------
# /admin — админ-панель добавления/удаления пользователей и управления лимитами
# ---------------------------------------------------------------------------
@router.message(Command("admin"))
async def cmd_admin(message: Message, state: FSMContext) -> None:
    if message.from_user.id != ADMIN_ID:
        return  # команда доступна только администратору

    await state.set_state(AdminStates.waiting_for_user_id)
    await message.answer(ADMIN_WAITING_ID_TEXT, reply_markup=admin_main_keyboard())


@router.message(AdminStates.waiting_for_user_id, F.text == BACK_BUTTON_TEXT)
async def admin_back(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(WELCOME_TEXT, reply_markup=ReplyKeyboardRemove())


@router.message(AdminStates.waiting_for_user_id, F.text == USERS_BUTTON_TEXT)
async def admin_list_users(message: Message, state: FSMContext) -> None:
    users = get_all_users()
    lines = []
    for i, u in enumerate(users, start=1):
        expires_at = u.get("expires_at")
        expiry_str = "навсегда" if not expires_at else f"до {expires_at}"
        active_note = "" if _is_active(u) else " ⛔ истёк"
        lines.append(f"{i}. <code>{u['id']}</code> - {u.get('granted', 0)} ({expiry_str}){active_note}")
    text = "<b>Пользователи с доступом:</b>\n" + "\n".join(lines) if lines else "Список пользователей пуст."
    await message.answer(text, reply_markup=admin_main_keyboard())


@router.message(AdminStates.waiting_for_user_id, F.text == QUOTA_BUTTON_TEXT)
async def admin_quota_start(message: Message, state: FSMContext) -> None:
    await state.set_state(AdminStates.waiting_for_quota_user_id)
    await message.answer(
        "Ожидаю Telegram ID пользователя, чтобы настроить ему количество запросов.",
        reply_markup=back_keyboard(),
    )


@router.message(AdminStates.waiting_for_user_id, F.text == REMOVE_BUTTON_TEXT)
async def admin_remove_start(message: Message, state: FSMContext) -> None:
    await state.set_state(AdminStates.waiting_for_remove_user_id)
    await message.answer(
        "Ожидаю Telegram ID пользователя, которого нужно удалить из белого списка.",
        reply_markup=back_keyboard(),
    )


@router.message(AdminStates.waiting_for_user_id, F.text)
async def admin_receive_id(message: Message, state: FSMContext) -> None:
    text = message.text.strip()
    if not text.isdigit():
        await message.answer(
            "Пожалуйста, введите корректный числовой ID или нажмите кнопку 'Назад'.",
            reply_markup=admin_main_keyboard(),
        )
        return

    new_id = int(text)
    await message.answer(
        f"Выберите срок доступа для пользователя <code>{new_id}</code>:",
        reply_markup=duration_keyboard(new_id),
    )


@router.callback_query(F.data.startswith("adddur:"))
async def duration_callback(callback: CallbackQuery) -> None:
    if callback.from_user.id != ADMIN_ID:
        await callback.answer()
        return

    _, user_id_str, code = callback.data.split(":")
    target_id = int(user_id_str)
    days = next((d for _, c, d in DURATION_OPTIONS if c == code), None)
    label = next((l for l, c, _ in DURATION_OPTIONS if c == code), code)

    expires_at = None
    if days is not None:
        expires_at = (date.today() + timedelta(days=days)).isoformat()

    add_allowed_user(target_id, expires_at=expires_at)

    expiry_note = "без ограничения по сроку" if expires_at is None else f"до {expires_at}"
    await callback.message.edit_text(
        f"✅ Пользователь <code>{target_id}</code> добавлен в белый список ({label}, {expiry_note})."
    )
    await callback.answer("Добавлено")


# ---------------------------------------------------------------------------
# Удаление пользователя из белого списка
# ---------------------------------------------------------------------------
@router.message(AdminStates.waiting_for_remove_user_id, F.text == BACK_BUTTON_TEXT)
async def remove_id_back(message: Message, state: FSMContext) -> None:
    await state.set_state(AdminStates.waiting_for_user_id)
    await message.answer(ADMIN_WAITING_ID_TEXT, reply_markup=admin_main_keyboard())


@router.message(AdminStates.waiting_for_remove_user_id, F.text)
async def remove_receive_id(message: Message, state: FSMContext) -> None:
    text = message.text.strip()
    if not text.isdigit():
        await message.answer(
            "Пожалуйста, введите корректный числовой ID или нажмите кнопку 'Назад'.",
            reply_markup=back_keyboard(),
        )
        return

    target_id = int(text)
    if target_id == ADMIN_ID:
        await message.answer(
            "⛔ Невозможно удалить администратора из белого списка.",
            reply_markup=back_keyboard(),
        )
        return

    removed = remove_allowed_user(target_id)
    await state.set_state(AdminStates.waiting_for_user_id)
    if removed:
        await message.answer(
            f"🗑 Пользователь <code>{target_id}</code> удалён из белого списка.",
            reply_markup=admin_main_keyboard(),
        )
    else:
        await message.answer(
            f"Пользователь <code>{target_id}</code> не найден в белом списке.",
            reply_markup=admin_main_keyboard(),
        )


# ---------------------------------------------------------------------------
# Настройка количества выданных запросов
# ---------------------------------------------------------------------------
@router.message(AdminStates.waiting_for_quota_user_id, F.text == BACK_BUTTON_TEXT)
async def quota_id_back(message: Message, state: FSMContext) -> None:
    await state.set_state(AdminStates.waiting_for_user_id)
    await message.answer(ADMIN_WAITING_ID_TEXT, reply_markup=admin_main_keyboard())


@router.message(AdminStates.waiting_for_quota_user_id, F.text)
async def quota_receive_id(message: Message, state: FSMContext) -> None:
    text = message.text.strip()
    if not text.isdigit():
        await message.answer(
            "Пожалуйста, введите корректный числовой ID или нажмите кнопку 'Назад'.",
            reply_markup=back_keyboard(),
        )
        return

    target_id = int(text)
    user = find_user(target_id)
    if user is None:
        await message.answer(
            f"Пользователь с ID {target_id} не найден в белом списке. "
            f"Сначала добавьте его через /admin.",
            reply_markup=back_keyboard(),
        )
        return

    granted = user.get("granted", 0)
    await message.answer(
        f"Пользователю {target_id} выдано {granted} запрос(ов) в день.",
        reply_markup=quota_edit_inline_keyboard(target_id),
    )


@router.callback_query(F.data.startswith("quota_edit:"))
async def quota_edit_callback(callback: CallbackQuery, state: FSMContext) -> None:
    if callback.from_user.id != ADMIN_ID:
        await callback.answer()
        return

    target_id = int(callback.data.split(":")[1])
    await state.set_state(AdminStates.waiting_for_quota_value)
    await state.update_data(quota_target_id=target_id)

    user = find_user(target_id)
    current_granted = user.get("granted", 0) if user else 0
    await callback.message.edit_text(
        f"Пользователю {target_id} выдано {current_granted} запрос(ов) в день.\n\n"
        f"✏️ Ожидаю новое количество запросов..."
    )
    await callback.message.answer(
        f"Введите новое количество запросов для пользователя {target_id}:",
        reply_markup=back_keyboard(),
    )
    await callback.answer()


@router.message(AdminStates.waiting_for_quota_value, F.text == BACK_BUTTON_TEXT)
async def quota_value_back(message: Message, state: FSMContext) -> None:
    await state.set_state(AdminStates.waiting_for_user_id)
    await message.answer(ADMIN_WAITING_ID_TEXT, reply_markup=admin_main_keyboard())


@router.message(AdminStates.waiting_for_quota_value, F.text)
async def quota_value_receive(message: Message, state: FSMContext) -> None:
    text = message.text.strip()
    if not text.isdigit():
        await message.answer(
            "Пожалуйста, введите корректное число запросов или нажмите кнопку 'Назад'.",
            reply_markup=back_keyboard(),
        )
        return

    data = await state.get_data()
    target_id = data.get("quota_target_id")
    new_granted = int(text)
    set_user_quota(target_id, new_granted)

    await state.set_state(AdminStates.waiting_for_user_id)
    await message.answer(
        f"Пользователю {target_id} выдано {new_granted} запрос(ов) в день.",
        reply_markup=admin_main_keyboard(),
    )


# ---------------------------------------------------------------------------
# /settings — настройка строгости оценки внешности (только админ)
# ---------------------------------------------------------------------------
@router.message(Command("settings"))
async def cmd_settings(message: Message) -> None:
    if message.from_user.id != ADMIN_ID:
        return

    pct = get_strictness_pct()
    await message.answer(settings_text(pct), reply_markup=settings_keyboard())


@router.callback_query(F.data.startswith("strict:"))
async def strict_callback(callback: CallbackQuery) -> None:
    if callback.from_user.id != ADMIN_ID:
        await callback.answer()
        return

    action = callback.data.split(":", 1)[1]
    if action == "reset":
        pct = set_strictness_pct(0.0)
        toast = "Строгость сброшена к стандартной."
    else:
        pct = adjust_strictness_pct(float(action))
        toast = "Строгость обновлена."

    await callback.message.edit_text(settings_text(pct), reply_markup=settings_keyboard())
    await callback.answer(toast)


# ---------------------------------------------------------------------------
# /psl — оценка внешности
# ---------------------------------------------------------------------------
PSL_FRONT_PROMPT = (
    "Для оценки вашей внешности отправьте две фотографии по очереди.\n\n"
    "1️⃣ Сначала отправьте фото в АНФАС (лицо прямо перед камерой).\n\n"
    "⚠️ ВАЖНО: Для точного анализа необходим хороший свет, качественная камера, "
    "отсутствие фильтров, очков и волос на лбу."
)

PSL_PROFILE_PROMPT = (
    "Отлично! Теперь отправьте второе фото в ПРОФИЛЬ "
    "(лицо сбоку, чтобы была видна челюсть)."
)


@router.message(Command("psl"))
async def cmd_psl(message: Message, state: FSMContext) -> None:
    remaining = get_remaining_requests(message.from_user.id)
    if remaining <= 0:
        await message.answer(
            "⛔ У вас не осталось доступных запросов на сегодня. "
            "Лимит обновится завтра, либо обратитесь к администратору.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return

    await state.set_state(PSLStates.waiting_for_front)
    await message.answer(PSL_FRONT_PROMPT, reply_markup=ReplyKeyboardRemove())


@router.message(PSLStates.waiting_for_front, F.photo)
async def psl_receive_front(message: Message, state: FSMContext, bot: Bot) -> None:
    user_id = message.from_user.id
    front_path = TEMP_DIR / f"{user_id}_front.jpg"
    await bot.download(message.photo[-1].file_id, destination=front_path)

    await state.update_data(front_path=str(front_path))
    await state.set_state(PSLStates.waiting_for_profile)
    await message.answer(PSL_PROFILE_PROMPT)


@router.message(PSLStates.waiting_for_front)
async def psl_front_invalid(message: Message) -> None:
    await message.answer("Пожалуйста, отправьте фотографию в АНФАС.")


def _apply_strictness(result: FaceAnalysisResult) -> None:
    """Применяет текущую настройку строгости (/settings) к итоговому баллу."""
    pct = get_strictness_pct()
    if abs(pct) < 1e-9:
        return

    factor = max(0.0, 1.0 - pct / 100.0)
    result.psl_score = round(max(0.0, min(10.0, result.psl_score * factor)), 1)
    result.potential_score = round(max(0.0, min(10.0, result.potential_score * factor)), 1)
    if result.potential_score < result.psl_score:
        result.potential_score = result.psl_score
    result.tier = score_to_tier(result.psl_score)


@router.message(PSLStates.waiting_for_profile, F.photo)
async def psl_receive_profile(message: Message, state: FSMContext, bot: Bot) -> None:
    user_id = message.from_user.id
    profile_path = TEMP_DIR / f"{user_id}_profile.jpg"
    await bot.download(message.photo[-1].file_id, destination=profile_path)

    data = await state.get_data()
    front_path = data.get("front_path")
    await state.clear()

    await message.answer(
        "🔬 Запускаю локальный анализ лица (OpenCV + MediaPipe)... Это может занять до минуты.",
        reply_markup=ReplyKeyboardRemove(),
    )

    result = analyze_face(front_path, str(profile_path))

    if not result.success:
        await message.answer(f"❌ Не удалось выполнить анализ: {result.error}\nПопробуйте снова: /psl")
        return

    _apply_strictness(result)

    output_path = TEMP_DIR / f"{user_id}_result.png"
    try:
        generate_infographic(front_path, str(profile_path), result, str(output_path))
    except Exception:
        logger.exception("Ошибка генерации инфографики")
        await message.answer("❌ Произошла ошибка при генерации инфографики. Попробуйте снова: /psl")
        return

    consume_request(user_id)

    await message.answer_photo(FSInputFile(output_path))
    await message.answer(_build_report_text(result), parse_mode="HTML")

    for path in (front_path, str(profile_path), str(output_path)):
        try:
            os.remove(path)
        except OSError:
            pass


@router.message(PSLStates.waiting_for_profile)
async def psl_profile_invalid(message: Message) -> None:
    await message.answer("Пожалуйста, отправьте фотографию в ПРОФИЛЬ.")


# ---------------------------------------------------------------------------
# Текстовый отчёт
# ---------------------------------------------------------------------------
def _build_report_text(r: FaceAnalysisResult) -> str:
    soft = "\n".join(f"• {tip}" for tip in r.advice_soft)
    hard = "\n".join(f"• {tip}" for tip in r.advice_hard)

    tilt_note = ""
    if r.tilt_detected:
        tilt_note = f"\n<i>Обнаружен наклон головы ({r.tilt_angle:.1f}°), координаты скорректированы.</i>\n"

    return (
        f"<b>📊 ОТЧЁТ PSL RATING</b>\n"
        f"{tilt_note}\n"
        f"<b>Итоговый балл:</b> {r.psl_score:.1f} / 10.0 — <b>{r.tier}</b>\n"
        f"<b>Potential Score:</b> {r.potential_score:.1f} / 10.0\n\n"
        f"<b>Пропорции:</b>\n"
        f"• Симметрия лица: {r.symmetry_score:.0f}%\n"
        f"• FWHR: {r.fwhr:.2f} (идеал 1.80–2.00) — {r.fwhr_score:.0f}%\n"
        f"• Нижняя треть лица: {r.lower_third_ratio:.2f} (идеал 0.50, т.е. 1:2) — {r.lower_third_score:.0f}%\n"
        f"• IPD (межзрачковое расстояние): {r.ipd_ratio:.2f} (идеал ~0.45) — {r.ipd_score:.0f}%\n"
        f"• Структура костей: {r.bone_score:.0f}%\n"
        f"• Состояние кожи: {r.skin_score:.0f}%\n"
        f"• Гармония черт: {r.harmony_score:.0f}%\n\n"
        f"<b>🧴 Softmaxxing:</b>\n{soft}\n\n"
        f"<b>🛠 Hardmaxxing:</b>\n{hard}"
    )


def setup_dispatcher(dp: Dispatcher) -> None:
    ensure_allowed_users_file()
    dp.message.middleware(WhitelistMiddleware())
    dp.callback_query.middleware(WhitelistMiddleware())
    dp.include_router(router)
