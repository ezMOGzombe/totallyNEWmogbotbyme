"""
main.py
Точка входа: загрузка конфигурации, инициализация, запуск веб-сервера
для предотвращения засыпания и запуск Telegram-бота PSL Rating.
"""
import asyncio
import logging
import os

import aiohttp
from aiohttp import web
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from dotenv import load_dotenv

from bot import setup_dispatcher

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Отвечает Render и пинг-сервисам, что бот активен
async def handle_root(request):
    return web.Response(text="PSL Rating Bot работает!")


# ✅ НОВОЕ: фоновый self-ping, чтобы Render не засыпал
async def self_ping(url: str, interval: int = 50):
    """
    Пингует собственный URL бота каждые `interval` секунд,
    чтобы предотвратить переход Render в спящий режим.
    """
    await asyncio.sleep(15)  # Ждём, пока веб-сервер полностью поднимется
    logger.info(f"Self-ping активен → {url} (каждые {interval} сек.)")
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    logger.info(f"Self-ping ✅ статус: {resp.status}")
            except Exception as e:
                logger.warning(f"Self-ping ⚠️ ошибка: {e}")
            await asyncio.sleep(interval)


async def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не найден в .env")

    # ✅ НОВОЕ: URL вашего сервиса на Render (задаётся в Environment Variables)
    # Пример: https://your-app-name.onrender.com
    render_url = os.getenv("RENDER_URL")

    bot = Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    setup_dispatcher(dp)

    # --- ВЕБ-СЕРВЕР ДЛЯ RENDER ---
    port = int(os.getenv("PORT", 10000))

    app = web.Application()
    app.router.add_get("/", handle_root)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"Веб-сервер запущен на порту {port}")

    # ✅ НОВОЕ: запускаем self-ping как фоновую задачу asyncio
    if render_url:
        asyncio.create_task(self_ping(render_url))
    else:
        logger.warning(
            "RENDER_URL не задан — self-ping отключён. "
            "Добавьте переменную окружения RENDER_URL в настройках Render."
        )

    # --- ЗАПУСК БОТА ---
    logger.info("PSL Rating Bot запущен.")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")
