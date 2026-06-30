import os
import json
import asyncio
import re
from datetime import datetime, time, timedelta
from pathlib import Path

import pytz
from telegram import Update, Bot
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from openai import AsyncOpenAI


# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN_SLEEP"]
OPENROUTER_API_KEY = os.environ["GEMINI_API_KEY"]
CHAT_ID = int(os.environ["CHAT_ID"])

TIMEZONE = pytz.timezone(os.environ.get("TIMEZONE", "Europe/Minsk"))

DATA_DIR = Path(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "/data"))
LOG_FILE = DATA_DIR / "sleep_log.json"
STATE_FILE = DATA_DIR / "sleep_state.json"

SCHEDULER_TICK_SECONDS = 30
MISSED_GRACE_MINUTES = 20

MODEL_NAME = "google/gemini-2.5-flash"
TELEGRAM_MAX_LEN = 4000

DEFAULT_SLEEP_TIME = "23:40"
EVENING_REMINDER_TIME = time(21, 0)
MORNING_CHECK_TIME = time(9, 0)


# ─── КЛИЕНТ ───────────────────────────────────────────────────────────────────

client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


# ─── ФАЙЛЫ ────────────────────────────────────────────────────────────────────

def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path, default):
    ensure_data_dir()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            backup = path.with_suffix(".broken.json")
            path.rename(backup)
    return default


def save_json(path, data):
    ensure_data_dir()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_log():
    return load_json(LOG_FILE, [])


def save_log(data):
    save_json(LOG_FILE, data)


def load_state():
    default = {
        "sleep_time": DEFAULT_SLEEP_TIME,
        "daily": {
            "date": "",
            "evening_sent": False,
            "morning_sent": False,
            "morning_answered": False
        }
    }
    state = load_json(STATE_FILE, default)
    if "sleep_time" not in state:
        state["sleep_time"] = DEFAULT_SLEEP_TIME
    if "daily" not in state:
        state["daily"] = default["daily"]
    return state


def save_state(data):
    save_json(STATE_FILE, data)


# ─── ВРЕМЯ ────────────────────────────────────────────────────────────────────

def now_dt():
    return datetime.now(TIMEZONE)


def today_key():
    return now_dt().strftime("%Y-%m-%d")


def combine_tz(date_str, hhmm):
    naive = datetime.strptime(f"{date_str} {hhmm}", "%Y-%m-%d %H:%M")
    return TIMEZONE.localize(naive)


def parse_iso(value):
    return datetime.fromisoformat(value)


def minutes_late(due_dt):
    return (now_dt() - due_dt).total_seconds() / 60


# ─── ПАРСИНГ ВРЕМЕНИ СНА ─────────────────────────────────────────────────────

def parse_sleep_time(text):
    """Ищет время в тексте вида 23:40, 00:30, в 22 и т.д."""
    normalized = text.lower().replace(".", ":")

    patterns = [
        r"\b(\d{1,2}):(\d{2})\b",
        r"в\s+(\d{1,2})\s*(?:час|ч\b)",
        r"\b(\d{1,2})\s*(?:час|ч\b)",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            groups = match.groups()
            hour = int(groups[0])
            minute = int(groups[1]) if len(groups) > 1 and groups[1] else 0
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"

    return None


def wants_to_change_time(text):
    triggers = [
        "перенеси", "измени время", "лягу в", "буду ложиться в",
        "поставь", "сдвинь", "новое время", "время сна"
    ]
    normalized = text.lower()
    return any(t in normalized for t in triggers)


# ─── АНАЛИЗ ОТВЕТА УТРОМ ─────────────────────────────────────────────────────

async def analyze_morning_answer(answer, sleep_target):
    """Gemini анализирует свободный ответ и извлекает факты + даёт совет."""
    prompt = f"""Человек должен был лечь спать в {sleep_target}.
Утром он написал: "{answer}"

Твои задачи:
1. Определи лёг ли он вовремя (да/нет/частично).
2. Если не вовремя — извлеки причину из текста.
3. Дай один конкретный практический совет как устранить эту причину завтра.
4. В конце верни структурированную запись в формате JSON (только JSON, без обёртки):
{{
  "on_time": true/false/null,
  "actual_time": "HH:MM или null",
  "reason": "причина или null",
  "advice": "совет"
}}

Отвечай на русском. Без markdown. Сначала короткий человеческий ответ (2-3 предложения), потом JSON."""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    raw = response.choices[0].message.content.strip()

    # отделяем текст от JSON
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    human_text = raw[:json_match.start()].strip() if json_match else raw

    record = {}
    if json_match:
        try:
            record = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return human_text, record


# ─── СТАТИСТИКА ──────────────────────────────────────────────────────────────

async def build_stats(text):
    log = load_log()

    if not log:
        return "Пока нет данных для статистики. Нужно хотя бы несколько ночей."

    total = len(log)
    on_time = sum(1 for e in log if e.get("on_time") is True)
    not_on_time = sum(1 for e in log if e.get("on_time") is False)
    no_data = total - on_time - not_on_time

    reasons = [e["reason"] for e in log if e.get("reason")]

    log_text = "\n".join([
        f"{e['date']}: {'вовремя' if e.get('on_time') else 'не вовремя'}"
        f"{' — ' + e['reason'] if e.get('reason') else ''}"
        for e in log[-30:]
    ])

    prompt = f"""Проанализируй статистику сна человека.

Всего записей: {total}
Вовремя: {on_time}
Не вовремя: {not_on_time}
Нет данных: {no_data}
Причины нарушений: {', '.join(reasons) if reasons else 'нет данных'}

Последние 30 дней:
{log_text}

Запрос пользователя: {text}

Найди паттерны: в какие дни чаще не ложится вовремя, какие причины повторяются, есть ли прогресс. Без мотивационного тона. Без markdown. Конкретно."""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )

    summary = f"Всего ночей: {total} | Вовремя: {on_time} ({round(on_time/total*100)}%) | Не вовремя: {not_on_time}\n\n"
    summary += response.choices[0].message.content.strip()
    return summary


# ─── ОТПРАВКА ────────────────────────────────────────────────────────────────

async def send_text(bot_or_reply, text, is_reply=False):
    while len(text) > TELEGRAM_MAX_LEN:
        split_at = text.rfind("\n", 0, TELEGRAM_MAX_LEN)
        if split_at == -1:
            split_at = TELEGRAM_MAX_LEN
        chunk = text[:split_at].strip()
        if is_reply:
            await bot_or_reply(chunk)
        else:
            await bot_or_reply.send_message(chat_id=CHAT_ID, text=chunk)
        text = text[split_at:].strip()
    if text:
        if is_reply:
            await bot_or_reply(text)
        else:
            await bot_or_reply.send_message(chat_id=CHAT_ID, text=text)


# ─── ПЛАНОВЫЕ СООБЩЕНИЯ ──────────────────────────────────────────────────────

async def send_evening_reminder(bot: Bot):
    state = load_state()
    sleep_time = state["sleep_time"]

    msg = f"Сегодня цель — лечь спать в {sleep_time}. До отбоя ещё есть время подготовиться."

    await bot.send_message(chat_id=CHAT_ID, text=msg)

    state["daily"]["evening_sent"] = True
    save_state(state)
    print(f"Вечернее напоминание отправлено. Цель: {sleep_time}")


async def send_morning_check(bot: Bot):
    state = load_state()
    sleep_time = state["sleep_time"]

    msg = f"Доброе утро. Удалось лечь спать в {sleep_time}? Если нет — что помешало?"

    await bot.send_message(chat_id=CHAT_ID, text=msg)

    state["daily"]["morning_sent"] = True
    save_state(state)
    print("Утренний чек-ин отправлен.")


# ─── ПЛАНИРОВЩИК ─────────────────────────────────────────────────────────────

async def scheduler(bot: Bot):
    while True:
        try:
            current = now_dt()
            current_date = today_key()
            state = load_state()
            daily = state["daily"]

            # сбрасываем дневное состояние если новый день
            if daily.get("date") != current_date:
                state["daily"] = {
                    "date": current_date,
                    "evening_sent": False,
                    "morning_sent": False,
                    "morning_answered": False
                }
                save_state(state)
                daily = state["daily"]

            # вечернее напоминание в 21:00
            if not daily["evening_sent"]:
                due = TIMEZONE.localize(
                    datetime.combine(current.date(), EVENING_REMINDER_TIME)
                )
                if current >= due:
                    delay = minutes_late(due)
                    if delay <= MISSED_GRACE_MINUTES:
                        await send_evening_reminder(bot)
                    else:
                        state["daily"]["evening_sent"] = True
                        save_state(state)
                        print(f"Вечернее напоминание пропущено ({delay:.0f} мин)")

            # утренний чек-ин в 9:00
            if not daily["morning_sent"]:
                due = TIMEZONE.localize(
                    datetime.combine(current.date(), MORNING_CHECK_TIME)
                )
                if current >= due:
                    delay = minutes_late(due)
                    if delay <= MISSED_GRACE_MINUTES:
                        await send_morning_check(bot)
                    else:
                        state["daily"]["morning_sent"] = True
                        save_state(state)
                        print(f"Утренний чек-ин пропущен ({delay:.0f} мин)")

        except Exception as e:
            print(f"Ошибка планировщика: {e}")

        await asyncio.sleep(SCHEDULER_TICK_SECONDS)


# ─── ОБРАБОТЧИК СООБЩЕНИЙ ────────────────────────────────────────────────────

STATS_TRIGGERS = ["статистика", "паттерны", "прогресс", "покажи", "анализ", "итоги"]


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != CHAT_ID:
        return
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    reply = update.message.reply_text
    state = load_state()

    # ─── запрос статистики
    if any(t in text.lower() for t in STATS_TRIGGERS):
        await reply("Собираю статистику...")
        try:
            stats = await build_stats(text)
            await send_text(reply, stats, is_reply=True)
        except Exception as e:
            await reply(f"Ошибка: {e}")
        return

    # ─── изменение времени сна
    if wants_to_change_time(text):
        new_time = parse_sleep_time(text)
        if new_time:
            state["sleep_time"] = new_time
            save_state(state)
            await reply(f"Время сна обновлено: {new_time}. Завтрашнее напоминание будет на это время.")
        else:
            await reply("Не смог распознать время. Напиши, например: 'перенеси на 00:15' или 'лягу в 23:00'.")
        return

    # ─── ответ на утренний чек-ин
    daily = state["daily"]
    if daily.get("morning_sent") and not daily.get("morning_answered"):
        try:
            human_text, record = await analyze_morning_answer(text, state["sleep_time"])

            # сохраняем в лог
            log = load_log()
            log.append({
                "date": today_key(),
                "sleep_target": state["sleep_time"],
                "on_time": record.get("on_time"),
                "actual_time": record.get("actual_time"),
                "reason": record.get("reason"),
                "advice": record.get("advice"),
                "raw_answer": text
            })
            save_log(log)

            state["daily"]["morning_answered"] = True
            save_state(state)

            await send_text(reply, human_text, is_reply=True)

        except Exception as e:
            await reply(f"Ошибка при обработке ответа: {e}")
            print(f"Ошибка анализа утреннего ответа: {e}")
        return

    # ─── любое другое сообщение
    await reply(
        f"Текущее время сна: {state['sleep_time']}.\n\n"
        "Команды:\n"
        "— 'перенеси на ЧЧ:ММ' — изменить время сна\n"
        "— 'статистика' — показать прогресс и паттерны"
    )


# ─── СТАРТ ────────────────────────────────────────────────────────────────────

async def post_init(application: Application):
    asyncio.create_task(scheduler(application.bot))

    state = load_state()
    current = now_dt()

    print("Sleep бот запущен.")
    print(f"Часовой пояс: {TIMEZONE.zone}")
    print(f"Текущее время: {current.strftime('%A %d.%m.%Y %H:%M')}")
    print(f"Время сна: {state['sleep_time']}")
    print(f"Вечернее напоминание: отправлено={state['daily'].get('evening_sent', False)}")
    print(f"Утренний чек-ин: отправлен={state['daily'].get('morning_sent', False)}, отвечено={state['daily'].get('morning_answered', False)}")


def main():
    app = (
        Application
        .builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
