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
DIALOG_FILE = DATA_DIR / "sleep_dialog.json"

SCHEDULER_TICK_SECONDS = 30
MISSED_GRACE_MINUTES = 20

MODEL_NAME = "google/gemini-2.5-flash"
TELEGRAM_MAX_LEN = 4000
DIALOG_CONTEXT_MESSAGES = 20

DEFAULT_SLEEP_TIME = "23:40"
EVENING_REMINDER_TIME = time(21, 0)
MORNING_CHECK_TIME = time(9, 0)

SYSTEM_PROMPT = """Ты — собеседник и аналитик по теме сна и режима дня.

Помогаешь человеку выработать стабильный режим сна. Отслеживаешь прогресс, замечаешь паттерны, даёшь конкретные практические советы.

Стиль: прямой, без мотивационного тона. Не говори "ты молодец", "всё получится", "не переживай". Общайся как думающий собеседник — анализируй, задавай уточняющие вопросы, предлагай конкретные решения.

Не используй markdown-разметку: никаких звёздочек, решёток, дефисов-маркеров. Пиши обычным текстом.

Не начинай ответ с "Понял", "Принято", "Записал". Переходи сразу к сути.

Объём ответа — по содержанию разговора.

Общайся на языке пользователя."""


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

def save_log(d):
    save_json(LOG_FILE, d)

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

def save_state(d):
    save_json(STATE_FILE, d)

def load_dialog():
    return load_json(DIALOG_FILE, [])

def save_dialog(d):
    save_json(DIALOG_FILE, d)


# ─── ВРЕМЯ ────────────────────────────────────────────────────────────────────

def now_dt():
    return datetime.now(TIMEZONE)

def today_key():
    return now_dt().strftime("%Y-%m-%d")

def minutes_late(due_time):
    current = now_dt()
    due = TIMEZONE.localize(datetime.combine(current.date(), due_time))
    return (current - due).total_seconds() / 60

def fmt_ts(dt):
    days = ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"]
    return f"[{days[dt.weekday()]}, {dt.strftime('%d.%m %H:%M')}]"


# ─── ПАРСИНГ ВРЕМЕНИ СНА ─────────────────────────────────────────────────────

def parse_sleep_time(text):
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
    triggers = ["перенеси", "измени время", "лягу в", "буду ложиться в",
                "поставь на", "сдвинь", "новое время", "время сна"]
    normalized = text.lower()
    return any(t in normalized for t in triggers)


# ─── АНАЛИЗ УТРЕННЕГО ОТВЕТА ─────────────────────────────────────────────────

async def analyze_morning_answer(answer, sleep_target):
    prompt = f"""Человек должен был лечь спать в {sleep_target}.
Утром он написал: "{answer}"

Задачи:
1. Определи лёг ли он вовремя (да/нет).
2. Если не вовремя — извлеки причину из текста.
3. Дай один конкретный практический совет как устранить эту причину.
4. В конце верни JSON (только JSON без обёртки):
{{"on_time": true/false/null, "actual_time": "HH:MM или null", "reason": "причина или null", "advice": "совет"}}

Отвечай на русском. Без markdown. Сначала короткий ответ (2-3 предложения), потом JSON."""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    raw = response.choices[0].message.content.strip()
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
    reasons = [e["reason"] for e in log if e.get("reason")]

    log_text = "\n".join([
        f"{e['date']}: {'вовремя' if e.get('on_time') else 'не вовремя'}"
        f"{' — ' + e['reason'] if e.get('reason') else ''}"
        for e in log[-30:]
    ])

    prompt = f"""Проанализируй статистику сна.

Всего записей: {total}
Вовремя: {on_time} ({round(on_time/total*100) if total else 0}%)
Не вовремя: {not_on_time}
Причины нарушений: {', '.join(reasons) if reasons else 'нет данных'}

Последние 30 дней:
{log_text}

Запрос пользователя: {text}

Найди паттерны: в какие дни чаще срывы, какие причины повторяются, есть ли прогресс. Без мотивационного тона. Без markdown. Конкретно."""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )

    header = f"Всего ночей: {total} | Вовремя: {on_time} ({round(on_time/total*100) if total else 0}%) | Не вовремя: {not_on_time}\n\n"
    return header + response.choices[0].message.content.strip()


# ─── AI ДИАЛОГ ───────────────────────────────────────────────────────────────

async def ask_ai(user_text, context_note=None):
    log = load_log()
    dialog = load_dialog()
    state = load_state()

    system_content = SYSTEM_PROMPT

    # контекст статистики
    if log:
        total = len(log)
        on_time = sum(1 for e in log if e.get("on_time") is True)
        recent_reasons = [e["reason"] for e in log[-10:] if e.get("reason")]
        system_content += (
            f"\n\nСТАТИСТИКА ПОЛЬЗОВАТЕЛЯ:\n"
            f"Целевое время сна: {state['sleep_time']}\n"
            f"Всего ночей в логе: {total}\n"
            f"Вовремя: {on_time} ({round(on_time/total*100)}%)\n"
            f"Недавние причины нарушений: {', '.join(recent_reasons) if recent_reasons else 'нет'}"
        )

    if context_note:
        system_content += f"\n\n{context_note}"

    now_str = now_dt().strftime("%A %d.%m.%Y %H:%M")
    system_content += f"\n\nТекущий момент: {now_str} ({TIMEZONE.zone})."

    messages = [{"role": "system", "content": system_content}]

    for msg in dialog[-DIALOG_CONTEXT_MESSAGES:]:
        messages.append({
            "role": msg["role"],
            "content": f"{msg.get('ts', '')} {msg['content']}".strip()
        })

    messages.append({
        "role": "user",
        "content": f"{fmt_ts(now_dt())} {user_text}"
    })

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )

    return response.choices[0].message.content.strip()


# ─── ОТПРАВКА ────────────────────────────────────────────────────────────────

async def send_text(reply_func, text):
    while len(text) > TELEGRAM_MAX_LEN:
        split_at = text.rfind("\n", 0, TELEGRAM_MAX_LEN)
        if split_at == -1:
            split_at = TELEGRAM_MAX_LEN
        await reply_func(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        await reply_func(text)

async def bot_send(bot: Bot, text):
    await send_text(lambda t: bot.send_message(chat_id=CHAT_ID, text=t), text)


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

            if daily.get("date") != current_date:
                state["daily"] = {
                    "date": current_date,
                    "evening_sent": False,
                    "morning_sent": False,
                    "morning_answered": False
                }
                save_state(state)
                daily = state["daily"]

            if not daily["evening_sent"]:
                delay = minutes_late(EVENING_REMINDER_TIME)
                if 0 <= delay <= MISSED_GRACE_MINUTES:
                    await send_evening_reminder(bot)
                elif delay > MISSED_GRACE_MINUTES:
                    state["daily"]["evening_sent"] = True
                    save_state(state)

            if not daily["morning_sent"]:
                delay = minutes_late(MORNING_CHECK_TIME)
                if 0 <= delay <= MISSED_GRACE_MINUTES:
                    await send_morning_check(bot)
                elif delay > MISSED_GRACE_MINUTES:
                    state["daily"]["morning_sent"] = True
                    save_state(state)

        except Exception as e:
            print(f"Ошибка планировщика: {e}")

        await asyncio.sleep(SCHEDULER_TICK_SECONDS)


# ─── ОБРАБОТЧИК СООБЩЕНИЙ ────────────────────────────────────────────────────

STATS_TRIGGERS = ["статистика", "паттерны", "прогресс", "покажи статистику",
                  "анализ сна", "итоги", "сколько раз"]


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != CHAT_ID:
        return
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    reply = update.message.reply_text
    state = load_state()

    # ─── статистика
    if any(t in text.lower() for t in STATS_TRIGGERS):
        await reply("Собираю статистику...")
        try:
            stats = await build_stats(text)
            await send_text(reply, stats)
        except Exception as e:
            await reply(f"Ошибка: {e}")
        return

    # ─── изменение времени сна
    if wants_to_change_time(text):
        new_time = parse_sleep_time(text)
        if new_time:
            state["sleep_time"] = new_time
            save_state(state)
            await reply(f"Время сна обновлено: {new_time}. Вечернее напоминание будет на это время.")
        else:
            await reply("Не смог распознать время. Напиши например: 'перенеси на 00:15'.")
        return

    # ─── утренний чек-ин — первый ответ идёт в лог
    daily = state["daily"]
    context_note = None

    if daily.get("morning_sent") and not daily.get("morning_answered"):
        try:
            human_text, record = await analyze_morning_answer(text, state["sleep_time"])

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

            # сохраняем в диалог и отвечаем
            dialog = load_dialog()
            dialog.append({"role": "user", "content": text,
                           "ts": fmt_ts(now_dt()), "time": now_dt().isoformat()})
            dialog.append({"role": "assistant", "content": human_text,
                           "ts": fmt_ts(now_dt()), "time": now_dt().isoformat()})
            if len(dialog) > 200:
                dialog = dialog[-200:]
            save_dialog(dialog)

            await send_text(reply, human_text)
            return

        except Exception as e:
            await reply(f"Ошибка при обработке ответа: {e}")
            print(f"Ошибка анализа: {e}")
            return

    # ─── свободный диалог
    try:
        response_text = await ask_ai(user_text=text, context_note=context_note)

        dialog = load_dialog()
        dialog.append({"role": "user", "content": text,
                       "ts": fmt_ts(now_dt()), "time": now_dt().isoformat()})
        dialog.append({"role": "assistant", "content": response_text,
                       "ts": fmt_ts(now_dt()), "time": now_dt().isoformat()})
        if len(dialog) > 200:
            dialog = dialog[-200:]
        save_dialog(dialog)

        await send_text(reply, response_text)

    except Exception as e:
        await reply(f"Ошибка: {e}")
        print(f"Ошибка диалога: {e}")


# ─── СТАРТ ────────────────────────────────────────────────────────────────────

async def post_init(application: Application):
    asyncio.create_task(scheduler(application.bot))
    state = load_state()
    current = now_dt()
    print("Sleep бот запущен.")
    print(f"Часовой пояс: {TIMEZONE.zone}")
    print(f"Текущее время: {current.strftime('%A %d.%m.%Y %H:%M')}")
    print(f"Время сна: {state['sleep_time']}")


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
