import logging
import os
import sys
import yaml
import aiohttp
import asyncio
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# ========== AI Factcheck Function ==========
async def get_ai_factcheck(news_text):
    prompt = f"""Проверь следующий текст на признаки фейк-ньюс.
Никогда не выдавай предположения или догадки как факты.
Если не можешь проверить информацию напрямую, напиши: "Не могу проверить."
Не интерпретируй и не перефразируй текст без запроса.
Любую непроверенную часть помечай: [Непроверено], [Домысел], [Спекуляция].
Если есть непроверенные части, пометь весь ответ.
Ответь одним из вариантов: "Фейк", "Реально", "Требует проверки".
Дай краткое пояснение.
Текст:
{news_text}
"""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    model = "qwen/qwen3-32b:free"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60
            ) as response:
                data = await response.json()
                if "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"]
                return "❌ Не удалось получить ответ от AI."
    except Exception as e:
        return f"❌ Ошибка при обращении к AI: {str(e)}"

# ========== Logging ==========
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ========== Config ==========
def load_config():
    try:
        with open("config.yml", "r") as config_file:
            config = yaml.safe_load(config_file)
            logger.info("Configuration loaded successfully")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

# ========== API Client ==========
class APIClient:
    def __init__(self, api_url):
        self.api_url = api_url
        self.session = None

    async def ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def analyze_news(self, text, url=None):
        await self.ensure_session()
        data = {"text": text}
        if url:
            data["url"] = url

        async with self.session.post(f"{self.api_url}/analyze", json=data) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logger.error(f"API error: {error_text}")
                raise Exception(f"API returned error: {response.status}")

# ========== Bot Command Handlers ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я FakeNewsDetector бот.\n\n"
        "Отправь мне текст новости или ссылку, и я проанализирую его на предмет достоверности.\n\n"
        "Используй /help для получения дополнительной информации."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔍 *FakeNewsDetector* — бот для анализа достоверности новостей\n\n"
        "*Как использовать:*\n"
        "1. Отправьте текст новости\n"
        "2. Или отправьте ссылку на новость\n"
        "3. Дождитесь завершения анализа\n\n"
        "*Доступные команды:*\n"
        "/start — Начать работу с ботом\n"
        "/help — Показать справку\n\n"
        "*О системе:*\n"
        "Бот использует многоуровневый лингвистический анализ и AI-фактчекер для оценки достоверности новостей.",
        parse_mode="Markdown"
    )

async def fetch_url_content(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    return f"Не удалось получить содержимое по URL (код ответа: {response.status})"

                content_type = response.headers.get('Content-Type', '')

                if 'text/html' in content_type:
                    html_content = await response.text()
                    text = re.sub(r'<[^>]+>', ' ', html_content)
                    text = re.sub(r'\s+', ' ', text).strip()
                    if len(text) > 4000:
                        text = text[:4000] + "... (текст обрезан)"
                    return text

                elif 'text/plain' in content_type:
                    return await response.text()

                else:
                    return f"Неподдерживаемый тип контента: {content_type}"

    except asyncio.TimeoutError:
        return "Время ожидания ответа от сервера истекло"
    except Exception as e:
        return f"Ошибка при получении содержимого URL: {str(e)}"

async def analyze_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message.text
    url = None
    extracted_text = message

    if message.startswith(("http://", "https://")):
        url = message
        progress_message = await update.message.reply_text("🔍 Получена ссылка. Извлекаю текст...")
        extracted_text = await fetch_url_content(url)
        await progress_message.edit_text("🔍 Текст извлечен. Начинаю анализ...")
    else:
        progress_message = await update.message.reply_text("🔍 Получен текст. Начинаю анализ...")

    try:
        await update.message.chat.send_action("typing")
        await progress_message.edit_text("⏳ Анализирую текст... (1/4)")

        api_client = context.bot_data["api_client"]
        analysis_result = await api_client.analyze_news(extracted_text, url)

        await progress_message.edit_text("⏳ Проверяю факты... (2/4)")
        await asyncio.sleep(1)
        await progress_message.edit_text("⏳ Формирую отчет... (3/4)")
        await asyncio.sleep(1)
        await progress_message.edit_text("⏳ Подготавливаю визуализацию... (4/4)")
        await asyncio.sleep(1)

        # Формируем основной результат анализа
        result_message = format_analysis_results(analysis_result)

        # Показываем вывод AI-фактчекера (если есть)
        ai_fact_check = analysis_result.get("ai_fact_check")
        if ai_fact_check:
            result_message += (
                "\n\n*AI-фактчекер (Qwen3-235B):*\n"
                f"{ai_fact_check}"
            )

        keyboard = [
            [
                InlineKeyboardButton("📊 Статистический анализ", callback_data="stats"),
                InlineKeyboardButton("🔤 Лингвистический анализ", callback_data="ling")
            ],
            [
                InlineKeyboardButton("💡 Семантический анализ", callback_data="sem"),
                InlineKeyboardButton("📋 Структурный анализ", callback_data="struct")
            ],
            [
                InlineKeyboardButton("⚠️ Подозрительные фрагменты", callback_data="suspicious"),
                InlineKeyboardButton("🤖 AI-Фактчек", callback_data="ai_factcheck")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Store analysis results for callback queries
        if "user_data" not in context.user_data:
            context.user_data["user_data"] = {}
        context.user_data["user_data"]["last_analysis"] = analysis_result
        context.user_data["user_data"]["original_text"] = extracted_text

        await progress_message.delete()
        await update.message.reply_text(result_message, reply_markup=reply_markup, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        await progress_message.edit_text(
            f"❌ Произошла ошибка при анализе: {str(e)}\n\n"
            "Пожалуйста, попробуйте позже или отправьте другой текст."
        )

# ========== Форматирование вывода ==========
def format_analysis_results(analysis_result):
    score = analysis_result.get("credibility_score", 0.5)

    if score >= 0.8:
        credibility = "✅ Высокая достоверность"
    elif score >= 0.6:
        credibility = "🟡 Средняя достоверность"
    elif score >= 0.4:
        credibility = "⚠️ Сомнительная достоверность"
    else:
        credibility = "❌ Низкая достоверность (вероятный фейк)"

    message = (
        f"*Результаты анализа:*\n\n"
        f"*Общая оценка достоверности:* {credibility} ({score:.0%})\n\n"
        f"*Краткое резюме:*\n"
        f"Текст был проанализирован по нескольким параметрам."
    )

    suspicious_count = len(analysis_result.get("suspicious_fragments", []))
    if suspicious_count > 0:
        message += f"\n\n⚠️ Обнаружено {suspicious_count} подозрительных фрагментов"

    message += "\n\nВыберите опцию ниже для просмотра подробных результатов анализа."

    return message

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if "user_data" not in context.user_data or "last_analysis" not in context.user_data["user_data"]:
        await query.edit_message_text("❌ Данные анализа не найдены. Пожалуйста, проведите новый анализ.")
        return

    analysis_result = context.user_data["user_data"]["last_analysis"]

    if query.data == "stats":
        stats_text = format_statistical_analysis(analysis_result)
        await query.edit_message_text(stats_text, reply_markup=query.message.reply_markup, parse_mode="Markdown")

    elif query.data == "ling":
        ling_text = format_linguistic_analysis(analysis_result)
        await query.edit_message_text(ling_text, reply_markup=query.message.reply_markup, parse_mode="Markdown")

    elif query.data == "sem":
        sem_text = format_semantic_analysis(analysis_result)
        await query.edit_message_text(sem_text, reply_markup=query.message.reply_markup, parse_mode="Markdown")

    elif query.data == "struct":
        struct_text = format_structural_analysis(analysis_result)
        await query.edit_message_text(struct_text, reply_markup=query.message.reply_markup, parse_mode="Markdown")

    elif query.data == "suspicious":
        suspicious_text = format_suspicious_fragments(analysis_result)
        await query.edit_message_text(suspicious_text, reply_markup=query.message.reply_markup, parse_mode="Markdown")

    elif query.data == "ai_factcheck":
        # Новый асинхронный AI-фактчек
        source_text = (
            context.user_data["user_data"].get("original_text")
            or analysis_result.get("original_text")
            or analysis_result.get("text")
        )
        if not source_text:
            await query.edit_message_text("❌ Нет исходного текста для AI-фактчека.")
            return
        await query.edit_message_text("🤖 Проверяю через AI...")
        factcheck = await get_ai_factcheck(source_text)
        await query.edit_message_text(f"🤖 *AI-фактчек:*\n\n{factcheck}", parse_mode="Markdown")

def format_statistical_analysis(analysis_result):
    stats = analysis_result.get("statistical", {})
    return (
        "*Статистический анализ:*\n\n"
        f"Соотношение фактов и мнений: {stats.get('fact_opinion_ratio', 'Н/Д')}\n"
        f"Читаемость текста: {stats.get('readability_score', 'Н/Д')}\n"
        f"Аномалии в частоте слов: {stats.get('word_frequency_anomalies', 'Не обнаружено')}\n\n"
        f"*Оценка достоверности по статистическим параметрам:* {stats.get('credibility_score', 0.5):.0%}"
    )

def format_linguistic_analysis(analysis_result):
    ling = analysis_result.get("linguistic", {})
    return (
        "*Лингвистический анализ:*\n\n"
        f"Тональность текста: {ling.get('sentiment', 'Н/Д')}\n"
        f"Эмоциональная окраска: {ling.get('emotional_tone', 'Н/Д')}\n"
        f"Манипулятивные конструкции: {ling.get('manipulative_constructs_count', 0)}\n\n"
        f"*Оценка достоверности по лингвистическим параметрам:* {ling.get('credibility_score', 0.5):.0%}"
    )

def format_semantic_analysis(analysis_result):
    sem = analysis_result.get("semantic", {})
    return (
        "*Семантический анализ:*\n\n"
        f"Смысловые противоречия: {sem.get('contradictions_count', 0)}\n"
        f"Связность текста: {sem.get('coherence_score', 'Н/Д')}\n"
        f"Ключевые тематические компоненты: {', '.join(sem.get('key_themes', ['Н/Д']))}\n\n"
        f"*Оценка достоверности по семантическим параметрам:* {sem.get('credibility_score', 0.5):.0%}"
    )

def format_structural_analysis(analysis_result):
    struct = analysis_result.get("structural", {})
    return (
        "*Структурный анализ:*\n\n"
        f"Соответствие журналистским стандартам: {struct.get('journalism_standards_score', 'Н/Д')}\n"
        f"Структура новости: {struct.get('structure_quality', 'Н/Д')}\n"
        f"Нарушения в построении материала: {struct.get('structure_violations', 'Не обнаружено')}\n\n"
        f"*Оценка достоверности по структурным параметрам:* {struct.get('credibility_score', 0.5):.0%}"
    )

def format_suspicious_fragments(analysis_result):
    fragments = analysis_result.get("suspicious_fragments", [])
    if not fragments:
        return "*Подозрительные фрагменты:*\n\nПодозрительных фрагментов не обнаружено."

    message = "*Подозрительные фрагменты:*\n\n"
    for i, fragment in enumerate(fragments[:5]):
        message += f"{i+1}. «{fragment['text']}»\n"
        message += f"   Причина: {fragment['reason']}\n"
        message += f"   Уверенность: {fragment['confidence']:.0%}\n\n"
    if len(fragments) > 5:
        message += f"И еще {len(fragments) - 5} фрагментов..."

    return message

# ========== Сервисные функции ==========
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")

async def shutdown(context: ContextTypes.DEFAULT_TYPE):
    logger.info("Shutting down bot...")
    if "api_client" in context.application.bot_data:
        await context.application.bot_data["api_client"].close()

def main():
    config = load_config()
    api_url = config.get("api", {}).get("url", "http://api-server:8000")
    api_client = APIClient(api_url)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set in environment")
        sys.exit(1)
    application = Application.builder().token(token).build()
    application.bot_data["api_client"] = api_client

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_text))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_error_handler(error_handler)
    try:
        if hasattr(application, 'post_shutdown') and application.post_shutdown is not None:
            application.post_shutdown.append(shutdown)
        else:
            logger.info("Shutdown callback registration not available, will rely on automatic cleanup")
    except Exception as e:
        logger.warning(f"Could not register shutdown handler: {e}")
        logger.info("Continuing without explicit shutdown handler")
    logger.info("Starting bot polling...")
    application.run_polling(stop_signals=None)

if __name__ == "__main__":
    logger.info("Starting Telegram bot")
    main()
