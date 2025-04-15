import logging
import os
import sys
import yaml
import aiohttp
import io
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    try:
        with open("config.yml", "r") as config_file:
            config = yaml.safe_load(config_file)
            logger.info("Configuration loaded successfully")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

# API client
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

# Bot command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "👋 Привет! Я FakeNewsDetector бот.\n\n"
        "Отправь мне текст новости или ссылку, и я проанализирую его на предмет достоверности.\n\n"
        "Используй /help для получения дополнительной информации."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "🔍 *FakeNewsDetector* - бот для анализа достоверности новостей\n\n"
        "*Как использовать:*\n"
        "1. Отправьте текст новости\n"
        "2. Или отправьте ссылку на новость\n"
        "3. Дождитесь завершения анализа\n\n"
        "*Доступные команды:*\n"
        "/start - Начать работу с ботом\n"
        "/help - Показать справку\n\n"
        "*О системе:*\n"
        "Бот использует многоуровневый лингвистический анализ и проверку фактов для определения достоверности новостей.",
        parse_mode="Markdown"
    )

async def analyze_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analyze the news text sent by the user."""
    message = update.message.text

    # Check if it's a URL
    url = None
    if message.startswith(("http://", "https://")):
        url = message
        progress_message = await update.message.reply_text(
            "🔍 Получена ссылка. Начинаю анализ..."
        )
    else:
        progress_message = await update.message.reply_text(
            "🔍 Получен текст. Начинаю анализ..."
        )

    try:
        # Show typing indicator
        await update.message.chat.send_action("typing")

        # Update progress
        await progress_message.edit_text("⏳ Анализирую текст... (1/4)")

        # Send request to API
        api_client = context.bot_data["api_client"]
        if url:
            # For URLs, we'd extract text and then analyze
            # This is simplified here
            analysis_result = await api_client.analyze_news("", url)
        else:
            analysis_result = await api_client.analyze_news(message)

        # Update progress
        await progress_message.edit_text("⏳ Проверяю факты... (2/4)")
        await asyncio.sleep(1)  # Simulate processing time

        await progress_message.edit_text("⏳ Формирую отчет... (3/4)")
        await asyncio.sleep(1)  # Simulate processing time

        await progress_message.edit_text("⏳ Подготавливаю визуализацию... (4/4)")
        await asyncio.sleep(1)  # Simulate processing time

        # Generate visualization from the results
        # This would be implemented in the real system
        score = analysis_result.get("credibility_score", 0.5)

        # Format the results
        result_message = format_analysis_results(analysis_result)

        # Create keyboard for detailed results
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
                InlineKeyboardButton("⚠️ Подозрительные фрагменты", callback_data="suspicious")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Store analysis results in context for callback queries
        if not "user_data" in context.user_data:
            context.user_data["user_data"] = {}
        context.user_data["user_data"]["last_analysis"] = analysis_result

        # Delete progress message and send results
        await progress_message.delete()
        await update.message.reply_text(result_message, reply_markup=reply_markup)

        # Send visualization if available
        # In a real system, this would be a chart generated from analysis_result
        # await update.message.reply_photo(open("temp_visualization.png", "rb"))

    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        await progress_message.edit_text(
            f"❌ Произошла ошибка при анализе: {str(e)}\n\n"
            "Пожалуйста, попробуйте позже или отправьте другой текст."
        )

def format_analysis_results(analysis_result):
    """Format the analysis results for Telegram message"""
    score = analysis_result.get("credibility_score", 0.5)

    # Determine credibility level based on score
    if score >= 0.8:
        credibility = "✅ Высокая достоверность"
    elif score >= 0.6:
        credibility = "🟡 Средняя достоверность"
    elif score >= 0.4:
        credibility = "⚠️ Сомнительная достоверность"
    else:
        credibility = "❌ Низкая достоверность (вероятный фейк)"

    # Format message
    message = (
        f"*Результаты анализа:*\n\n"
        f"*Общая оценка достоверности:* {credibility} ({score:.0%})\n\n"
        f"*Краткое резюме:*\n"
        f"Текст был проанализирован по нескольким параметрам. "
    )

    # Add specific findings if available
    suspicious_count = len(analysis_result.get("suspicious_fragments", []))
    if suspicious_count > 0:
        message += f"\n\n⚠️ Обнаружено {suspicious_count} подозрительных фрагментов"

    message += "\n\nВыберите опцию ниже для просмотра подробных результатов анализа."

    return message

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button presses from inline keyboards"""
    query = update.callback_query
    await query.answer()

    # Get stored analysis data
    if not "user_data" in context.user_data or "last_analysis" not in context.user_data["user_data"]:
        await query.edit_message_text("❌ Данные анализа не найдены. Пожалуйста, проведите новый анализ.")
        return

    analysis_result = context.user_data["user_data"]["last_analysis"]

    # Handle different button callbacks
    if query.data == "stats":
        stats_text = format_statistical_analysis(analysis_result)
        await query.edit_message_text(stats_text, reply_markup=query.message.reply_markup)
    elif query.data == "ling":
        ling_text = format_linguistic_analysis(analysis_result)
        await query.edit_message_text(ling_text, reply_markup=query.message.reply_markup)
    elif query.data == "sem":
        sem_text = format_semantic_analysis(analysis_result)
        await query.edit_message_text(sem_text, reply_markup=query.message.reply_markup)
    elif query.data == "struct":
        struct_text = format_structural_analysis(analysis_result)
        await query.edit_message_text(struct_text, reply_markup=query.message.reply_markup)
    elif query.data == "suspicious":
        suspicious_text = format_suspicious_fragments(analysis_result)
        await query.edit_message_text(suspicious_text, reply_markup=query.message.reply_markup)

def format_statistical_analysis(analysis_result):
    """Format statistical analysis results"""
    stats = analysis_result.get("statistical", {})
    return (
        "*Статистический анализ:*\n\n"
        f"Соотношение фактов и мнений: {stats.get('fact_opinion_ratio', 'Н/Д')}\n"
        f"Читаемость текста: {stats.get('readability_score', 'Н/Д')}\n"
        f"Аномалии в частоте слов: {stats.get('word_frequency_anomalies', 'Не обнаружено')}\n\n"
        f"*Оценка достоверности по статистическим параметрам:* {stats.get('credibility_score', 0.5):.0%}"
    )

def format_linguistic_analysis(analysis_result):
    """Format linguistic analysis results"""
    ling = analysis_result.get("linguistic", {})
    return (
        "*Лингвистический анализ:*\n\n"
        f"Тональность текста: {ling.get('sentiment', 'Н/Д')}\n"
        f"Эмоциональная окраска: {ling.get('emotional_tone', 'Н/Д')}\n"
        f"Манипулятивные конструкции: {ling.get('manipulative_constructs_count', 0)}\n\n"
        f"*Оценка достоверности по лингвистическим параметрам:* {ling.get('credibility_score', 0.5):.0%}"
    )

def format_semantic_analysis(analysis_result):
    """Format semantic analysis results"""
    sem = analysis_result.get("semantic", {})
    return (
        "*Семантический анализ:*\n\n"
        f"Смысловые противоречия: {sem.get('contradictions_count', 0)}\n"
        f"Связность текста: {sem.get('coherence_score', 'Н/Д')}\n"
        f"Ключевые тематические компоненты: {', '.join(sem.get('key_themes', ['Н/Д']))}\n\n"
        f"*Оценка достоверности по семантическим параметрам:* {sem.get('credibility_score', 0.5):.0%}"
    )

def format_structural_analysis(analysis_result):
    """Format structural analysis results"""
    struct = analysis_result.get("structural", {})
    return (
        "*Структурный анализ:*\n\n"
        f"Соответствие журналистским стандартам: {struct.get('journalism_standards_score', 'Н/Д')}\n"
        f"Структура новости: {struct.get('structure_quality', 'Н/Д')}\n"
        f"Нарушения в построении материала: {struct.get('structure_violations', 'Не обнаружено')}\n\n"
        f"*Оценка достоверности по структурным параметрам:* {struct.get('credibility_score', 0.5):.0%}"
    )

def format_suspicious_fragments(analysis_result):
    """Format suspicious fragments"""
    fragments = analysis_result.get("suspicious_fragments", [])
    if not fragments:
        return "*Подозрительные фрагменты:*\n\nПодозрительных фрагментов не обнаружено."

    message = "*Подозрительные фрагменты:*\n\n"
    for i, fragment in enumerate(fragments[:5]):  # Limit to 5 fragments
        message += f"{i+1}. «{fragment['text']}»\n"
        message += f"   Причина: {fragment['reason']}\n"
        message += f"   Уверенность: {fragment['confidence']:.0%}\n\n"

    if len(fragments) > 5:
        message += f"И еще {len(fragments) - 5} фрагментов..."

    return message

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors caused by updates."""
    logger.error(f"Update {update} caused error {context.error}")

async def shutdown(application):
    """Shut down the bot gracefully"""
    logger.info("Shutting down bot...")
    if "api_client" in application.bot_data:
        await application.bot_data["api_client"].close()

def main():
    """Start the bot."""
    # Load configuration
    config = load_config()

    # Create API client
    api_url = config.get("api", {}).get("url", "http://api-server:8000")
    api_client = APIClient(api_url)

    # Get bot token from environment
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set in environment")
        sys.exit(1)

    # Create application and add handlers
    application = Application.builder().token(token).build()
    application.bot_data["api_client"] = api_client

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Add message handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_text))

    # Add callback query handler
    application.add_handler(CallbackQueryHandler(button_handler))

    # Add error handler
    application.add_error_handler(error_handler)

    # Закомментируем эту строку, которая вызывает ошибку:
    # application.add_shutdown_callback(shutdown)

    # Start the Bot
    logger.info("Starting bot polling...")
    application.run_polling()

if __name__ == "__main__":
    logger.info("Starting Telegram bot")
    main()
