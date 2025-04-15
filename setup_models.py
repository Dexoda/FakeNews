#!/usr/bin/env python3
"""
Скрипт для установки и подготовки моделей машинного обучения,
необходимых для работы FakeNewsDetector.

Этот скрипт должен запускаться перед первым запуском системы.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("setup_models")

def setup_nltk():
    """Загружает необходимые данные для NLTK."""
    logger.info("Настройка NLTK...")

    try:
        import nltk

        # Загружаем необходимые ресурсы
        nltk_packages = [
            'punkt',         # Токенизатор
            'stopwords',     # Стоп-слова
        ]

        for package in nltk_packages:
            logger.info(f"Загрузка пакета NLTK: {package}")
            nltk.download(package, quiet=True)

        logger.info("Настройка NLTK завершена успешно")

    except Exception as e:
        logger.error(f"Ошибка при настройке NLTK: {e}")
        logger.warning("Продолжаем без NLTK")

def setup_spacy():
    """Загружает модели для spaCy."""
    logger.info("Настройка spaCy...")

    try:
        # Загружаем русскоязычную модель
        logger.info("Загрузка модели spaCy для русского языка")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "ru_core_news_md"])

        logger.info("Настройка spaCy завершена успешно")

    except Exception as e:
        logger.error(f"Ошибка при настройке spaCy: {e}")
        logger.warning("Продолжаем без spaCy")

def check_dostoevsky():
    """Проверяет наличие Dostoevsky, но не пытается скачать модель."""
    logger.info("Проверка наличия Dostoevsky...")

    try:
        # Только проверяем наличие и печатаем сообщение
        import dostoevsky
        logger.info(f"Dostoevsky установлен (версия {dostoevsky.__version__ if hasattr(dostoevsky, '__version__') else 'неизвестна'})")

        # НЕ пытаемся загружать модель
        logger.info("Модель Dostoevsky не будет загружаться автоматически.")
        logger.info("Система настроена для работы без анализа тональности.")

    except ImportError:
        logger.warning("Dostoevsky не установлен. Анализ тональности будет недоступен.")
    except Exception as e:
        logger.error(f"Ошибка при проверке Dostoevsky: {e}")
        logger.warning("Продолжаем без Dostoevsky")

def setup_fonts():
    """Устанавливает необходимые шрифты для визуализации."""
    logger.info("Настройка шрифтов...")

    try:
        # Проверяем наличие DejaVu Sans (используется в visualization/heatmap.py)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

        if not os.path.exists(font_path):
            logger.info("Устанавливаем пакет шрифтов DejaVu...")
            # Для Debian/Ubuntu
            try:
                subprocess.check_call(["apt-get", "update"])
                subprocess.check_call(["apt-get", "install", "-y", "fonts-dejavu"])
                logger.info("Шрифты DejaVu установлены успешно")
            except Exception:
                # Для других ОС - просто сообщаем о проблеме, не останавливаем скрипт
                logger.warning("Не удалось установить шрифты DejaVu. Возможно, потребуется ручная установка.")
        else:
            logger.info("Шрифты DejaVu уже установлены")

        logger.info("Настройка шрифтов завершена")

    except Exception as e:
        logger.error(f"Ошибка при настройке шрифтов: {e}")
        logger.warning("Продолжаем без настройки шрифтов")

def main():
    """Выполняет настройку всех необходимых моделей и ресурсов."""
    logger.info("Начало настройки моделей и ресурсов для FakeNewsDetector")

    try:
        # Настраиваем NLTK
        setup_nltk()

        # Настраиваем spaCy
        setup_spacy()

        # Проверяем Dostoevsky (без загрузки модели)
        check_dostoevsky()

        # Настраиваем шрифты
        setup_fonts()

        logger.info("Настройка моделей и ресурсов завершена успешно")

    except Exception as e:
        logger.error(f"Произошла ошибка при настройке: {e}")
        # Не завершаем скрипт с ошибкой, чтобы Docker build не прерывался
        logger.warning("Некоторые компоненты могут быть недоступны, но система продолжит работу")

if __name__ == "__main__":
    main()
