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
            'wordnet',       # Тезаурус
            'averaged_perceptron_tagger'  # POS-теггер
        ]
        
        for package in nltk_packages:
            logger.info(f"Загрузка пакета NLTK: {package}")
            nltk.download(package, quiet=True)
        
        logger.info("Настройка NLTK завершена успешно")
        
    except Exception as e:
        logger.error(f"Ошибка при настройке NLTK: {e}")
        raise

def setup_spacy():
    """Загружает модели для spaCy."""
    logger.info("Настройка spaCy...")
    
    try:
        # Проверяем наличие spaCy
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
        
        # Загружаем русскоязычную модель
        logger.info("Загрузка модели spaCy для русского языка")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "ru_core_news_md"])
        
        logger.info("Настройка spaCy завершена успешно")
        
    except Exception as e:
        logger.error(f"Ошибка при настройке spaCy: {e}")
        raise

def setup_dostoevsky():
    """Загружает модели для Dostoevsky (анализ тональности)."""
    logger.info("Настройка Dostoevsky...")
    
    try:
        # Проверяем наличие Dostoevsky
        subprocess.check_call([sys.executable, "-m", "pip", "install", "dostoevsky"])
        
        # Загружаем модель
        from dostoevsky.data import DATA_BASE_PATH
        from dostoevsky.models import FastTextSocialNetworkModel
        
        model_path = Path(DATA_BASE_PATH) / "models" / "fasttext-social-network-model.bin"
        
        if not model_path.exists():
            logger.info("Загрузка модели Dostoevsky...")
            FastTextSocialNetworkModel.MODEL_PATH = model_path
            model = FastTextSocialNetworkModel(tokenizer=None)
            # Первый вызов загрузит модель
            _ = model.get_model()
            logger.info("Модель Dostoevsky загружена успешно")
        else:
            logger.info("Модель Dostoevsky уже существует")
        
        logger.info("Настройка Dostoevsky завершена успешно")
        
    except Exception as e:
        logger.error(f"Ошибка при настройке Dostoevsky: {e}")
        raise

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
        # Не останавливаем скрипт, так как это не критическая ошибка

def main():
    """Выполняет настройку всех необходимых моделей и ресурсов."""
    logger.info("Начало настройки моделей и ресурсов для FakeNewsDetector")
    
    try:
        # Настраиваем NLTK
        setup_nltk()
        
        # Настраиваем spaCy
        setup_spacy()
        
        # Настраиваем Dostoevsky
        setup_dostoevsky()
        
        # Настраиваем шрифты
        setup_fonts()
        
        logger.info("Настройка моделей и ресурсов завершена успешно")
        
    except Exception as e:
        logger.error(f"Произошла ошибка при настройке: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
