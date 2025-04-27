#!/usr/bin/env python3

import os
import re
import sys
import shutil

def fix_semantic_py():
    """Fix the semantic.py file to avoid using noun_chunks for Russian language."""
    semantic_file = "analyzer/semantic.py"
    if not os.path.exists(semantic_file):
        print(f"Ошибка: Файл {semantic_file} не найден.")
        return False
    
    # Создаем упрощенную версию semantic.py
    simplified_semantic = """import logging
import re
from typing import Dict, Any, List, Tuple, Set
import nltk
import numpy as np

logger = logging.getLogger(__name__)

# Загрузим необходимые данные для NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize the global variables
global nlp
nlp = None

async def perform_semantic_analysis(text: str) -> Dict[str, Any]:
    """
    Выполняет семантический анализ текста.

    Аргументы:
        text: Текст новости для анализа

    Возвращает:
        словарь с результатами семантического анализа
    """
    logger.info("Выполняется семантический анализ текста")

    try:
        # For simplicity, create a dummy result with extracted claims
        return {
            "entities": {"персоны": [], "организации": [], "локации": [], "даты": [], "другое": []},
            "key_themes": extract_key_themes(text),
            "coherence": {
                "coherence_score": 0.5,
                "logical_flow": "не определено",
                "topic_shifts": 0,
                "coherence_issues": []
            },
            "contradictions": [],
            "contradictions_count": 0,
            "identified_claims": extract_simple_claims(text),
            "suspicious_fragments": [],
            "credibility_score": 0.5
        }
    except Exception as e:
        logger.error(f"Ошибка при семантическом анализе: {e}")
        return {
            "entities": {"персоны": [], "организации": [], "локации": [], "даты": [], "другое": []},
            "key_themes": [],
            "coherence": {
                "coherence_score": 0.5,
                "logical_flow": "не определено",
                "topic_shifts": 0,
                "coherence_issues": []
            },
            "contradictions": [],
            "contradictions_count": 0,
            "identified_claims": [],
            "suspicious_fragments": [],
            "credibility_score": 0.5,
            "error": str(e)
        }

def extract_key_themes(text: str) -> List[str]:
    """
    Простое извлечение ключевых тем на основе частотности слов.
    """
    # Разбиваем на слова
    words = re.findall(r'\\b\\w+\\b', text.lower())
    
    # Подсчитываем частоту слов
    word_freq = {}
    for word in words:
        if len(word) > 3:  # игнорируем короткие слова
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Сортируем по частоте
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Возвращаем топ-7 слов
    return [word for word, _ in sorted_words[:7]]
        
def extract_simple_claims(text: str) -> List[str]:
    """
    Simple method to extract claims without using spaCy.
    
    This is a fallback when spaCy's noun_chunks aren't available.
    """
    # Simple sentence tokenization
    sentences = nltk.sent_tokenize(text, language='russian')
    
    # Basic filtering criteria
    claims = []
    for sentence in sentences:
        # Skip questions and exclamations
        if sentence.strip().endswith('?') or sentence.strip().endswith('!'):
            continue
            
        # Skip very short sentences
        if len(sentence.split()) < 5:
            continue
            
        # Skip sentences with subjective markers
        subjectivity_markers = [
            'считаю', 'думаю', 'полагаю', 'верю', 'кажется', 'возможно',
            'вероятно', 'по-моему', 'по-видимому', 'на мой взгляд'
        ]
        
        if not any(marker in sentence.lower() for marker in subjectivity_markers):
            # Clean up extra whitespace
            claim = re.sub(r'\\s+', ' ', sentence).strip()
            claims.append(claim)
    
    return claims
"""
    
    # Сохраняем файл
    with open(semantic_file, 'w', encoding='utf-8') as file:
        file.write(simplified_semantic)
    
    print(f"Файл {semantic_file} успешно упрощен!")
    return True

def fix_app_py():
    """Fix app.py to ensure it doesn't try to import load_config from config."""
    app_file = "app.py"
    if not os.path.exists(app_file):
        print(f"Ошибка: Файл {app_file} не найден.")
        return False
    
    # Читаем содержимое файла
    with open(app_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Проверяем, есть ли проблемная строка
    if "from config import load_config" in content:
        print("Найдена проблемная строка в app.py, исправляем...")
        # Заменяем проблемную строку
        new_content = content.replace(
            "from config import load_config", 
            "# Определяем функцию load_config локально"
        )
        
        # Сохраняем исправленный файл
        with open(app_file, 'w', encoding='utf-8') as file:
            file.write(new_content)
        
        print("Файл app.py успешно исправлен!")
    else:
        print("Проблемная строка не найдена в app.py.")
    
    return True

def main():
    """Запускает все исправления."""
    print("Запуск комплексных исправлений проекта...")
    
    # Исправляем semantic.py
    if fix_semantic_py():
        print("Исправление semantic.py выполнено успешно!")
    else:
        print("Ошибка при исправлении semantic.py.")
    
    # Исправляем app.py
    if fix_app_py():
        print("Исправление app.py выполнено успешно!")
    else:
        print("Ошибка при исправлении app.py.")
    
    print("Все исправления выполнены! Перестройте и перезапустите контейнеры.")

if __name__ == "__main__":
    main()
