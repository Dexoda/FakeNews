"""
Пакет для многоуровневого анализа текста.

Модули:
- statistical.py: Статистический анализ текста
- linguistic.py: Лингвистический анализ текста
- semantic.py: Семантический анализ текста
- structural.py: Структурный анализ текста
- pipeline.py: Объединение всех уровней анализа
"""

from .pipeline import analyze_text

__all__ = ['analyze_text']
