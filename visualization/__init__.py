"""
Пакет для визуализации результатов анализа.

Модули:
- charts.py: Генерация графиков и диаграмм
- heatmap.py: Создание тепловой карты текста
- reports.py: Генерация отчетов
"""

from .charts import create_chart
from .heatmap import create_text_heatmap
from .reports import generate_report

__all__ = ['create_chart', 'create_text_heatmap', 'generate_report']
