import logging
import os
import io
import base64
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import tempfile

# Настраиваем matplotlib для работы без GUI
matplotlib.use('Agg')

# Настраиваем логгер
logger = logging.getLogger(__name__)

class ChartGenerator:
    """
    Генератор графиков и диаграмм для визуализации результатов анализа.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализирует генератор графиков с настройками из конфигурации.

        Args:
            config: Настройки визуализации из конфигурационного файла
        """
        self.config = config or {}

        # Настройка стилей matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')

        # Настройка поддержки русского языка
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'

        # Создаем специальные цветовые карты
        self.credibility_cmap = LinearSegmentedColormap.from_list(
            'credibility',
            [(0, 'red'), (0.5, 'yellow'), (1, 'green')]
        )

    def generate_credibility_gauge(
        self,
        score: float,
        title: str = "Оценка достоверности"
    ) -> str:
        """
        Создает круговой индикатор для отображения оценки достоверности.

        Args:
            score: Оценка достоверности (от 0 до 1)
            title: Заголовок графика

        Returns:
            Изображение в формате base64
        """
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(7, 4), subplot_kw={'projection': 'polar'})

        # Нормализуем оценку
        score = max(0, min(1, score))

        # Настраиваем полярные координаты
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)

        # Создаем заливку для фона шкалы
        ax.fill_between(theta, 0, r, color='lightgray', alpha=0.3)

        # Создаем заливку для значения
        score_theta = np.linspace(0, np.pi * score, 100)
        score_r = np.ones_like(score_theta)

        # Определяем цвет на основе оценки
        if score >= 0.7:
            color = 'green'
        elif score >= 0.4:
            color = 'yellow'
        else:
            color = 'red'

        ax.fill_between(score_theta, 0, score_r, color=color, alpha=0.7)

        # Настраиваем внешний вид
        ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_rticks([])

        # Добавляем заголовок и значение
        ax.set_title(title, pad=15, fontsize=14)
        ax.text(np.pi/2, 0.5, f"{score:.0%}",
                ha='center', va='center', fontsize=18, fontweight='bold')

        # Настраиваем осевые линии
        ax.grid(True, alpha=0.3)
        ax.spines['polar'].set_visible(False)

        # Ограничиваем угол отображения
        ax.set_thetamin(0)
        ax.set_thetamax(180)

        # Преобразуем в изображение
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close(fig)

        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def generate_analysis_pie_charts(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Создает круговые диаграммы для разных аспектов анализа.

        Args:
            analysis_results: Результаты анализа текста

        Returns:
            Словарь с изображениями в формате base64
        """
        charts = {}

        # Диаграмма соотношения фактов и мнений
        if 'statistical' in analysis_results:
            fact_opinion_ratio = analysis_results['statistical'].get('fact_opinion_ratio', 0.5)
            charts['fact_opinion'] = self.generate_fact_opinion_pie(fact_opinion_ratio)

        # Диаграмма видов анализа и их оценок
        credibility_scores = {}

        if 'statistical' in analysis_results:
            credibility_scores['Статистический'] = analysis_results['statistical'].get('credibility_score', 0.5)

        if 'linguistic' in analysis_results:
            credibility_scores['Лингвистический'] = analysis_results['linguistic'].get('credibility_score', 0.5)

        if 'semantic' in analysis_results:
            credibility_scores['Семантический'] = analysis_results['semantic'].get('credibility_score', 0.5)

        if 'structural' in analysis_results:
            credibility_scores['Структурный'] = analysis_results['structural'].get('credibility_score', 0.5)

        if credibility_scores:
            charts['analysis_types'] = self.generate_analysis_types_bar(credibility_scores)

        # Диаграмма проверки фактов
        if 'factcheck_results' in analysis_results:
            charts['factcheck'] = self.generate_factcheck_bar(analysis_results['factcheck_results'])

        return charts

    def generate_fact_opinion_pie(self, fact_ratio: float) -> str:
        """
        Создает круговую диаграмму соотношения фактов и мнений.

        Args:
            fact_ratio: Доля фактов в тексте (от 0 до 1)

        Returns:
            Изображение в формате base64
        """
        # Нормализуем значение
        fact_ratio = max(0, min(1, fact_ratio))
        opinion_ratio = 1 - fact_ratio

        # Создаем диаграмму
        fig, ax = plt.subplots(figsize=(7, 5))

        labels = ['Факты', 'Мнения']
        sizes = [fact_ratio, opinion_ratio]
        colors = ['#4CAF50', '#FFC107']
        explode = (0.05, 0)

        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=False, startangle=90, wedgeprops={'edgecolor': 'w', 'linewidth': 1})

        # Круговая форма
        ax.axis('equal')

        # Добавляем заголовок
        ax.set_title('Соотношение фактов и мнений в тексте', pad=20, fontsize=14)

        # Преобразуем в изображение
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close(fig)

        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def generate_analysis_types_bar(self, credibility_scores: Dict[str, float]) -> str:
        """
        Создает столбчатую диаграмму оценок разных типов анализа.

        Args:
            credibility_scores: Словарь с оценками достоверности по разным типам анализа

        Returns:
            Изображение в формате base64
        """
        # Создаем диаграмму
        fig, ax = plt.subplots(figsize=(8, 5))

        types = list(credibility_scores.keys())
        scores = list(credibility_scores.values())

        # Создаем градиентные цвета в зависимости от оценки
        colors = [self.credibility_cmap(score) for score in scores]

        # Строим столбчатую диаграмму
        bars = ax.bar(types, scores, color=colors, width=0.6, edgecolor='white', linewidth=1)

        # Добавляем подписи значений
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.0%}', ha='center', va='bottom')

        # Настраиваем оси
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Оценка достоверности')
        ax.set_title('Результаты разных типов анализа', pad=20, fontsize=14)

        # Настраиваем сетку
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Преобразуем в изображение
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close(fig)

        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def generate_factcheck_bar(self, factcheck_results: List[Dict[str, Any]]) -> str:
        """
        Создает столбчатую диаграмму результатов проверки фактов.

        Args:
            factcheck_results: Результаты проверки фактов

        Returns:
            Изображение в формате base64
        """
        # Подсчитываем количество утверждений по категориям
        categories = {
            'подтверждено': 0,
            'вероятно правда': 0,
            'спорно': 0,
            'вероятно ложь': 0,
            'опровергнуто': 0,
            'не проверено': 0
        }

        for result in factcheck_results:
            status = result.get('status', 'не проверено')
            if status in categories:
                categories[status] += 1
            else:
                categories['не проверено'] += 1

        # Фильтруем пустые категории
        categories = {k: v for k, v in categories.items() if v > 0}

        # Создаем диаграмму
        fig, ax = plt.subplots(figsize=(8, 5))

        labels = list(categories.keys())
        values = list(categories.values())

        # Определяем цвета для категорий
        colors_map = {
            'подтверждено': 'green',
            'вероятно правда': 'lightgreen',
            'спорно': 'yellow',
            'вероятно ложь': 'orange',
            'опровергнуто': 'red',
            'не проверено': 'gray'
        }

        colors = [colors_map.get(label, 'gray') for label in labels]

        # Строим горизонтальную столбчатую диаграмму
        bars = ax.barh(labels, values, color=colors, edgecolor='white', linewidth=1)

        # Добавляем подписи значений
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center')

        # Настраиваем оси
        ax.set_xlabel('Количество утверждений')
        ax.set_title('Результаты проверки фактов', pad=20, fontsize=14)

        # Инвертируем оси, чтобы первая категория была сверху
        ax.invert_yaxis()

        # Настраиваем сетку
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Преобразуем в изображение
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close(fig)

        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def generate_emotion_bar(self, emotional_markers: Dict[str, List[str]]) -> str:
        """
        Создает столбчатую диаграмму эмоциональных маркеров в тексте.

        Args:
            emotional_markers: Словарь с категориями эмоций и найденными словами

        Returns:
            Изображение в формате base64
        """
        # Подсчитываем количество слов в каждой категории
        emotions = {}
        for category, words in emotional_markers.items():
            if words:  # Пропускаем пустые категории
                emotions[category] = len(words)

        if not emotions:
            # Если нет эмоциональных маркеров, создаем заглушку
            emotions = {'нейтральный текст': 1}

        # Создаем диаграмму
        fig, ax = plt.subplots(figsize=(8, 5))

        labels = list(emotions.keys())
        values = list(emotions.values())

        # Определяем цвета для категорий
        color_map = {
            'страх': 'purple',
            'гнев': 'red',
            'радость': 'green',
            'печаль': 'blue',
            'удивление': 'orange',
            'преувеличение': 'brown',
            'нейтральный текст': 'gray'
        }

        colors = [color_map.get(label, 'gray') for label in labels]

        # Строим столбчатую диаграмму
        bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor='white', linewidth=1)

        # Добавляем подписи значений
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')

        # Настраиваем оси
        ax.set_ylabel('Количество маркеров')
        ax.set_title('Эмоциональные маркеры в тексте', pad=20, fontsize=14)

        # Настраиваем сетку
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Настраиваем подписи на оси x
        plt.xticks(rotation=30, ha='right')

        # Преобразуем в изображение
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close(fig)

        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def generate_manipulative_constructs_chart(self, constructs: List[Dict[str, Any]]) -> str:
        """
        Создает диаграмму для отображения манипулятивных конструкций.

        Args:
            constructs: Список найденных манипулятивных конструкций

        Returns:
            Изображение в формате base64
        """
        if not constructs:
            # Если нет манипулятивных конструкций, создаем заглушку
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, 'Манипулятивные конструкции не обнаружены',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')

            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=100)
            plt.close(fig)

            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')

        # Подсчитываем количество конструкций каждого типа
        construct_types = {}
        for construct in constructs:
            construct_type = construct.get('type', 'неизвестный')
            construct_types[construct_type] = construct_types.get(construct_type, 0) + 1

        # Создаем диаграмму
        fig, ax = plt.subplots(figsize=(8, 5))

        labels = list(construct_types.keys())
        values = list(construct_types.values())

        # Строим горизонтальную столбчатую диаграмму
        bars = ax.barh(labels, values, color='salmon', edgecolor='white', linewidth=1)

        # Добавляем подписи значений
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center')

        # Настраиваем оси
        ax.set_xlabel('Количество')
        ax.set_title('Манипулятивные речевые конструкции', pad=20, fontsize=14)

        # Инвертируем оси, чтобы первая категория была сверху
        ax.invert_yaxis()

        # Настраиваем сетку
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Преобразуем в изображение
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close(fig)

        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

def create_chart(chart_type: str, data: Dict[str, Any], config: Dict[str, Any] = None) -> str:
    """
    Создает график указанного типа.

    Args:
        chart_type: Тип графика
        data: Данные для построения графика
        config: Дополнительные настройки

    Returns:
        Изображение в формате base64
    """
    generator = ChartGenerator(config)

    if chart_type == 'credibility_gauge':
        return generator.generate_credibility_gauge(
            score=data.get('score', 0.5),
            title=data.get('title', 'Оценка достоверности')
        )

    elif chart_type == 'fact_opinion_pie':
        return generator.generate_fact_opinion_pie(
            fact_ratio=data.get('fact_ratio', 0.5)
        )

    elif chart_type == 'analysis_types_bar':
        return generator.generate_analysis_types_bar(
            credibility_scores=data.get('credibility_scores', {})
        )

    elif chart_type == 'factcheck_bar':
        return generator.generate_factcheck_bar(
            factcheck_results=data.get('factcheck_results', [])
        )

    elif chart_type == 'emotion_bar':
        return generator.generate_emotion_bar(
            emotional_markers=data.get('emotional_markers', {})
        )

    elif chart_type == 'manipulative_constructs_chart':
        return generator.generate_manipulative_constructs_chart(
            constructs=data.get('constructs', [])
        )

    else:
        logger.warning(f"Неизвестный тип графика: {chart_type}")
        return ""
