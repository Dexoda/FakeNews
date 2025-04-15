import logging
import re
import io
import base64
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import html

logger = logging.getLogger(__name__)

class TextHeatmap:
    """
    Создает тепловую карту текста с выделением подозрительных фрагментов.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализирует генератор тепловой карты с настройками из конфигурации.

        Args:
            config: Настройки визуализации из конфигурационного файла
        """
        self.config = config or {}

        # Настройки цветов для тепловой карты
        self.colors = {
            'high_risk': self.config.get('heatmap', {}).get('colors', {}).get('high_risk', '#FF0000'),  # Красный
            'medium_risk': self.config.get('heatmap', {}).get('colors', {}).get('medium_risk', '#FFA500'),  # Оранжевый
            'low_risk': self.config.get('heatmap', {}).get('colors', {}).get('low_risk', '#FFFF00'),  # Желтый
            'safe': self.config.get('heatmap', {}).get('colors', {}).get('safe', '#00FF00'),  # Зеленый
            'background': '#FFFFFF',  # Белый
            'text': '#000000'  # Черный
        }

        # Загружаем шрифт
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 14)
            self.small_font = ImageFont.truetype("DejaVuSans.ttf", 12)
        except IOError:
            logger.warning("Не удалось загрузить шрифт DejaVuSans.ttf, используется шрифт по умолчанию")
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

    def generate_html_heatmap(self, text: str, suspicious_fragments: List[Dict[str, Any]]) -> str:
        """
        Создает HTML-код тепловой карты текста.

        Args:
            text: Исходный текст
            suspicious_fragments: Список подозрительных фрагментов

        Returns:
            HTML-код для отображения тепловой карты
        """
        # Подготавливаем текст
        escaped_text = html.escape(text)

        # Сортируем фрагменты по позиции в обратном порядке
        # чтобы не сбивать позиции при вставке HTML-тегов
        sorted_fragments = sorted(suspicious_fragments, key=lambda x: x.get('start', 0), reverse=True)

        # Вставляем HTML-теги для подсветки фрагментов
        for fragment in sorted_fragments:
            start = fragment.get('start', 0)
            end = fragment.get('end', 0)
            confidence = fragment.get('confidence', 0.5)
            reason = fragment.get('reason', 'Подозрительный фрагмент')

            # Определяем цвет в зависимости от уровня подозрительности
            if confidence >= 0.8:
                color = self.colors['high_risk']
            elif confidence >= 0.6:
                color = self.colors['medium_risk']
            elif confidence >= 0.4:
                color = self.colors['low_risk']
            else:
                color = self.colors['safe']

            # Создаем HTML-тег для подсветки
            highlight_tag = f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;" title="{html.escape(reason)}">'

            # Вставляем открывающий тег
            escaped_text = escaped_text[:start] + highlight_tag + escaped_text[start:]

            # Вставляем закрывающий тег
            escaped_text = escaped_text[:end + len(highlight_tag)] + '</span>' + escaped_text[end + len(highlight_tag):]

        # Заменяем переносы строк на HTML-теги
        escaped_text = escaped_text.replace('\n', '<br>')

        # Формируем HTML-код
        html_content = f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; padding: 10px; background-color: #f9f9f9; border-radius: 5px; max-width: 800px;">
            <h3 style="margin-top: 0; color: #333;">Тепловая карта текста</h3>
            <p style="margin-bottom: 5px; color: #666;">Наведите курсор на выделенные фрагменты для просмотра причины подозрительности</p>
            <div style="margin-top: 10px;">
                {escaped_text}
            </div>
            <div style="margin-top: 15px; font-size: 0.8em; color: #666;">
                Цветовая легенда:
                <span style="background-color: {self.colors['high_risk']}; padding: 2px 5px; border-radius: 3px; margin: 0 5px;">Высокий риск</span>
                <span style="background-color: {self.colors['medium_risk']}; padding: 2px 5px; border-radius: 3px; margin: 0 5px;">Средний риск</span>
                <span style="background-color: {self.colors['low_risk']}; padding: 2px 5px; border-radius: 3px; margin: 0 5px;">Низкий риск</span>
            </div>
        </div>
        """

        return html_content

    def generate_image_heatmap(self, text: str, suspicious_fragments: List[Dict[str, Any]]) -> str:
        """
        Создает изображение тепловой карты текста.

        Args:
            text: Исходный текст
            suspicious_fragments: Список подозрительных фрагментов

        Returns:
            Изображение в формате base64
        """
        # Параметры изображения
        padding = 20
        line_height = 20
        char_width = 10  # Примерная средняя ширина символа
        max_line_length = 80  # Максимальное количество символов в строке

        # Разбиваем текст на строки с учетом максимальной длины
        lines = []
        current_line = ""

        for char in text:
            if char == '\n' or len(current_line) >= max_line_length:
                lines.append(current_line)
                current_line = ""
                if char == '\n':
                    continue
            current_line += char

        if current_line:
            lines.append(current_line)

        # Определяем размеры изображения
        width = max(len(line) for line in lines) * char_width + 2 * padding
        height = len(lines) * line_height + 2 * padding

        # Создаем изображение с белым фоном
        image = Image.new('RGB', (width, height), color=self.colors['background'])
        draw = ImageDraw.Draw(image)

        # Отображаем текст
        for i, line in enumerate(lines):
            y = padding + i * line_height
            draw.text((padding, y), line, font=self.font, fill=self.colors['text'])

        # Подсвечиваем подозрительные фрагменты
        for fragment in suspicious_fragments:
            start = fragment.get('start', 0)
            end = fragment.get('end', 0)
            confidence = fragment.get('confidence', 0.5)

            # Определяем цвет в зависимости от уровня подозрительности
            if confidence >= 0.8:
                color = self.colors['high_risk']
            elif confidence >= 0.6:
                color = self.colors['medium_risk']
            elif confidence >= 0.4:
                color = self.colors['low_risk']
            else:
                color = self.colors['safe']

            # Находим позиции фрагмента в координатах изображения
            # (Это упрощенный подход, в реальной системе нужен более точный расчет)

            # Подсчитываем количество символов до начала фрагмента
            chars_before_start = 0
            for i, char in enumerate(text):
                if i >= start:
                    break
                chars_before_start += 1

            # Подсчитываем количество символов до конца фрагмента
            chars_before_end = 0
            for i, char in enumerate(text):
                if i >= end:
                    break
                chars_before_end += 1

            # Определяем строки и позиции
            start_line = chars_before_start // max_line_length
            start_pos = chars_before_start % max_line_length

            end_line = chars_before_end // max_line_length
            end_pos = chars_before_end % max_line_length

            # Рисуем выделение для каждой строки фрагмента
            for line in range(start_line, end_line + 1):
                if line < len(lines):
                    start_x = padding
                    if line == start_line:
                        start_x += start_pos * char_width

                    end_x = padding + len(lines[line]) * char_width
                    if line == end_line:
                        end_x = padding + end_pos * char_width

                    draw.rectangle([start_x, padding + line * line_height,
                                    end_x, padding + (line + 1) * line_height - 2],
                                    fill=color, outline=None)

                    # Перерисовываем текст поверх выделения
                    draw.text((padding, padding + line * line_height),
                            lines[line], font=self.font, fill=self.colors['text'])

        # Добавляем легенду
        legend_y = height - padding - 15
        draw.text((padding, legend_y), "Легенда:", font=self.small_font, fill=self.colors['text'])

        # Высокий риск
        draw.rectangle([padding + 80, legend_y, padding + 100, legend_y + 10],
                        fill=self.colors['high_risk'], outline=None)
        draw.text((padding + 110, legend_y), "Высокий риск", font=self.small_font, fill=self.colors['text'])

        # Средний риск
        draw.rectangle([padding + 220, legend_y, padding + 240, legend_y + 10],
                        fill=self.colors['medium_risk'], outline=None)
        draw.text((padding + 250, legend_y), "Средний риск", font=self.small_font, fill=self.colors['text'])

        # Низкий риск
        draw.rectangle([padding + 360, legend_y, padding + 380, legend_y + 10],
                        fill=self.colors['low_risk'], outline=None)
        draw.text((padding + 390, legend_y), "Низкий риск", font=self.small_font, fill=self.colors['text'])

        # Преобразуем в изображение
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode('utf-8')

def create_text_heatmap(
    text: str,
    suspicious_fragments: List[Dict[str, Any]],
    format_type: str = 'html',
    config: Dict[str, Any] = None
) -> str:
    """
    Создает тепловую карту текста в указанном формате.

    Args:
        text: Исходный текст
        suspicious_fragments: Список подозрительных фрагментов
        format_type: Тип формата ('html' или 'image')
        config: Дополнительные настройки

    Returns:
        Тепловая карта в указанном формате
    """
    heatmap = TextHeatmap(config)

    if format_type == 'html':
        return heatmap.generate_html_heatmap(text, suspicious_fragments)
    elif format_type == 'image':
        return heatmap.generate_image_heatmap(text, suspicious_fragments)
    else:
        logger.warning(f"Неизвестный формат тепловой карты: {format_type}")
        return ""
