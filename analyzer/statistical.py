import logging
import re
from typing import Dict, Any, List, Tuple
from collections import Counter
import nltk
import numpy as np

# Загрузим необходимые данные для NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

async def perform_statistical_analysis(text: str) -> Dict[str, Any]:
    """
    Выполняет статистический анализ текста.

    Аргументы:
        text: Текст новости для анализа

    Возвращает:
        словарь с результатами статистического анализа
    """
    logger.info("Выполняется статистический анализ текста")

    try:
        # Очистка текста
        cleaned_text = clean_text(text)

        # Токенизация
        sentences = nltk.sent_tokenize(cleaned_text, language='russian')
        words = nltk.word_tokenize(cleaned_text, language='russian')

        # Удаление стоп-слов
        russian_stopwords = set(nltk.corpus.stopwords.words('russian'))
        filtered_words = [word.lower() for word in words if word.lower() not in russian_stopwords and word.isalpha()]

        # Вычисление статистических параметров
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / max(1, sentence_count)
        unique_words_count = len(set(filtered_words))
        lexical_diversity = unique_words_count / max(1, word_count)

        # Частотный анализ
        word_freq = Counter(filtered_words)
        most_common_words = word_freq.most_common(20)

        # Выявление аномалий в частоте слов
        frequency_anomalies = detect_frequency_anomalies(word_freq, filtered_words)

        # Анализ фактов и мнений
        fact_opinion_ratio = estimate_fact_opinion_ratio(sentences)

        # Оценка читаемости
        readability_score = calculate_readability_score(text)

        # Расчет итоговой оценки достоверности
        credibility_score = calculate_credibility_score(
            fact_opinion_ratio,
            readability_score,
            frequency_anomalies,
            lexical_diversity
        )

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "unique_words_count": unique_words_count,
            "lexical_diversity": lexical_diversity,
            "most_common_words": most_common_words,
            "frequency_anomalies": frequency_anomalies,
            "fact_opinion_ratio": fact_opinion_ratio,
            "readability_score": readability_score,
            "credibility_score": credibility_score
        }

    except Exception as e:
        logger.error(f"Ошибка при статистическом анализе: {e}")
        raise

def clean_text(text: str) -> str:
    """Очищает текст от лишних символов"""
    # Удаляем все URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Удаляем все HTML-теги
    text = re.sub(r'<.*?>', '', text)
    # Удаляем все символы, кроме букв, цифр, точек, запятых, восклицательных и вопросительных знаков
    text = re.sub(r'[^\w\s.,!?\-\']', '', text)
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_frequency_anomalies(word_freq: Counter, filtered_words: List[str]) -> List[str]:
    """
    Выявляет слова с аномальной частотой употребления.

    Возвращает список слов с аномально высокой частотой употребления.
    """
    total_words = len(filtered_words)
    anomalies = []

    # Вычисляем среднюю частоту и стандартное отклонение
    frequencies = np.array([count / total_words for word, count in word_freq.items()])
    mean_freq = np.mean(frequencies)
    std_freq = np.std(frequencies)

    # Порог для определения аномалии (напр., 2 стандартных отклонения)
    threshold = mean_freq + 2 * std_freq

    # Находим слова с частотой выше порога
    for word, count in word_freq.items():
        freq = count / total_words
        if freq > threshold and count > 3:  # Игнорируем редкие слова
            anomalies.append(word)

    return anomalies

def estimate_fact_opinion_ratio(sentences: List[str]) -> float:
    """
    Оценивает соотношение фактов и мнений в тексте.

    Возвращает значение от 0 до 1, где 1 означает только факты,
    а 0 - только мнения.
    """
    # Маркеры субъективных мнений
    opinion_markers = [
        'считаю', 'считает', 'мнение', 'полагаю', 'полагает', 'думаю', 'думает',
        'уверен', 'кажется', 'похоже', 'возможно', 'вероятно', 'наверное',
        'очевидно', 'конечно', 'несомненно', 'по-моему', 'по-видимому', 'ужасно',
        'отвратительно', 'прекрасно', 'великолепно', 'удивительно'
    ]

    # Подсчитываем предложения с маркерами мнений
    opinion_sentences = 0
    for sentence in sentences:
        if any(marker in sentence.lower() for marker in opinion_markers):
            opinion_sentences += 1

    # Вычисляем соотношение (от 0 до 1)
    fact_ratio = 1 - (opinion_sentences / max(1, len(sentences)))
    return fact_ratio

def calculate_readability_score(text: str) -> float:
    """
    Рассчитывает оценку читаемости текста (адаптированный индекс Флеша для русского языка).

    Возвращает значение от 0 до 1, где 1 означает очень простой текст,
    а 0 - очень сложный.
    """
    sentences = nltk.sent_tokenize(text, language='russian')
    words = nltk.word_tokenize(text, language='russian')

    # Подсчет слогов (приблизительно, для русского языка)
    def count_syllables(word):
        vowels = 'аеёиоуыэюя'
        syllables = 0
        prev_is_vowel = False

        for char in word.lower():
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                syllables += 1
            prev_is_vowel = is_vowel

        return max(1, syllables)

    syllable_count = sum(count_syllables(word) for word in words if word.isalpha())

    # Адаптированная формула читаемости
    if len(sentences) == 0 or len(words) == 0:
        return 0.5  # Средняя оценка для пустого текста

    asl = len(words) / len(sentences)  # Average Sentence Length
    asw = syllable_count / len(words)  # Average Syllables per Word

    # Адаптированный индекс Флеша для русского языка
    # Значение от 0 (сложный) до 100 (простой)
    readability = 206.835 - (1.3 * asl) - (60.1 * asw)

    # Нормализуем значение до диапазона от 0 до 1
    normalized_readability = max(0, min(1, readability / 100))

    return normalized_readability

def calculate_credibility_score(
    fact_opinion_ratio: float,
    readability_score: float,
    frequency_anomalies: List[str],
    lexical_diversity: float
) -> float:
    """
    Рассчитывает итоговую оценку достоверности на основе статистических параметров.

    Возвращает значение от 0 до 1, где 1 означает высокую вероятность достоверности.
    """
    # Веса для разных параметров
    fact_opinion_weight = 0.4  # Соотношение фактов и мнений
    readability_weight = 0.2   # Читаемость
    anomalies_weight = 0.2     # Частотные аномалии
    diversity_weight = 0.2     # Лексическое разнообразие

    # Оценка аномалий (обратная зависимость)
    anomalies_score = 1.0 if not frequency_anomalies else max(0, 1 - (len(frequency_anomalies) / 10))

    # Лексическое разнообразие (нормализация)
    # Обычно для нормального текста характерно разнообразие около 0.4-0.6
    # Низкое разнообразие может указывать на искусственный текст
    diversity_score = 0
    if lexical_diversity < 0.2:
        diversity_score = lexical_diversity * 2  # Низкое разнообразие
    elif lexical_diversity > 0.8:
        diversity_score = 2 - lexical_diversity  # Слишком высокое разнообразие
    else:
        # Оптимальное значение около 0.5
        diversity_score = 1 - 2 * abs(lexical_diversity - 0.5)

    # Итоговая оценка
    credibility_score = (
        fact_opinion_ratio * fact_opinion_weight +
        readability_score * readability_weight +
        anomalies_score * anomalies_weight +
        diversity_score * diversity_weight
    )

    return max(0, min(1, credibility_score))
