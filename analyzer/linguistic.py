import logging
import re
from typing import Dict, Any, List, Tuple
import nltk
import numpy as np

# Don't try to import Dostoevsky models that might fail
# from dostoevsky.tokenization import RegexTokenizer
# from dostoevsky.models import FastTextSocialNetworkModel

# Загрузим необходимые данные для NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

# Flag to indicate if sentiment analysis is available
sentiment_available = False

# Try to import and initialize Dostoevsky, but don't fail if it doesn't work
try:
    from dostoevsky.tokenization import RegexTokenizer
    from dostoevsky.models import FastTextSocialNetworkModel

    # Initialize tokenizer
    tokenizer = RegexTokenizer()

    # Try to load model
    try:
        model = FastTextSocialNetworkModel(tokenizer=tokenizer)
        # Test the model with a simple prediction to ensure it works
        test_result = model.predict(['Тестовое предложение'])
        if test_result:
            sentiment_available = True
            logger.info("Модель анализа тональности успешно загружена и проверена")
    except Exception as e:
        logger.warning(f"Не удалось загрузить модель анализа тональности: {e}")
        model = None
except Exception as e:
    logger.warning(f"Не удалось импортировать модули Dostoevsky: {e}")
    model = None

async def perform_linguistic_analysis(text: str) -> Dict[str, Any]:
    """
    Выполняет лингвистический анализ текста.

    Аргументы:
        text: Текст новости для анализа

    Возвращает:
        словарь с результатами лингвистического анализа
    """
    logger.info("Выполняется лингвистический анализ текста")

    try:
        # Токенизация
        sentences = nltk.sent_tokenize(text, language='russian')

        # Анализ тональности (если доступен)
        if sentiment_available and model is not None:
            try:
                sentiment_results = analyze_sentiment(sentences)
                logger.info("Анализ тональности выполнен успешно")
            except Exception as e:
                logger.warning(f"Ошибка при анализе тональности: {e}")
                sentiment_results = get_default_sentiment()
        else:
            logger.info("Анализ тональности недоступен, используется заглушка")
            sentiment_results = get_default_sentiment()

        # Выделение эмоционально окрашенной лексики
        emotional_markers = detect_emotional_markers(text)

        # Выявление манипулятивных речевых конструкций
        manipulative_constructs = detect_manipulative_constructs(sentences)

        # Анализ синтаксических особенностей
        syntax_analysis = analyze_syntax(sentences)

        # Подозрительные фрагменты
        suspicious_fragments = identify_suspicious_fragments(
            text,
            sentences,
            sentiment_results,
            manipulative_constructs
        )

        # Расчет итоговой оценки достоверности
        credibility_score = calculate_credibility_score(
            sentiment_results,
            emotional_markers,
            manipulative_constructs
        )

        return {
            "sentiment": sentiment_results["overall_sentiment"],
            "sentiment_scores": sentiment_results["scores"],
            "emotional_tone": get_emotional_tone(emotional_markers),
            "emotional_markers": emotional_markers,
            "manipulative_constructs": manipulative_constructs,
            "manipulative_constructs_count": len(manipulative_constructs),
            "syntax_analysis": syntax_analysis,
            "suspicious_fragments": suspicious_fragments,
            "credibility_score": credibility_score
        }

    except Exception as e:
        logger.error(f"Ошибка при лингвистическом анализе: {e}")
        # Возвращаем базовые результаты вместо ошибки
        return {
            "sentiment": "нейтральная",
            "sentiment_scores": {
                'negative': 0.0,
                'positive': 0.0,
                'neutral': 1.0,
                'speech': 0.0,
                'skip': 0.0
            },
            "emotional_tone": "нейтральный",
            "emotional_markers": {},
            "manipulative_constructs": [],
            "manipulative_constructs_count": 0,
            "syntax_analysis": {
                "avg_sentence_length": 0.0,
                "complex_sentences_ratio": 0.0,
                "passive_voice_ratio": 0.0
            },
            "suspicious_fragments": [],
            "credibility_score": 0.5,
            "error": str(e)
        }

def get_default_sentiment():
    """
    Возвращает заглушку для анализа тональности.
    """
    return {
        "overall_sentiment": "нейтральная",
        "scores": {
            'negative': 0.0,
            'positive': 0.0,
            'neutral': 1.0,
            'speech': 0.0,
            'skip': 0.0
        },
        "variation": 0.0,
        "sentence_sentiments": []
    }

def analyze_sentiment(sentences: List[str]) -> Dict[str, Any]:
    """
    Определяет тональность текста.

    Возвращает словарь с общей тональностью и подробными оценками.
    """
    if not sentiment_available or model is None:
        return get_default_sentiment()

    try:
        # Используем Dostoevsky для анализа тональности
        results = model.predict(sentences)

        # Получаем средние значения по всем предложениям
        total_negative = 0
        total_positive = 0
        total_neutral = 0
        total_speech = 0
        total_skip = 0

        for result in results:
            total_negative += result.get('negative', 0)
            total_positive += result.get('positive', 0)
            total_neutral += result.get('neutral', 0)
            total_speech += result.get('speech', 0)
            total_skip += result.get('skip', 0)

        # Рассчитываем средние значения
        sentence_count = max(1, len(sentences))
        avg_negative = total_negative / sentence_count
        avg_positive = total_positive / sentence_count
        avg_neutral = total_neutral / sentence_count
        avg_speech = total_speech / sentence_count
        avg_skip = total_skip / sentence_count

        # Определяем преобладающую тональность
        scores = {
            'negative': avg_negative,
            'positive': avg_positive,
            'neutral': avg_neutral,
            'speech': avg_speech,
            'skip': avg_skip
        }

        # Определяем общую тональность
        max_score = max(avg_negative, avg_positive, avg_neutral)
        if max_score == avg_neutral:
            overall_sentiment = "нейтральная"
        elif max_score == avg_positive:
            overall_sentiment = "позитивная"
        else:
            overall_sentiment = "негативная"

        # Анализируем разброс эмоций
        sentiment_variation = np.std([avg_negative, avg_positive, avg_neutral])

        return {
            "overall_sentiment": overall_sentiment,
            "scores": scores,
            "variation": sentiment_variation,
            "sentence_sentiments": [
                {"text": sentences[i], "sentiment": list(results[i].items())}
                for i in range(min(len(sentences), len(results)))
            ]
        }
    except Exception as e:
        logger.error(f"Ошибка при анализе тональности: {e}")
        # Возвращаем нейтральные значения в случае ошибки
        return get_default_sentiment()

def detect_emotional_markers(text: str) -> Dict[str, List[str]]:
    """
    Выделяет эмоционально окрашенную лексику в тексте.

    Возвращает словарь с категориями эмоций и соответствующими словами.
    """
    # Словари эмоционально окрашенных слов
    emotional_words = {
        'страх': [
            'ужас', 'страх', 'боязнь', 'паника', 'ужасный', 'страшный', 'пугающий',
            'катастрофа', 'кошмар', 'трагедия', 'крах', 'опасность'
        ],
        'гнев': [
            'ярость', 'гнев', 'злость', 'ненависть', 'возмущение', 'враг', 'противник',
            'чудовищный', 'отвратительный', 'вопиюще', 'агрессивно'
        ],
        'радость': [
            'счастье', 'радость', 'восторг', 'удовольствие', 'ликование', 'успех',
            'победа', 'достижение', 'замечательный', 'отличный', 'превосходный'
        ],
        'печаль': [
            'грусть', 'печаль', 'тоска', 'уныние', 'разочарование', 'потеря',
            'поражение', 'крах', 'неудача', 'провал', 'горький', 'печальный'
        ],
        'удивление': [
            'шок', 'удивление', 'изумление', 'потрясение', 'неожиданность',
            'невероятный', 'немыслимый', 'небывалый', 'ошеломляющий'
        ],
        'преувеличение': [
            'абсолютно', 'чрезвычайно', 'непременно', 'колоссальный', 'огромный',
            'безусловно', 'несомненно', 'невероятно', 'грандиозный', 'тотальный'
        ]
    }

    # Преобразуем текст в нижний регистр и разбиваем на слова
    words = re.findall(r'\b\w+\b', text.lower())

    # Поиск эмоционально окрашенных слов
    found_markers = {category: [] for category in emotional_words}

    for word in words:
        for category, markers in emotional_words.items():
            if word in markers:
                found_markers[category].append(word)

    return found_markers

def get_emotional_tone(emotional_markers: Dict[str, List[str]]) -> str:
    """
    Определяет общую эмоциональную окраску текста на основе найденных маркеров.

    Возвращает строку, описывающую эмоциональный тон.
    """
    # Подсчет количества маркеров по каждой категории
    counts = {category: len(markers) for category, markers in emotional_markers.items()}

    # Общее количество эмоциональных маркеров
    total_markers = sum(counts.values())

    if total_markers == 0:
        return "нейтральный"

    # Определение преобладающей эмоциональной категории
    max_category = max(counts, key=counts.get)
    max_count = counts[max_category]

    # Проверка на явное преобладание (более 50% от всех маркеров)
    if max_count > total_markers * 0.5:
        # Проверяем категорию для определения тона
        if max_category in ['страх', 'гнев', 'печаль']:
            return f"негативный (преобладает {max_category})"
        elif max_category in ['радость']:
            return f"позитивный (преобладает {max_category})"
        elif max_category in ['удивление']:
            return f"удивление"
        elif max_category in ['преувеличение']:
            return f"преувеличение"

    # Если нет явного преобладания, но есть много негативных маркеров
    negative_count = counts['страх'] + counts['гнев'] + counts['печаль']
    positive_count = counts['радость']

    if negative_count > positive_count * 2:
        return "преимущественно негативный"
    elif positive_count > negative_count * 2:
        return "преимущественно позитивный"
    elif counts['преувеличение'] > total_markers * 0.3:
        return "эмоционально преувеличенный"
    else:
        return "смешанный"

def detect_manipulative_constructs(sentences: List[str]) -> List[Dict[str, Any]]:
    """
    Выявляет манипулятивные речевые конструкции в тексте.

    Возвращает список словарей с информацией о выявленных конструкциях.
    """
    manipulative_patterns = [
        {
            'name': 'обобщение',
            'patterns': [
                r'\bвсе\s+(?:знают|понимают|согласны|считают)',
                r'\bкаждый\s+(?:знает|понимает|согласен|считает)',
                r'\bникто\s+не\s+(?:знает|понимает|согласен|считает)',
                r'\bвсегда\b', r'\bникогда\b', r'\bвезде\b', r'\bнигде\b',
                r'\bлюбой\s+человек\b', r'\bлюбой\s+эксперт\b'
            ],
            'confidence': 0.7,
            'description': 'Необоснованное обобщение'
        },
        {
            'name': 'апелляция к авторитету',
            'patterns': [
                r'\bэксперты\s+(?:говорят|считают|подтверждают)',
                r'\bученые\s+(?:доказали|обнаружили|выяснили)',
                r'\bисследования\s+(?:показывают|доказывают)',
                r'\bпо\s+(?:мнению|словам|утверждению)\s+(?:ученых|экспертов)',
                r'\bсогласно\s+(?:данным|исследованиям|источникам)'
            ],
            'confidence': 0.6,
            'description': 'Апелляция к неназванным авторитетам'
        },
        {
            'name': 'эмоциональное давление',
            'patterns': [
                r'\bшокирующий\b', r'\bужасающий\b', r'\bпотрясающий\b',
                r'\bневероятный\b', r'\bсенсационный\b', r'\bскандальный\b',
                r'\bнеслыханный\b', r'\bбеспрецедентный\b', r'\bкатастрофический\b'
            ],
            'confidence': 0.65,
            'description': 'Использование эмоционально окрашенных прилагательных'
        },
        {
            'name': 'ложная дилемма',
            'patterns': [
                r'\bлибо\s+[\w\s]+,\s+либо\s+[\w\s]+',
                r'\bили\s+[\w\s]+,\s+или\s+[\w\s]+',
                r'\bтолько\s+два\s+(?:выхода|пути|варианта)',
                r'\bнет\s+другого\s+(?:выхода|пути|выбора)',
                r'\bтретьего\s+не\s+дано\b'
            ],
            'confidence': 0.75,
            'description': 'Представление только двух крайних вариантов'
        },
        {
            'name': 'логические ошибки',
            'patterns': [
                r'\bа\s+значит\b', r'\bследовательно\b', r'\bпоэтому\b',
                r'\bтаким\s+образом\b', r'\bотсюда\s+следует\b',
                r'\bтогда\s+получается\b'
            ],
            'confidence': 0.4,  # Низкая уверенность, так как может быть корректным
            'description': 'Потенциально некорректный логический вывод'
        }
    ]

    found_constructs = []

    for i, sentence in enumerate(sentences):
        for pattern_group in manipulative_patterns:
            for pattern in pattern_group['patterns']:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    found_constructs.append({
                        'type': pattern_group['name'],
                        'text': sentence,
                        'match': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'sentence_index': i,
                        'confidence': pattern_group['confidence'],
                        'description': pattern_group['description']
                    })

    return found_constructs

def analyze_syntax(sentences: List[str]) -> Dict[str, Any]:
    """
    Анализирует синтаксические особенности текста.

    Возвращает словарь с результатами синтаксического анализа.
    """
    # Анализ средней длины предложения
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))

    # Анализ сложности структуры предложений
    complex_sentences = 0
    for sentence in sentences:
        # Подсчет запятых и союзов как индикаторов сложности
        commas = sentence.count(',')
        conjunctions = len(re.findall(r'\b(и|но|или|а|хотя|если|поскольку|чтобы)\b', sentence, re.IGNORECASE))

        # Считаем предложение сложным, если в нем много запятых и союзов
        if commas + conjunctions > 3:
            complex_sentences += 1

    complex_sentences_ratio = complex_sentences / max(1, len(sentences))

    # Анализ использования пассивного залога
    passive_voice_sentences = 0
    for sentence in sentences:
        # Упрощенное обнаружение пассивного залога для русского языка
        passive_patterns = [
            r'\b\w+(?:ется|ются|ится|ятся)\b',  # для глаголов типа "считается", "делается"
            r'\b(?:был|была|было|были)\s+\w+н[ыоа]\b',  # для форм типа "был сделан", "была найдена"
            r'\b(?:будет|будут)\s+\w+н[ыоа]\b'  # для форм типа "будет сделано", "будут найдены"
        ]

        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in passive_patterns):
            passive_voice_sentences += 1

    passive_voice_ratio = passive_voice_sentences / max(1, len(sentences))

    return {
        "avg_sentence_length": avg_sentence_length,
        "complex_sentences_ratio": complex_sentences_ratio,
        "passive_voice_ratio": passive_voice_ratio
    }

def identify_suspicious_fragments(
    text: str,
    sentences: List[str],
    sentiment_results: Dict[str, Any],
    manipulative_constructs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Выявляет подозрительные фрагменты текста на основе лингвистического анализа.

    Возвращает список словарей с информацией о подозрительных фрагментах.
    """
    suspicious_fragments = []

    # 1. Фрагменты с манипулятивными конструкциями
    for construct in manipulative_constructs:
        if construct['confidence'] >= 0.6:  # Только с высокой уверенностью
            # Находим позицию в полном тексте
            sentence_start = text.find(construct['text'])
            if sentence_start >= 0:
                fragment_start = sentence_start + construct['start']
                fragment_end = sentence_start + construct['end']

                suspicious_fragments.append({
                    'start': fragment_start,
                    'end': fragment_end,
                    'text': construct['match'],
                    'reason': f"Манипулятивная конструкция: {construct['description']}",
                    'confidence': construct['confidence']
                })

    # 2. Фрагменты с сильными эмоциональными выбросами
    if sentiment_available and 'sentence_sentiments' in sentiment_results:
        try:
            for i, sentence_sentiment in enumerate(sentiment_results.get('sentence_sentiments', [])):
                # Проверяем наличие сильных эмоциональных выбросов
                sentiment_dict = dict(sentence_sentiment['sentiment'])
                if (sentiment_dict.get('negative', 0) > 0.7 or
                    sentiment_dict.get('positive', 0) > 0.8):

                    sentence_text = sentence_sentiment['text']
                    sentence_start = text.find(sentence_text)

                    if sentence_start >= 0:
                        suspicious_fragments.append({
                            'start': sentence_start,
                            'end': sentence_start + len(sentence_text),
                            'text': sentence_text,
                            'reason': "Сильная эмоциональная окраска",
                            'confidence': max(sentiment_dict.get('negative', 0),
                                            sentiment_dict.get('positive', 0))
                        })
        except Exception as e:
            logger.warning(f"Ошибка при анализе эмоциональных выбросов: {e}")

    return suspicious_fragments

def calculate_credibility_score(
    sentiment_results: Dict[str, Any],
    emotional_markers: Dict[str, List[str]],
    manipulative_constructs: List[Dict[str, Any]]
) -> float:
    """
    Рассчитывает итоговую оценку достоверности на основе лингвистического анализа.

    Возвращает значение от 0 до 1, где 1 означает высокую вероятность достоверности.
    """
    # Веса для разных параметров
    sentiment_weight = 0.3      # Тональность текста
    emotional_weight = 0.3      # Эмоциональная окраска
    manipulative_weight = 0.4   # Манипулятивные конструкции

    # 1. Оценка тональности
    # Нейтральные тексты получают высокий балл, сильно отклоняющиеся - низкий
    sentiment_variation = sentiment_results.get('variation', 0)
    sentiment_score = 1 - min(1, sentiment_variation * 3)  # Чем больше разброс, тем ниже балл

    # 2. Оценка эмоциональной окраски
    # Подсчет общего количества эмоциональных маркеров
    total_emotional_markers = sum(len(markers) for markers in emotional_markers.values())

    # Пороговые значения для количества эмоциональных маркеров
    emotional_threshold = 10  # Допустимое количество на 1000 слов

    # Предполагаем, что текст содержит примерно 1000 слов
    emotional_markers_score = 1 - min(1, total_emotional_markers / emotional_threshold)

    # 3. Оценка манипулятивных конструкций
    # Пороговые значения для количества манипулятивных конструкций
    manipulative_threshold = 5  # Допустимое количество

    manipulative_score = 1 - min(1, len(manipulative_constructs) / manipulative_threshold)

    # Итоговая оценка
    credibility_score = (
        sentiment_score * sentiment_weight +
        emotional_markers_score * emotional_weight +
        manipulative_score * manipulative_weight
    )

    return max(0, min(1, credibility_score))
