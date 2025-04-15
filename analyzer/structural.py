import logging
import re
from typing import Dict, Any, List, Optional, Tuple
import nltk
import numpy as np

# Загрузим необходимые данные для NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

async def perform_structural_analysis(text: str, url: Optional[str] = None) -> Dict[str, Any]:
    """
    Выполняет структурный анализ новостного текста.

    Аргументы:
        text: Текст новости для анализа
        url: Опциональный URL источника новости

    Возвращает:
        словарь с результатами структурного анализа
    """
    logger.info("Выполняется структурный анализ текста")

    try:
        # Разбиваем текст на составляющие
        structure_components = identify_structure_components(text)

        # Анализ заголовка (если есть)
        headline_analysis = analyze_headline(structure_components.get('headline', ''))

        # Анализ лида (если есть)
        lead_analysis = analyze_lead(structure_components.get('lead', ''))

        # Анализ основной части
        body_analysis = analyze_body(structure_components.get('body', ''))

        # Анализ соответствия стандартам журналистики
        journalism_standards = analyze_journalism_standards(
            structure_components,
            headline_analysis,
            lead_analysis,
            body_analysis
        )

        # Выявление нарушений в построении новостного материала
        structure_violations = identify_structure_violations(
            structure_components,
            headline_analysis,
            lead_analysis,
            body_analysis
        )

        # Определение источников информации
        sources_analysis = analyze_information_sources(text)

        # Расчет итоговой оценки достоверности
        credibility_score = calculate_credibility_score(
            journalism_standards,
            structure_violations,
            sources_analysis,
            structure_components
        )

        return {
            "structure_components": structure_components,
            "headline_analysis": headline_analysis,
            "lead_analysis": lead_analysis,
            "body_analysis": body_analysis,
            "journalism_standards_score": journalism_standards['overall_score'],
            "journalism_standards": journalism_standards,
            "structure_violations": structure_violations,
            "structure_quality": get_structure_quality(journalism_standards['overall_score']),
            "sources_analysis": sources_analysis,
            "credibility_score": credibility_score
        }

    except Exception as e:
        logger.error(f"Ошибка при структурном анализе: {e}")
        raise

def identify_structure_components(text: str) -> Dict[str, str]:
    """
    Разделяет текст на структурные компоненты новости.

    Возвращает словарь с компонентами: заголовок, лид, основная часть, заключение.
    """
    components = {
        'headline': '',
        'lead': '',
        'body': '',
        'conclusion': ''
    }

    # Разделяем текст на абзацы
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    if not paragraphs:
        return components

    # Предполагаем, что первый абзац может быть заголовком
    headline_candidates = [paragraphs[0]]

    # Проверяем, является ли первый абзац заголовком
    is_headline = False
    if len(paragraphs) > 1:
        first_para = paragraphs[0]
        # Типичные признаки заголовка: короткий, без точки в конце, все слова с большой буквы
        if (len(first_para) < 200 and
            not first_para.endswith('.') and
            len(first_para.split()) < 15):
            is_headline = True
            components['headline'] = first_para
            paragraphs = paragraphs[1:]

    # Если первый абзац не похож на заголовок, заголовок отсутствует

    # Проверяем наличие лида (первый абзац после заголовка)
    if paragraphs:
        first_para = paragraphs[0]
        # Типичные признаки лида: относительно короткий, содержит ключевую информацию
        if len(first_para) < 1000 and len(first_para.split()) < 100:
            components['lead'] = first_para
            paragraphs = paragraphs[1:]

    # Проверяем наличие заключения
    if len(paragraphs) > 1:
        last_para = paragraphs[-1]
        # Признаки заключения: содержит характерные фразы
        conclusion_markers = [
            'таким образом', 'в заключение', 'итак', 'подводя итог',
            'в итоге', 'в результате', 'в конце концов'
        ]

        if any(marker in last_para.lower() for marker in conclusion_markers):
            components['conclusion'] = last_para
            paragraphs = paragraphs[:-1]

    # Всё, что осталось - основная часть
    components['body'] = '\n'.join(paragraphs)

    return components

def analyze_headline(headline: str) -> Dict[str, Any]:
    """
    Анализирует заголовок новости.

    Возвращает словарь с результатами анализа заголовка.
    """
    if not headline:
        return {
            'present': False,
            'clickbait_score': 0,
            'issues': ['Заголовок отсутствует']
        }

    issues = []

    # Проверка длины заголовка
    words = headline.split()
    if len(words) < 3:
        issues.append('Слишком короткий заголовок')
    elif len(words) > 15:
        issues.append('Слишком длинный заголовок')

    # Проверка на кликбейт
    clickbait_markers = [
        'шок', 'шокирующий', 'сенсация', 'сенсационный', 'невероятный',
        'не поверите', 'фантастический', 'потрясающий', 'срочно',
        'эксклюзив', 'только у нас', 'вы должны это увидеть', 'вы не поверите',
        '!', '!!!', '???', 'официально', 'срочная новость', 'срочное сообщение'
    ]

    clickbait_score = 0
    for marker in clickbait_markers:
        if marker.lower() in headline.lower():
            clickbait_score += 0.2
            issues.append(f'Маркер кликбейта: "{marker}"')

    # Проверка использования CAPS LOCK
    uppercase_ratio = sum(1 for c in headline if c.isupper()) / max(1, len(headline))
    if uppercase_ratio > 0.5:
        clickbait_score += 0.3
        issues.append('Чрезмерное использование заглавных букв')

    # Проверка на наличие вопросов
    if '?' in headline:
        clickbait_score += 0.1
        issues.append('Заголовок в форме вопроса')

    # Проверка на наличие эмоциональных слов
    emotional_words = [
        'невероятный', 'ужасный', 'поразительный', 'шокирующий',
        'восхитительный', 'пугающий', 'кошмарный', 'душераздирающий',
        'феноменальный', 'потрясающий', 'ужасающий', 'сногсшибательный'
    ]

    for word in emotional_words:
        if word.lower() in headline.lower():
            clickbait_score += 0.1
            issues.append(f'Эмоционально окрашенное слово: "{word}"')

    # Ограничиваем максимальное значение
    clickbait_score = min(1.0, clickbait_score)

    return {
        'present': True,
        'length': len(words),
        'clickbait_score': clickbait_score,
        'issues': issues if issues else ['Без замечаний']
    }

def analyze_lead(lead: str) -> Dict[str, Any]:
    """
    Анализирует лид (вводный абзац) новости.

    Возвращает словарь с результатами анализа лида.
    """
    if not lead:
        return {
            'present': False,
            'completeness_score': 0,
            'issues': ['Лид отсутствует']
        }

    issues = []

    # Проверка наличия ключевых вопросов в лиде (5W+H)
    five_w_scores = check_five_w(lead)

    # Подсчёт охвата вопросов
    covered_questions = sum(1 for score in five_w_scores.values() if score > 0)
    completeness_score = covered_questions / 6  # 6 вопросов всего

    # Проверка длины лида
    words = lead.split()
    sentences = nltk.sent_tokenize(lead, language='russian')

    if len(words) < 20:
        issues.append('Слишком короткий лид')
    elif len(words) > 100:
        issues.append('Слишком длинный лид')

    if len(sentences) < 1:
        issues.append('Слишком мало предложений в лиде')
    elif len(sentences) > 5:
        issues.append('Слишком много предложений в лиде')

    # Проверка на повторение заголовка
    # (этот функционал требует сравнения с заголовком, поэтому здесь только заглушка)

    # Анализ пропущенных ключевых вопросов
    missing_questions = []
    if five_w_scores['who'] == 0:
        missing_questions.append('кто')
    if five_w_scores['what'] == 0:
        missing_questions.append('что')
    if five_w_scores['when'] == 0:
        missing_questions.append('когда')
    if five_w_scores['where'] == 0:
        missing_questions.append('где')
    if five_w_scores['why'] == 0:
        missing_questions.append('почему')
    if five_w_scores['how'] == 0:
        missing_questions.append('как')

    if missing_questions:
        issues.append(f'Не освещены вопросы: {", ".join(missing_questions)}')

    return {
        'present': True,
        'word_count': len(words),
        'sentence_count': len(sentences),
        'five_w_scores': five_w_scores,
        'completeness_score': completeness_score,
        'issues': issues if issues else ['Без замечаний']
    }

def check_five_w(text: str) -> Dict[str, float]:
    """
    Проверяет наличие ответов на 5W+H вопросы в тексте.

    Возвращает словарь с оценками наличия каждого вопроса.
    """
    five_w_scores = {
        'who': 0,  # Кто
        'what': 0,  # Что
        'when': 0,  # Когда
        'where': 0,  # Где
        'why': 0,    # Почему
        'how': 0     # Как
    }

    # Маркеры для "кто"
    who_markers = [
        r'\b(?:президент|министр|губернатор|мэр|директор|глава|руководитель|лидер)\b',
        r'\b(?:компания|организация|фирма|корпорация|холдинг|ведомство|учреждение)\b',
        r'\b(?:ученые|исследователи|эксперты|специалисты|аналитики)\b',
        r'\b(?:человек|люди|граждане|жители|население)\b'
    ]

    # Маркеры для "что"
    what_markers = [
        r'\b(?:сообщил|заявил|объявил|рассказал|информировал|поделился)\b',
        r'\b(?:произошло|случилось|состоялось|имело место)\b',
        r'\b(?:исследование|событие|инцидент|авария|катастрофа|встреча|переговоры)\b'
    ]

    # Маркеры для "когда"
    when_markers = [
        r'\b(?:сегодня|вчера|завтра|утром|днем|вечером|ночью)\b',
        r'\b(?:в понедельник|во вторник|в среду|в четверг|в пятницу|в субботу|в воскресенье)\b',
        r'\b(?:в январе|в феврале|в марте|в апреле|в мае|в июне|в июле|в августе|в сентябре|в октябре|в ноябре|в декабре)\b',
        r'\b(?:в \d{4} году)\b',
        r'\b(?:\d{1,2} (?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря))\b'
    ]

    # Маркеры для "где"
    where_markers = [
        r'\b(?:в Москве|в Санкт-Петербурге|в России|в США|в Европе|в Азии)\b',
        r'\b(?:в городе|в регионе|в области|в республике|в крае|в стране)\b',
        r'\b(?:на территории|на месте|в районе|в центре|на окраине)\b'
    ]

    # Маркеры для "почему"
    why_markers = [
        r'\b(?:потому что|из-за|в связи с|в результате|вследствие|по причине)\b',
        r'\b(?:благодаря|в силу|из-за того, что|по вине|под влиянием)\b'
    ]

    # Маркеры для "как"
    how_markers = [
        r'\b(?:с помощью|путем|способом|методом|посредством)\b',
        r'\b(?:при участии|при поддержке|при содействии|с использованием)\b'
    ]

    # Проверяем наличие маркеров для каждого вопроса
    for marker in who_markers:
        if re.search(marker, text, re.IGNORECASE):
            five_w_scores['who'] = 1
            break

    for marker in what_markers:
        if re.search(marker, text, re.IGNORECASE):
            five_w_scores['what'] = 1
            break

    for marker in when_markers:
        if re.search(marker, text, re.IGNORECASE):
            five_w_scores['when'] = 1
            break

    for marker in where_markers:
        if re.search(marker, text, re.IGNORECASE):
            five_w_scores['where'] = 1
            break

    for marker in why_markers:
        if re.search(marker, text, re.IGNORECASE):
            five_w_scores['why'] = 1
            break

    for marker in how_markers:
        if re.search(marker, text, re.IGNORECASE):
            five_w_scores['how'] = 1
            break

    return five_w_scores

def analyze_body(body: str) -> Dict[str, Any]:
    """
    Анализирует основную часть новости.

    Возвращает словарь с результатами анализа основной части.
    """
    if not body:
        return {
            'present': False,
            'structure_score': 0,
            'issues': ['Основная часть отсутствует']
        }

    issues = []

    # Разбиваем текст на абзацы и предложения
    paragraphs = [p.strip() for p in body.split('\n') if p.strip()]
    sentences = nltk.sent_tokenize(body, language='russian')

    # Проверка структуры основной части
    structure_score = 0.0

    # Проверка наличия достаточного количества абзацев
    if len(paragraphs) < 2:
        issues.append('Недостаточное количество абзацев')
    else:
        structure_score += 0.2

    # Проверка средней длины абзацев
    avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / max(1, len(paragraphs))
    if avg_paragraph_length < 10:
        issues.append('Слишком короткие абзацы')
    elif avg_paragraph_length > 150:
        issues.append('Слишком длинные абзацы')
    else:
        structure_score += 0.2

    # Проверка наличия цитат
    quote_markers = ['"', '«', '»', '"', '"']
    has_quotes = any(marker in body for marker in quote_markers)
    if has_quotes:
        structure_score += 0.2
    else:
        issues.append('Отсутствуют цитаты')

    # Проверка наличия фактов
    fact_markers = [
        r'\b(?:по данным|согласно|по информации|по сведениям)\b',
        r'\b(?:исследование показало|результаты показывают|статистика свидетельствует)\b',
        r'\b(?:установлено|выявлено|обнаружено|зафиксировано)\b'
    ]

    has_facts = any(re.search(marker, body, re.IGNORECASE) for marker in fact_markers)
    if has_facts:
        structure_score += 0.2
    else:
        issues.append('Мало фактологической информации')

    # Проверка наличия дат и цифр
    date_pattern = r'\b\d{1,2}(?:\.\d{1,2}(?:\.\d{2,4})?|\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+\d{4})?)\b'
    number_pattern = r'\b\d+(?:,\d+)?\b'

    has_dates = bool(re.search(date_pattern, body))
    has_numbers = bool(re.search(number_pattern, body))

    if has_dates and has_numbers:
        structure_score += 0.2
    elif not has_dates and not has_numbers:
        issues.append('Отсутствуют конкретные даты и цифры')
    else:
        structure_score += 0.1
        if not has_dates:
            issues.append('Отсутствуют конкретные даты')
        if not has_numbers:
            issues.append('Отсутствуют конкретные цифры')

    return {
        'present': True,
        'paragraph_count': len(paragraphs),
        'sentence_count': len(sentences),
        'avg_paragraph_length': avg_paragraph_length,
        'has_quotes': has_quotes,
        'has_facts': has_facts,
        'has_dates': has_dates,
        'has_numbers': has_numbers,
        'structure_score': structure_score,
        'issues': issues if issues else ['Без замечаний']
    }

def analyze_journalism_standards(
    structure_components: Dict[str, str],
    headline_analysis: Dict[str, Any],
    lead_analysis: Dict[str, Any],
    body_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Анализирует соответствие текста стандартам журналистики.

    Возвращает словарь с оценкой соответствия разным стандартам.
    """
    standards = {
        'objectivity': analyze_objectivity(structure_components),
        'balance': analyze_balance(structure_components),
        'accuracy': analyze_accuracy(structure_components, body_analysis),
        'completeness': analyze_completeness(headline_analysis, lead_analysis, body_analysis),
        'clarity': analyze_clarity(headline_analysis, lead_analysis, body_analysis)
    }

    # Расчет общей оценки соответствия стандартам
    overall_score = sum(standards.values()) / len(standards)

    # Определение сильных и слабых сторон
    strengths = [k for k, v in standards.items() if v >= 0.7]
    weaknesses = [k for k, v in standards.items() if v <= 0.3]

    return {
        'standards': standards,
        'overall_score': overall_score,
        'strengths': strengths,
        'weaknesses': weaknesses
    }

def analyze_objectivity(structure_components: Dict[str, str]) -> float:
    """
    Анализирует объективность текста.

    Возвращает оценку объективности от 0 до 1.
    """
    # Объединяем все части текста для анализа
    full_text = '\n'.join(structure_components.values())

    # Маркеры субъективности
    subjectivity_markers = [
        r'\b(?:я считаю|по моему мнению|я думаю|я полагаю|я уверен)\b',
        r'\b(?:на мой взгляд|с моей точки зрения|как мне кажется)\b',
        r'\b(?:возможно|вероятно|скорее всего|похоже|кажется)\b',
        r'\b(?:потрясающий|ужасный|невероятный|фантастический|шокирующий)\b',
        r'\b(?:восхитительный|отвратительный|замечательный|ужасающий)\b'
    ]

    # Подсчет маркеров субъективности
    subjectivity_count = 0
    for marker in subjectivity_markers:
        subjectivity_count += len(re.findall(marker, full_text, re.IGNORECASE))

    # Нормализация оценки (больше маркеров - ниже объективность)
    # Предполагаем, что 10 или более маркеров уже означают низкую объективность
    objectivity_score = max(0, 1 - (subjectivity_count / 10))

    return objectivity_score

def analyze_balance(structure_components: Dict[str, str]) -> float:
    """
    Анализирует сбалансированность представления разных точек зрения.

    Возвращает оценку сбалансированности от 0 до 1.
    """
    # Объединяем все части текста для анализа
    full_text = '\n'.join(structure_components.values())

    # Маркеры разных точек зрения
    perspective_markers = [
        r'\b(?:с одной стороны|с другой стороны)\b',
        r'\b(?:по мнению|по словам|по оценке|как заявил|как сообщил)\b',
        r'\b(?:однако|тем не менее|в то же время|несмотря на это)\b',
        r'\b(?:эксперты считают|критики утверждают|сторонники полагают)\b'
    ]

    # Подсчет маркеров разных точек зрения
    perspective_count = 0
    for marker in perspective_markers:
        perspective_count += len(re.findall(marker, full_text, re.IGNORECASE))

    # Нормализация оценки (больше маркеров - выше сбалансированность)
    # Предполагаем, что 5 или более маркеров уже означают хорошую сбалансированность
    balance_score = min(1, perspective_count / 5)

    return balance_score

def analyze_accuracy(
    structure_components: Dict[str, str],
    body_analysis: Dict[str, Any]
) -> float:
    """
    Анализирует точность информации.

    Возвращает оценку точности от 0 до 1.
    """
    # В реальной системе здесь был бы более сложный анализ с проверкой фактов
    # Здесь мы используем упрощенный подход, основанный на наличии указаний на источники

    # Объединяем все части текста для анализа
    full_text = '\n'.join(structure_components.values())

    # Маркеры указания на источники информации
    source_markers = [
        r'\b(?:по данным|согласно|по информации|по сведениям)\b',
        r'\b(?:как сообщает|как передает|как пишет|как отмечает)\b',
        r'\b(?:по словам|со ссылкой на|как заявил|как рассказал)\b',
        r'\b(?:цитата|цитирует|приводит слова|приводит данные)\b'
    ]

    # Подсчет маркеров источников
    source_count = 0
    for marker in source_markers:
        source_count += len(re.findall(marker, full_text, re.IGNORECASE))

    # Учитываем наличие дат и цифр из анализа основной части
    has_facts = body_analysis.get('has_facts', False)
    has_dates = body_analysis.get('has_dates', False)
    has_numbers = body_analysis.get('has_numbers', False)

    # Рассчитываем базовую оценку на основе количества источников
    base_score = min(1, source_count / 5)

    # Корректируем оценку на основе наличия фактов, дат и цифр
    if has_facts:
        base_score += 0.2
    if has_dates:
        base_score += 0.1
    if has_numbers:
        base_score += 0.1

    # Итоговая оценка (не более 1)
    accuracy_score = min(1, base_score)

    return accuracy_score

def analyze_completeness(
    headline_analysis: Dict[str, Any],
    lead_analysis: Dict[str, Any],
    body_analysis: Dict[str, Any]
) -> float:
    """
    Анализирует полноту освещения темы.

    Возвращает оценку полноты от 0 до 1.
    """
    completeness_score = 0.0

    # Проверяем наличие всех структурных элементов
    if headline_analysis.get('present', False):
        completeness_score += 0.2

    if lead_analysis.get('present', False):
        completeness_score += 0.2

    if body_analysis.get('present', False):
        completeness_score += 0.2

    # Учитываем полноту лида (наличие ответов на 5W+H)
    five_w_scores = lead_analysis.get('five_w_scores', {})
    covered_questions = sum(five_w_scores.values())

    # Добавляем до 0.2 балла в зависимости от покрытия вопросов
    completeness_score += (covered_questions / 6) * 0.2

    # Учитываем структуру основной части
    structure_score = body_analysis.get('structure_score', 0)

    # Добавляем до 0.2 балла в зависимости от структуры основной части
    completeness_score += structure_score * 0.2

    return min(1, completeness_score)

def analyze_clarity(
    headline_analysis: Dict[str, Any],
    lead_analysis: Dict[str, Any],
    body_analysis: Dict[str, Any]
) -> float:
    """
    Анализирует ясность и понятность текста.

    Возвращает оценку ясности от 0 до 1.
    """
    clarity_score = 0.0

    # Проверяем ясность заголовка (отсутствие кликбейта)
    if headline_analysis.get('present', False):
        clickbait_score = headline_analysis.get('clickbait_score', 0)
        clarity_score += 0.3 * (1 - clickbait_score)

    # Проверяем среднюю длину предложений в лиде
    if lead_analysis.get('present', False):
        sentences = nltk.sent_tokenize(lead_analysis.get('lead', ''), language='russian')
        words = lead_analysis.get('word_count', 0)

        if sentences:
            avg_sentence_length = words / len(sentences)

            # Оптимальная длина предложения: 15-25 слов
            if 15 <= avg_sentence_length <= 25:
                clarity_score += 0.3
            elif 10 <= avg_sentence_length < 15 or 25 < avg_sentence_length <= 30:
                clarity_score += 0.2
            elif 5 <= avg_sentence_length < 10 or 30 < avg_sentence_length <= 40:
                clarity_score += 0.1

    # Проверяем структуру абзацев в основной части
    if body_analysis.get('present', False):
        avg_paragraph_length = body_analysis.get('avg_paragraph_length', 0)

        # Оптимальная длина абзаца: 40-80 слов
        if 40 <= avg_paragraph_length <= 80:
            clarity_score += 0.4
        elif 20 <= avg_paragraph_length < 40 or 80 < avg_paragraph_length <= 120:
            clarity_score += 0.3
        elif 10 <= avg_paragraph_length < 20 or 120 < avg_paragraph_length <= 150:
            clarity_score += 0.2
        else:
            clarity_score += 0.1

    return min(1, clarity_score)

def identify_structure_violations(
    structure_components: Dict[str, str],
    headline_analysis: Dict[str, Any],
    lead_analysis: Dict[str, Any],
    body_analysis: Dict[str, Any]
) -> List[str]:
    """
    Выявляет нарушения в построении новостного материала.

    Возвращает список выявленных нарушений.
    """
    violations = []

    # Проверяем наличие обязательных компонентов
    if not headline_analysis.get('present', False):
        violations.append('Отсутствует заголовок')

    if not lead_analysis.get('present', False):
        violations.append('Отсутствует лид (вводный абзац)')

    if not body_analysis.get('present', False):
        violations.append('Отсутствует основная часть')

    # Проверяем проблемы с заголовком
    if headline_analysis.get('present', False):
        if headline_analysis.get('clickbait_score', 0) > 0.5:
            violations.append('Заголовок имеет признаки кликбейта')

    # Проверяем проблемы с лидом
    if lead_analysis.get('present', False):
        completeness_score = lead_analysis.get('completeness_score', 0)
        if completeness_score < 0.3:
            violations.append('Лид не отвечает на основные вопросы (кто, что, где, когда, почему, как)')

    # Проверяем проблемы с основной частью
    if body_analysis.get('present', False):
        if not body_analysis.get('has_quotes', False):
            violations.append('Отсутствуют цитаты источников')

        if not body_analysis.get('has_facts', False):
            violations.append('Недостаточно фактологической информации')

        if body_analysis.get('paragraph_count', 0) < 2:
            violations.append('Недостаточное количество абзацев в основной части')

    # Проверяем общую структуру
    if headline_analysis.get('present', False) and lead_analysis.get('present', False):
        # В реальной системе здесь был бы анализ соответствия заголовка и лида
        pass

    return violations

def analyze_information_sources(text: str) -> Dict[str, Any]:
    """
    Анализирует источники информации в тексте.

    Возвращает словарь с результатами анализа источников.
    """
    # Маркеры цитирования и указания на источники
    citation_patterns = [
        r'(?:«|")(.*?)(?:»|")',  # Цитаты в кавычках
        r'по\s+(?:словам|заявлению|мнению|оценке)\s+([^,\.]+)',
        r'(?:сообщил|заявил|отметил|отметила|сказал|рассказал|подчеркнул|объяснил|уточнил)\s+([^,\.]+)',
        r'(?:как|согласно)\s+(?:сообщает|пишет|передает|отмечает|заявляет|указывает)\s+([^,\.]+)'
    ]

    # Ищем все источники
    sources = []
    for pattern in citation_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            source_text = match.group(1) if len(match.groups()) > 0 else match.group(0)
            sources.append(source_text.strip())

    # Классифицируем источники
    official_sources = 0
    expert_sources = 0
    unnamed_sources = 0
    media_sources = 0

    official_markers = [
        'пресс-служб', 'министерств', 'ведомств', 'администрац', 'правительств',
        'президент', 'губернатор', 'мэр', 'глава', 'официальн'
    ]

    expert_markers = [
        'эксперт', 'специалист', 'ученый', 'исследователь', 'аналитик',
        'профессор', 'доктор', 'кандидат наук'
    ]

    unnamed_markers = [
        'источник', 'знакомый с ситуацией', 'близкий к', 'пожелавший остаться неизвестным',
        'на условиях анонимности'
    ]

    media_markers = [
        'газет', 'журнал', 'телеканал', 'радиостанц', 'информационн', 'агентств', 'издани',
        'РИА', 'ТАСС', 'Интерфакс', 'Коммерсант', 'Ведомости', 'РБК'
    ]

    for source in sources:
        source_lower = source.lower()

        if any(marker in source_lower for marker in official_markers):
            official_sources += 1
        elif any(marker in source_lower for marker in expert_markers):
            expert_sources += 1
        elif any(marker in source_lower for marker in unnamed_markers):
            unnamed_sources += 1
        elif any(marker in source_lower for marker in media_markers):
            media_sources += 1

    # Рассчитываем надежность источников
    reliability_score = 0.0
    total_sources = len(sources)

    if total_sources > 0:
        # Официальные и экспертные источники имеют больший вес
        weighted_sum = (official_sources * 0.4 +
                        expert_sources * 0.3 +
                        media_sources * 0.2 +
                        unnamed_sources * 0.1)

        reliability_score = weighted_sum / total_sources

    # Формируем анализ разнообразия источников
    return {
        'total_sources': total_sources,
        'official_sources': official_sources,
        'expert_sources': expert_sources,
        'media_sources': media_sources,
        'unnamed_sources': unnamed_sources,
        'reliability_score': reliability_score,
        'diversity_score': min(1.0, (official_sources > 0) + (expert_sources > 0) +
                                   (media_sources > 0) + (unnamed_sources > 0) / 4)
    }

def get_structure_quality(score: float) -> str:
    """
    Определяет качество структуры на основе оценки.

    Возвращает строковое описание качества.
    """
    if score >= 0.8:
        return "отличная"
    elif score >= 0.6:
        return "хорошая"
    elif score >= 0.4:
        return "удовлетворительная"
    elif score >= 0.2:
        return "слабая"
    else:
        return "плохая"

def calculate_credibility_score(
    journalism_standards: Dict[str, Any],
    structure_violations: List[str],
    sources_analysis: Dict[str, Any],
    structure_components: Dict[str, str]
) -> float:
    """
    Рассчитывает итоговую оценку достоверности на основе структурного анализа.

    Возвращает значение от 0 до 1, где 1 означает высокую вероятность достоверности.
    """
    # Веса для разных параметров
    standards_weight = 0.5      # Соответствие журналистским стандартам
    violations_weight = 0.3     # Отсутствие структурных нарушений
    sources_weight = 0.2        # Надежность источников

    # 1. Оценка соответствия журналистским стандартам
    standards_score = journalism_standards.get('overall_score', 0)

    # 2. Оценка структурных нарушений
    # Пороговое значение для количества нарушений
    violations_threshold = 5
    violations_count = min(len(structure_violations), violations_threshold)
    violations_score = 1 - (violations_count / violations_threshold)

    # 3. Оценка надежности источников
    sources_score = sources_analysis.get('reliability_score', 0)

    # Итоговая оценка
    credibility_score = (
        standards_score * standards_weight +
        violations_score * violations_weight +
        sources_score * sources_weight
    )

    return max(0, min(1, credibility_score))
