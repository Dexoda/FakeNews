import logging
import re
from typing import Dict, Any, List, Tuple, Set

import nltk
import numpy as np
from config import load_config

# Настройка логирования
logger = logging.getLogger(__name__)

# Инициализация переменных
nlp = None

# Загрузим необходимые данные для NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    import spacy
    from spacy.lang.ru.stop_words import STOP_WORDS
except ImportError:
    logging.error("spaCy не установлен. Установите его командой: pip install spacy")
    logging.error("Также необходимо загрузить русскую модель: python -m spacy download ru_core_news_md")
    raise

def load_spacy_model():
    """Лениво загружает модель spaCy при первой необходимости"""
    global nlp
    if nlp is None:
        try:
            cfg = load_config()
            model_name = cfg["analysis"]["semantic"]["model_name"]
            nlp = spacy.load(model_name)
            logger.info(f"Модель spaCy '{model_name}' успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели spaCy: {e}")
            raise
    return nlp

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
        # Загрузка модели spaCy
        global nlp
        nlp = load_spacy_model()

        # Обработка текста с помощью spaCy
        doc = nlp(text)

        # Выделение ключевых сущностей
        entities = extract_entities(doc)

        # Выделение ключевых тем (без использования noun_chunks)
        key_themes = extract_key_themes_alt(doc)

        # Анализ смысловых связей и противоречий
        coherence_analysis = analyze_coherence(doc)

        # Выделение ключевых утверждений (фактов)
        claims = extract_claims(doc)

        # Анализ смысловых противоречий
        contradictions = detect_contradictions(doc, claims)

        # Подозрительные фрагменты
        suspicious_fragments = identify_suspicious_fragments(
            text,
            doc,
            contradictions,
            coherence_analysis
        )

        # Расчет итоговой оценки достоверности
        credibility_score = calculate_credibility_score(
            coherence_analysis,
            contradictions,
            claims
        )

        return {
            "entities": entities,
            "key_themes": key_themes,
            "coherence": coherence_analysis,
            "contradictions": contradictions,
            "contradictions_count": len(contradictions),
            "identified_claims": claims,
            "suspicious_fragments": suspicious_fragments,
            "credibility_score": credibility_score
        }

    except Exception as e:
        logger.error(f"Ошибка при семантическом анализе: {e}")
        # Возвращаем базовый результат вместо ошибки
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

def extract_entities(doc) -> Dict[str, List[Dict[str, Any]]]:
    """
    Извлекает именованные сущности из текста.

    Возвращает словарь с категориями сущностей и их значениями.
    """
    entities = {
        "персоны": [],
        "организации": [],
        "локации": [],
        "даты": [],
        "другое": []
    }

    for ent in doc.ents:
        entity_info = {
            "text": ent.text,
            "start": ent.start_char,
            "end": ent.end_char
        }

        if ent.label_ in ["PER", "PERSON"]:
            entities["персоны"].append(entity_info)
        elif ent.label_ in ["ORG", "ORGANIZATION"]:
            entities["организации"].append(entity_info)
        elif ent.label_ in ["LOC", "GPE", "LOCATION"]:
            entities["локации"].append(entity_info)
        elif ent.label_ in ["DATE", "TIME"]:
            entities["даты"].append(entity_info)
        else:
            entity_info["label"] = ent.label_
            entities["другое"].append(entity_info)

    return entities

def extract_key_themes_alt(doc) -> List[str]:
    """
    Альтернативная версия извлечения ключевых тем без использования noun_chunks.

    Возвращает список ключевых тем.
    """
    # Словарь для подсчета фраз
    phrases = {}

    # Находим прилагательное + существительное комбинации (типичные именные фразы)
    i = 0
    while i < len(doc) - 1:  # Проверяем, что есть хотя бы 2 токена впереди
        # Ищем существительное
        if doc[i].pos_ in ["NOUN", "PROPN"]:
            # Если перед существительным есть прилагательное, считаем это фразой
            if i > 0 and doc[i-1].pos_ == "ADJ":
                phrase_tokens = [doc[i-1], doc[i]]
                # Фильтруем стоп-слова
                filtered_tokens = [t for t in phrase_tokens if not t.is_stop and not t.is_punct]
                if filtered_tokens:
                    phrase = " ".join(t.lemma_ for t in filtered_tokens)
                    phrases[phrase] = phrases.get(phrase, 0) + 1

            # В любом случае считаем само существительное
            if not doc[i].is_stop:
                phrases[doc[i].lemma_] = phrases.get(doc[i].lemma_, 0) + 1
        i += 1

    # Словарь для подсчета важных слов
    important_words = {}
    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "PROPN"] and not token.is_stop and len(token.text) > 3:
            word = token.lemma_
            important_words[word] = important_words.get(word, 0) + 1

    # Выбираем наиболее частотные фразы
    sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
    top_phrases = sorted_phrases[:10] if len(sorted_phrases) >= 10 else sorted_phrases

    # Выбираем наиболее частотные слова, не входящие в фразы
    top_words = []
    for word, count in sorted(important_words.items(), key=lambda x: x[1], reverse=True):
        if all(word not in phrase for phrase, _ in top_phrases):
            top_words.append((word, count))
            if len(top_words) >= 5:
                break

    # Объединяем результаты
    key_themes = [phrase for phrase, _ in top_phrases] + [word for word, _ in top_words]

    # Удаляем дубликаты и ограничиваем количество
    return list(dict.fromkeys(key_themes))[:7]

def analyze_coherence(doc) -> Dict[str, Any]:
    """
    Анализирует связность текста и логические цепочки.

    Возвращает словарь с результатами анализа связности.
    """
    # Анализ связности на уровне предложений
    sentences = list(doc.sents)

    if len(sentences) <= 1:
        return {
            "coherence_score": 1.0,
            "logical_flow": "невозможно определить (слишком короткий текст)",
            "topic_shifts": 0,
            "coherence_issues": []
        }

    # Матрица схожести между предложениями
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i, sent1 in enumerate(sentences):
        for j, sent2 in enumerate(sentences):
            if i != j:
                similarity_matrix[i][j] = sent1.similarity(sent2)

    # Средняя схожесть между соседними предложениями
    adjacent_similarity = 0
    for i in range(len(sentences) - 1):
        adjacent_similarity += similarity_matrix[i][i+1]
    adjacent_similarity /= len(sentences) - 1

    # Выявление резких смен темы
    topic_shifts = 0
    coherence_issues = []

    for i in range(len(sentences) - 1):
        # Если схожесть между соседними предложениями низкая
        if similarity_matrix[i][i+1] < 0.3:  # Порог для определения смены темы
            topic_shifts += 1
            coherence_issues.append({
                "type": "смена темы",
                "position": i+1,
                "sentence1": sentences[i].text,
                "sentence2": sentences[i+1].text,
                "similarity": similarity_matrix[i][i+1]
            })

    # Выявление несвязных предложений
    for i in range(len(sentences)):
        # Если предложение слабо связано со всеми остальными
        avg_similarity = np.sum(similarity_matrix[i]) / (len(sentences) - 1)
        if avg_similarity < 0.25:  # Порог для определения несвязного предложения
            coherence_issues.append({
                "type": "несвязное предложение",
                "position": i,
                "sentence": sentences[i].text,
                "avg_similarity": avg_similarity
            })

    # Общая оценка связности
    coherence_score = adjacent_similarity

    # Определение логического потока
    if coherence_score > 0.7:
        logical_flow = "сильный"
    elif coherence_score > 0.5:
        logical_flow = "умеренный"
    elif coherence_score > 0.3:
        logical_flow = "слабый"
    else:
        logical_flow = "нарушенный"

    return {
        "coherence_score": coherence_score,
        "logical_flow": logical_flow,
        "topic_shifts": topic_shifts,
        "coherence_issues": coherence_issues
    }

def extract_claims(doc) -> List[str]:
    """
    Извлекает ключевые утверждения (факты) из текста без использования noun_chunks.

    Возвращает список извлеченных утверждений.
    """
    claims = []

    for sent in doc.sents:
        # Пропускаем короткие предложения
        if len(sent) < 5:
            continue

        # Пропускаем вопросительные и восклицательные предложения
        if sent.text.strip().endswith('?') or sent.text.strip().endswith('!'):
            continue

        # Проверяем наличие подлежащего и сказуемого
        has_subject = False
        has_predicate = False

        for token in sent:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                has_subject = True
            if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp"]:
                has_predicate = True

        if has_subject and has_predicate:
            # Проверяем на наличие указателей субъективности
            subjectivity_markers = [
                'считаю', 'думаю', 'полагаю', 'верю', 'кажется', 'возможно',
                'вероятно', 'по-моему', 'по-видимому', 'на мой взгляд'
            ]

            if not any(marker in sent.text.lower() for marker in subjectivity_markers):
                # Это может быть фактическое утверждение
                # Очищаем текст от лишних пробелов
                claim = re.sub(r'\s+', ' ', sent.text).strip()
                claims.append(claim)

    return claims

def detect_contradictions(doc, claims: List[str]) -> List[Dict[str, Any]]:
    """
    Выявляет смысловые противоречия в тексте.

    Возвращает список обнаруженных противоречий.
    """
    contradictions = []

    # Проверка наличия загруженной модели
    global nlp
    if nlp is None:
        nlp = load_spacy_model()

    # Преобразуем утверждения в объекты spaCy для анализа
    claim_docs = [nlp(claim) for claim in claims]

    # Поиск противоречий между утверждениями
    for i, claim1_doc in enumerate(claim_docs):
        for j, claim2_doc in enumerate(claim_docs):
            if i >= j:
                continue

            # Проверяем пары утверждений на противоречия
            contradiction_type = check_contradiction(claim1_doc, claim2_doc)

            if contradiction_type:
                contradictions.append({
                    "type": contradiction_type,
                    "claim1": claims[i],
                    "claim2": claims[j],
                    "confidence": 0.7  # Базовая уверенность в противоречии
                })

    # Поиск противоречивых маркеров в тексте
    contradiction_markers = [
        (r'с одной стороны.+с другой стороны', 'противопоставление'),
        (r'не только.+но и', 'противопоставление'),
        (r'несмотря на', 'уступка'),
        (r'тем не менее', 'противоречие'),
        (r'однако', 'противоречие'),
        (r'хотя.+но', 'уступка'),
        (r'вопреки', 'противоречие')
    ]

    text = doc.text.lower()

    for pattern, marker_type in contradiction_markers:
        matches = re.finditer(pattern, text)
        for match in matches:
            # Находим ближайшие утверждения
            match_position = match.start()

            # Найти ближайшее предложение, содержащее совпадение
            containing_sentence = None
            for sent in doc.sents:
                if sent.start_char <= match_position and sent.end_char >= match_position:
                    containing_sentence = sent.text
                    break

            if containing_sentence:
                contradictions.append({
                    "type": marker_type,
                    "marker": match.group(0),
                    "context": containing_sentence,
                    "confidence": 0.5  # Более низкая уверенность, так как это только маркер
                })

    return contradictions

def check_contradiction(doc1, doc2) -> str:
    """
    Проверяет наличие противоречий между двумя утверждениями.

    Возвращает тип противоречия или None, если противоречий не найдено.
    """
    # Проверяем схожесть утверждений
    similarity = doc1.similarity(doc2)

    # Если утверждения не связаны, то противоречий нет
    if similarity < 0.5:
        return None

    # Извлекаем субъекты и предикаты
    subj1, pred1 = extract_subject_predicate(doc1)
    subj2, pred2 = extract_subject_predicate(doc2)

    # Если у утверждений одинаковые субъекты, но разные предикаты,
    # проверяем на противоречия
    if subj1 and subj2 and are_similar_entities(subj1, subj2):
        # Проверяем наличие отрицания в одном из утверждений
        neg1 = has_negation(doc1)
        neg2 = has_negation(doc2)

        # Если у одного предложения есть отрицание, а у другого нет,
        # и они говорят об одном и том же субъекте, это может быть противоречием
        if neg1 != neg2 and are_similar_predicates(pred1, pred2):
            return "прямое отрицание"

        # Если у предикатов противоположные значения
        if are_opposing_predicates(pred1, pred2):
            return "противоположные утверждения"

    # Проверяем на исключающие друг друга временные или количественные утверждения
    if has_time_contradiction(doc1, doc2):
        return "временное противоречие"

    if has_quantity_contradiction(doc1, doc2):
        return "количественное противоречие"

    return None

def extract_subject_predicate(doc):
    """
    Извлекает главное подлежащее и сказуемое из предложения.

    Возвращает кортеж (подлежащее, сказуемое) или (None, None),
    если их не удалось определить.
    """
    subject = None
    predicate = None

    # Ищем корневой глагол (сказуемое)
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            predicate = token
            break

    # Если не нашли корневой глагол, ищем любой глагол
    if predicate is None:
        for token in doc:
            if token.pos_ == "VERB":
                predicate = token
                break

    # Ищем подлежащее для найденного сказуемого
    if predicate is not None:
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"] and token.head == predicate:
                subject = token
                break

    # Если не нашли подлежащее для сказуемого, ищем любое подлежащее
    if subject is None:
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                subject = token
                break

    return subject, predicate

def are_similar_entities(entity1, entity2):
    """
    Проверяет, относятся ли два токена к одной и той же сущности.
    """
    if entity1 is None or entity2 is None:
        return False

    # Проверяем совпадение лемм
    if entity1.lemma_ == entity2.lemma_:
        return True

    # Проверяем высокую схожесть векторов
    similarity = entity1.similarity(entity2)
    return similarity > 0.8

def are_similar_predicates(pred1, pred2):
    """
    Проверяет, имеют ли два сказуемых схожее значение.
    """
    if pred1 is None or pred2 is None:
        return False

    # Проверяем совпадение лемм
    if pred1.lemma_ == pred2.lemma_:
        return True

    # Проверяем высокую схожесть векторов
    similarity = pred1.similarity(pred2)
    return similarity > 0.7

def are_opposing_predicates(pred1, pred2):
    """
    Проверяет, имеют ли два сказуемых противоположное значение.
    """
    if pred1 is None or pred2 is None:
        return False

    # Список пар противоположных глаголов
    opposing_pairs = [
        {'увеличиться', 'уменьшиться'},
        {'расти', 'падать'},
        {'начаться', 'закончиться'},
        {'открыть', 'закрыть'},
        {'разрешить', 'запретить'},
        {'согласиться', 'отказаться'},
        {'подтвердить', 'опровергнуть'},
        {'одобрить', 'осудить'}
    ]

    # Проверяем, входят ли леммы в одну из пар противоположностей
    for pair in opposing_pairs:
        if pred1.lemma_ in pair and pred2.lemma_ in pair:
            return True

    return False

def has_negation(doc):
    """
    Проверяет наличие отрицания в предложении.
    """
    for token in doc:
        # Проверяем частицу "не" или "ни"
        if token.lower_ in ["не", "ни"]:
            return True

        # Проверяем префикс "не" у глаголов
        if token.pos_ == "VERB" and token.text.startswith("не"):
            return True

    return False

def has_time_contradiction(doc1, doc2):
    """
    Проверяет наличие временных противоречий между предложениями.
    """
    # Извлекаем временные указатели из обоих предложений
    time_markers1 = extract_time_markers(doc1)
    time_markers2 = extract_time_markers(doc2)

    # Если в обоих предложениях есть временные указатели,
    # проверяем на возможные противоречия
    if time_markers1 and time_markers2:
        # Простая проверка на несовместимые временные указатели
        incompatible_pairs = [
            {'вчера', 'сегодня'}, {'сегодня', 'завтра'}, {'вчера', 'завтра'},
            {'утром', 'вечером'}, {'днём', 'ночью'}, {'утром', 'ночью'}
        ]

        for marker1 in time_markers1:
            for marker2 in time_markers2:
                for pair in incompatible_pairs:
                    if marker1 in pair and marker2 in pair and marker1 != marker2:
                        return True

    return False

def extract_time_markers(doc):
    """
    Извлекает временные указатели из предложения.
    """
    time_markers = set()

    # Список временных указателей
    time_words = {
        'сегодня', 'вчера', 'завтра', 'утром', 'днём', 'вечером', 'ночью',
        'в понедельник', 'во вторник', 'в среду', 'в четверг', 'в пятницу',
        'в субботу', 'в воскресенье', 'на прошлой неделе', 'на следующей неделе'
    }

    # Ищем временные маркеры
    text = doc.text.lower()
    for marker in time_words:
        if marker in text:
            time_markers.add(marker)

    return time_markers

def has_quantity_contradiction(doc1, doc2):
    """
    Проверяет наличие количественных противоречий между предложениями.
    """
    # Извлекаем числа из обоих предложений вместе с контекстом
    numbers1 = extract_numbers_with_context(doc1)
    numbers2 = extract_numbers_with_context(doc2)

    # Проверка наличия загруженной модели
    global nlp
    if nlp is None:
        nlp = load_spacy_model()

    # Если в обоих предложениях есть числа, проверяем на противоречия
    for num1, context1 in numbers1:
        for num2, context2 in numbers2:
            # Если контекст схож (говорим об одном и том же), но числа разные
            context_similarity = nlp(context1).similarity(nlp(context2))

            if context_similarity > 0.7 and num1 != num2:
                # Проверяем, насколько сильно отличаются числа
                difference = abs(num1 - num2) / max(1, max(num1, num2))

                # Если отличие более 10%, считаем это противоречием
                if difference > 0.1:
                    return True

    return False

def extract_numbers_with_context(doc):
    """
    Извлекает числа из предложения вместе с контекстом.
    """
    numbers_with_context = []

    for token in doc:
        if token.like_num:
            try:
                # Пытаемся преобразовать токен в число
                number = float(token.text.replace(',', '.'))

                # Извлекаем контекст (3 токена до и после числа)
                context_start = max(0, token.i - 3)
                context_end = min(len(doc), token.i + 4)
                context = doc[context_start:context_end].text

                numbers_with_context.append((number, context))
            except ValueError:
                # Если не удалось преобразовать в число, пропускаем
                pass

    return numbers_with_context

def identify_suspicious_fragments(
    text: str,
    doc,
    contradictions: List[Dict[str, Any]],
    coherence_analysis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Выявляет подозрительные фрагменты текста на основе семантического анализа.

    Возвращает список словарей с информацией о подозрительных фрагментах.
    """
    suspicious_fragments = []

    # 1. Фрагменты с противоречиями
    for contradiction in contradictions:
        if contradiction['type'] == "прямое отрицание" or contradiction['confidence'] > 0.6:
            claim1 = contradiction.get('claim1', '')
            claim2 = contradiction.get('claim2', '')

            if claim1 and claim2:
                # Находим позиции в тексте
                pos1 = text.find(claim1)
                pos2 = text.find(claim2)

                if pos1 >= 0:
                    suspicious_fragments.append({
                        'start': pos1,
                        'end': pos1 + len(claim1),
                        'text': claim1,
                        'reason': f"Противоречие ({contradiction['type']})",
                        'confidence': contradiction['confidence']
                    })

                if pos2 >= 0:
                    suspicious_fragments.append({
                        'start': pos2,
                        'end': pos2 + len(claim2),
                        'text': claim2,
                        'reason': f"Противоречие ({contradiction['type']})",
                        'confidence': contradiction['confidence']
                    })

            # Если это маркер противоречия
            context = contradiction.get('context', '')
            if context:
                pos = text.find(context)
                if pos >= 0:
                    suspicious_fragments.append({
                        'start': pos,
                        'end': pos + len(context),
                        'text': context,
                        'reason': f"Маркер противоречия ({contradiction['type']})",
                        'confidence': contradiction['confidence']
                    })

    # 2. Фрагменты с нарушениями связности
    for issue in coherence_analysis.get('coherence_issues', []):
        if issue['type'] == 'несвязное предложение' and issue.get('avg_similarity', 1) < 0.2:
            sentence = issue.get('sentence', '')
            pos = text.find(sentence)

            if pos >= 0:
                suspicious_fragments.append({
                    'start': pos,
                    'end': pos + len(sentence),
                    'text': sentence,
                    'reason': "Несвязное предложение",
                    'confidence': 0.7
                })

        elif issue['type'] == 'смена темы' and issue.get('similarity', 1) < 0.2:
            sentence = issue.get('sentence2', '')
            pos = text.find(sentence)

            if pos >= 0:
                suspicious_fragments.append({
                    'start': pos,
                    'end': pos + len(sentence),
                    'text': sentence,
                    'reason': "Резкая смена темы",
                    'confidence': 0.6
                })

    return suspicious_fragments

def calculate_credibility_score(
    coherence_analysis: Dict[str, Any],
    contradictions: List[Dict[str, Any]],
    claims: List[str]
) -> float:
    """
    Рассчитывает итоговую оценку достоверности на основе семантического анализа.

    Возвращает значение от 0 до 1, где 1 означает высокую вероятность достоверности.
    """
    # Веса для разных параметров
    coherence_weight = 0.4    # Связность текста
    contradictions_weight = 0.4  # Отсутствие противоречий
    claims_weight = 0.2       # Наличие проверяемых утверждений

    # 1. Оценка связности
    coherence_score = coherence_analysis.get('coherence_score', 0.5)

    # 2. Оценка противоречий
    # Пороговое значение для количества противоречий
    contradictions_threshold = 3
    contradictions_count = min(len(contradictions), contradictions_threshold)
    contradictions_score = 1 - (contradictions_count / contradictions_threshold)

    # 3. Оценка наличия проверяемых утверждений
    # Оптимальное количество проверяемых утверждений на 1000 слов
    optimal_claims_count = 10
    actual_claims_count = len(claims)

    # Если утверждений слишком мало или слишком много, это подозрительно
    if actual_claims_count < 2:
        claims_score = 0.3  # Очень мало утверждений
    elif actual_claims_count > optimal_claims_count * 2:
        claims_score = 0.5  # Слишком много утверждений
    else:
        claims_score = min(1.0, actual_claims_count / optimal_claims_count)

    # Итоговая оценка
    credibility_score = (
        coherence_score * coherence_weight +
        contradictions_score * contradictions_weight +
        claims_score * claims_weight
    )

    return max(0, min(1, credibility_score))
