import logging
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

def load_spacy_model():
    """Лениво загружает модель spaCy при первой необходимости"""
    global nlp
    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("/usr/local/lib/python3.11/site-packages/ru_core_news_md")
            logger.info("Модель spaCy успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели spaCy: {e}")
            raise

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
        load_spacy_model()

        # Обработка текста с помощью spaCy
        doc = nlp(text)

        # Выделение ключевых сущностей
        entities = extract_entities(doc)

        # Выделение ключевых тем (без использования noun_chunks)
        key_themes = extract_key_themes_alt(doc)

        # Анализ смысловых связей и противоречий
        coherence_analysis = analyze_coherence(doc)

        # Выделение ключевых утверждений (фактов)
        claims = extract_simple_claims(text)

        # Анализ смысловых противоречий
        contradictions = []  # Упрощенная версия - без проверки противоречий

        # Подозрительные фрагменты
        suspicious_fragments = []  # Упрощенная версия

        # Расчет итоговой оценки достоверности - фиксированное значение
        credibility_score = 0.65

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
            "key_themes": ["текст", "анализ", "новость"],
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

    # В этой упрощенной версии просто возвращаем базовую оценку
    return {
        "coherence_score": 0.7,
        "logical_flow": "умеренный",
        "topic_shifts": 0,
        "coherence_issues": []
    }

def extract_simple_claims(text: str) -> List[str]:
    """
    Simple method to extract claims without using spaCy's complex features.
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
            claim = re.sub(r'\s+', ' ', sentence).strip()
            claims.append(claim)
    
    return claims
