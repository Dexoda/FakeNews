#!/usr/bin/env python3

import os
import re
import sys

def fix_semantic_py():
    """Fix the semantic.py file to avoid using noun_chunks for Russian language."""
    
    # Путь к файлу semantic.py
    semantic_file = "analyzer/semantic.py"
    
    if not os.path.exists(semantic_file):
        print(f"Ошибка: Файл {semantic_file} не найден.")
        return False
    
    print(f"Модифицирую файл {semantic_file}...")
    
    # Читаем содержимое файла
    with open(semantic_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Создаем резервную копию
    backup_file = f"{semantic_file}.bak"
    with open(backup_file, 'w', encoding='utf-8') as file:
        file.write(content)
    print(f"Создана резервная копия: {backup_file}")
    
    # Заменяем функцию extract_key_themes, которая использует noun_chunks
    extract_key_themes_function = r"""def extract_key_themes_alt\(doc\).*?return list\(dict\.fromkeys\(key_themes\)\)\[:7\]"""
    
    # Новая реализация функции
    new_extract_key_themes = """def extract_key_themes_alt(doc) -> List[str]:
    \"\"\"
    Альтернативная версия извлечения ключевых тем без использования noun_chunks.

    Возвращает список ключевых тем.
    \"\"\"
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
    return list(dict.fromkeys(key_themes))[:7]"""
    
    # Применяем замену с использованием регулярных выражений
    new_content = re.sub(extract_key_themes_function, new_extract_key_themes, content, flags=re.DOTALL)
    
    # Также добавим функцию extract_simple_claims, если она отсутствует
    if "def extract_simple_claims" not in new_content:
        extract_claims_function = r"""def extract_claims\(doc\).*?return claims"""
        
        new_extract_claims = """def extract_claims(doc) -> List[str]:
    \"\"\"
    Извлекает ключевые утверждения (факты) из текста без использования noun_chunks.

    Возвращает список извлеченных утверждений.
    \"\"\"
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
                claim = re.sub(r'\\\\s+', ' ', sent.text).strip()
                claims.append(claim)

    return claims"""
        
        new_content = re.sub(extract_claims_function, new_extract_claims, new_content, flags=re.DOTALL)
    
    # Записываем обновленный контент
    with open(semantic_file, 'w', encoding='utf-8') as file:
        file.write(new_content)
    
    print(f"Файл {semantic_file} успешно модифицирован!")
    return True

if __name__ == "__main__":
    print("Фикс для проблемы с noun_chunks в русском языке")
    success = fix_semantic_py()
    
    if success:
        print("""
Исправление успешно применено!

Для перезапуска сервисов выполните:
docker-compose restart api-server
        """)
    else:
        print("Произошла ошибка при применении исправления.")
