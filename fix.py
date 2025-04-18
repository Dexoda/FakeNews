import os

def fix_semantic_file():
    file_path = 'analyzer/semantic.py'
    
    # Проверяем, существует ли файл
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден")
        return False
    
    # Создаем резервную копию
    backup_path = f"{file_path}.bak"
    with open(file_path, 'r') as file:
        content = file.read()
    
    with open(backup_path, 'w') as file:
        file.write(content)
    print(f"Создана резервная копия {backup_path}")
    
    # Заменяем проблемную функцию
    search_text = "def extract_key_themes_alt"
    end_marker = "return list(dict.fromkeys(key_themes))"
    
    replacement = """def extract_key_themes_alt(doc) -> List[str]:
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

    # Удаляем дубликаты и ограничиваем количество"""
    
    start_idx = content.find(search_text)
    if start_idx == -1:
        print("Функция extract_key_themes_alt не найдена")
        return False
    
    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        print("Не найден конец функции")
        return False
    
    end_idx += len(end_marker)
    new_content = content[:start_idx] + replacement + content[end_idx:]
    
    # Записываем исправленный файл
    with open(file_path, 'w') as file:
        file.write(new_content)
    
    print(f"Файл {file_path} успешно исправлен")
    return True

if __name__ == "__main__":
    if fix_semantic_file():
        print("Исправление успешно применено. Теперь перезапустите API-сервер:")
        print("docker-compose restart api-server")
    else:
        print("Произошла ошибка при исправлении файла")
