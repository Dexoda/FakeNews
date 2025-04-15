import logging
import re
import os
import json
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class OpenSourcesChecker:
    """
    Сервис поиска и проверки информации по открытым источникам.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует проверку по открытым источникам с настройками из конфигурационного файла.

        Args:
            config: Настройки для поиска из конфигурационного файла
        """
        self.config = config
        self.session = None

        # Получаем ключи API из переменных окружения
        self.google_api_key = os.environ.get('GOOGLE_SEARCH_API_KEY', '')
        self.search_engine_id = os.environ.get('GOOGLE_SEARCH_ENGINE_ID', '')

    async def ensure_session(self):
        """
        Создает сессию для HTTP-запросов, если она еще не существует.
        """
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """
        Закрывает сессию HTTP-запросов.
        """
        if self.session:
            await self.session.close()
            self.session = None

    async def search_sources(self, claims: List[str]) -> List[Dict[str, Any]]:
        """
        Выполняет поиск по открытым источникам для проверки утверждений.

        Args:
            claims: Список утверждений для проверки

        Returns:
            Результаты поиска для каждого утверждения
        """
        await self.ensure_session()

        results = []

        for claim in claims:
            # Формируем поисковый запрос на основе утверждения
            search_query = self.generate_search_query(claim)

            # Выполняем поиск с помощью выбранных поисковых систем
            sources_results = await self.perform_search(search_query)

            # Анализируем найденные источники
            analysis = self.analyze_sources(sources_results, claim)

            results.append({
                "claim": claim,
                "search_query": search_query,
                "sources": sources_results,
                "confirmation_level": analysis["confirmation_level"],
                "confirmation_score": analysis["confirmation_score"],
                "contradicting_sources": analysis["contradicting_sources"],
                "supporting_sources": analysis["supporting_sources"],
                "neutral_sources": analysis["neutral_sources"],
                "credible_sources_count": analysis["credible_sources_count"]
            })

        return results

    def generate_search_query(self, claim: str) -> str:
        """
        Генерирует поисковый запрос на основе утверждения.

        Args:
            claim: Исходное утверждение

        Returns:
            Поисковый запрос для поисковых систем
        """
        # Очищаем утверждение от лишних символов
        clean_claim = re.sub(r'[^\w\s]', ' ', claim)
        clean_claim = re.sub(r'\s+', ' ', clean_claim).strip()

        # Ограничиваем длину запроса
        if len(clean_claim) > 150:
            # Разбиваем на слова и выбираем наиболее важные
            words = clean_claim.split()
            if len(words) > 20:
                # Оставляем только первые и последние слова, предполагая что они содержат суть
                clean_claim = ' '.join(words[:10] + words[-10:])

        # Добавляем кавычки для поиска точной фразы и ключевые слова для проверки фактов
        return f'"{clean_claim}" проверка факты достоверность'

    async def perform_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Выполняет поиск по открытым источникам.

        Args:
            query: Поисковый запрос

        Returns:
            Результаты поиска
        """
        search_engines = self.config.get('open_sources', {}).get('search_engines', ['google'])
        max_results = self.config.get('open_sources', {}).get('max_results_per_query', 10)

        # В реальной системе здесь был бы код для отправки запросов к API поисковых систем
        # Но для демонстрации мы создаем искусственные результаты поиска

        await asyncio.sleep(0.5)  # Имитация задержки запроса

        # Искусственные результаты поиска
        search_results = []

        # Добавляем результаты от разных типов источников
        # Официальные источники
        search_results.append({
            "title": f"Официальное заявление по теме: {query[:30]}...",
            "url": "https://example.gov/statement",
            "snippet": f"Официальный источник подтверждает: {query[:50]}...",
            "source_type": "official",
            "credibility": 0.9
        })

        # Новостные издания
        search_results.append({
            "title": f"Новости: {query[:30]}...",
            "url": "https://news-example.com/article",
            "snippet": f"По сообщениям, {query[:50]}...",
            "source_type": "news",
            "credibility": 0.7
        })

        # Фактчекинговые организации
        search_results.append({
            "title": f"Проверка факта: {query[:30]}...",
            "url": "https://factcheck-example.org/check",
            "snippet": f"Мы проверили утверждение: {query[:50]}... и обнаружили...",
            "source_type": "factcheck",
            "credibility": 0.8
        })

        # Случайно добавляем противоречащие источники
        if np.random.random() < 0.5:
            search_results.append({
                "title": f"Опровержение: {query[:30]}...",
                "url": "https://alternative-news.com/article",
                "snippet": f"Вопреки распространенному мнению, {query[:50]}... на самом деле не соответствует действительности...",
                "source_type": "news",
                "credibility": 0.5
            })

        # Добавляем еще несколько источников для разнообразия
        for i in range(min(5, max_results - len(search_results))):
            source_type = np.random.choice(["news", "blog", "academic", "social"])
            credibility = 0.3 + np.random.random() * 0.6  # От 0.3 до 0.9

            search_results.append({
                "title": f"Результат поиска {i+1}: {query[:20]}...",
                "url": f"https://example{i}.com/page",
                "snippet": f"Контекст результата поиска: {query[:40]}..." + ("подтверждает" if np.random.random() > 0.3 else "опровергает") + "...",
                "source_type": source_type,
                "credibility": credibility
            })

        return search_results

    def analyze_sources(self, sources: List[Dict[str, Any]], claim: str) -> Dict[str, Any]:
        """
        Анализирует найденные источники и определяет степень подтверждения утверждения.

        Args:
            sources: Список найденных источников
            claim: Исходное утверждение

        Returns:
            Результат анализа источников
        """
        supporting_sources = []
        contradicting_sources = []
        neutral_sources = []

        # Для каждого источника определяем, подтверждает ли он утверждение
        for source in sources:
            # В реальной системе здесь был бы сложный анализ содержимого источника
            # Но для демонстрации используем вероятностный подход

            snippet = source.get("snippet", "").lower()
            source_credibility = source.get("credibility", 0.5)

            # Ищем ключевые слова, указывающие на подтверждение или опровержение
            supporting_words = ["подтверждает", "доказывает", "установлено", "правда", "действительно"]
            contradicting_words = ["опровергает", "ложь", "неправда", "ошибка", "неверно", "не соответствует"]

            # Определяем отношение источника к утверждению
            supports = any(word in snippet for word in supporting_words)
            contradicts = any(word in snippet for word in contradicting_words)

            source_info = {
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "source_type": source.get("source_type", "unknown"),
                "credibility": source_credibility
            }

            if supports and not contradicts:
                supporting_sources.append(source_info)
            elif contradicts and not supports:
                contradicting_sources.append(source_info)
            else:
                neutral_sources.append(source_info)

        # Подсчитываем количество надежных источников
        credible_sources = [s for s in sources if s.get("credibility", 0) > 0.7]
        credible_sources_count = len(credible_sources)

        # Вычисляем общую оценку подтверждения
        total_score = 0
        total_weight = 0

        for source in supporting_sources:
            weight = source.get("credibility", 0.5)
            total_score += weight
            total_weight += weight

        for source in contradicting_sources:
            weight = source.get("credibility", 0.5)
            total_score -= weight
            total_weight += weight

        for source in neutral_sources:
            weight = source.get("credibility", 0.5) * 0.2  # Нейтральные источники имеют меньший вес
            total_weight += weight

        # Если есть источники, нормализуем оценку
        confirmation_score = 0.5  # Нейтральная оценка по умолчанию
        if total_weight > 0:
            raw_score = total_score / total_weight
            # Нормализуем до диапазона [0, 1]
            confirmation_score = (raw_score + 1) / 2

        # Определяем уровень подтверждения
        if confirmation_score >= 0.8:
            confirmation_level = "подтверждено"
        elif confirmation_score >= 0.6:
            confirmation_level = "вероятно правда"
        elif confirmation_score <= 0.2:
            confirmation_level = "опровергнуто"
        elif confirmation_score <= 0.4:
            confirmation_level = "вероятно ложь"
        else:
            confirmation_level = "неопределенно"

        return {
            "confirmation_level": confirmation_level,
            "confirmation_score": confirmation_score,
            "supporting_sources": supporting_sources,
            "contradicting_sources": contradicting_sources,
            "neutral_sources": neutral_sources,
            "credible_sources_count": credible_sources_count
        }

async def search_sources(claims: List[str], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Выполняет поиск по открытым источникам для проверки утверждений.

    Args:
        claims: Список утверждений для проверки
        config: Настройки для поиска

    Returns:
        Результаты поиска
    """
    if not config:
        # Временная конфигурация для случая, когда настоящая не предоставлена
        config = {
            "open_sources": {
                "enabled": True,
                "search_engines": ["google", "yandex"],
                "max_results_per_query": 10,
                "timeout": 15
            }
        }

    checker = OpenSourcesChecker(config)

    try:
        return await checker.search_sources(claims)
    finally:
        await checker.close()
