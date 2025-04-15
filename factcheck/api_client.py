import logging
import os
import json
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class FactCheckAPIClient:
    """
    Клиент для работы с внешними API фактчекинговых сервисов.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует клиент с настройками из конфигурационного файла.

        Args:
            config: Настройки для API-клиента из конфигурационного файла
        """
        self.config = config
        self.session = None

        # Получаем ключи API из переменных окружения
        self.google_api_key = os.environ.get('GOOGLE_FACT_CHECK_API_KEY', '')

        if not self.google_api_key and config.get('apis', {}).get('google_fact_check', {}).get('enabled', False):
            logger.warning("Google Fact Check API включен в конфигурации, но ключ API не найден в переменных окружения")

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

    async def check_facts(self, claims: List[str]) -> List[Dict[str, Any]]:
        """
        Проверяет список утверждений через внешние API.

        Args:
            claims: Список утверждений для проверки

        Returns:
            Список результатов проверки для каждого утверждения
        """
        await self.ensure_session()

        results = []

        for claim in claims:
            # Проверяем, достаточно ли длинное утверждение для проверки
            if len(claim.split()) < 5:
                logger.debug(f"Пропуск слишком короткого утверждения: {claim}")
                continue

            # Параллельная проверка через разные API
            api_tasks = []

            # Google Fact Check API
            if self.config.get('apis', {}).get('google_fact_check', {}).get('enabled', False):
                api_tasks.append(self.check_with_google_fact_check(claim))

            # ClaimReview API
            if self.config.get('apis', {}).get('claim_review', {}).get('enabled', False):
                api_tasks.append(self.check_with_claim_review(claim))

            # Ожидаем выполнения всех запросов
            api_results = await asyncio.gather(*api_tasks, return_exceptions=True)

            # Фильтруем ошибки и объединяем результаты
            valid_results = []
            for result in api_results:
                if isinstance(result, Exception):
                    logger.error(f"Ошибка проверки факта: {result}")
                else:
                    valid_results.append(result)

            # Структурируем результаты проверки
            fact_check_result = {
                "claim": claim,
                "status": "не проверено",
                "sources": [],
                "rating": 0.5,  # Нейтральная оценка по умолчанию
                "api_results": valid_results
            }

            # Анализируем результаты и определяем итоговый статус
            if valid_results:
                fact_check_result.update(self.analyze_fact_check_results(valid_results))

            results.append(fact_check_result)

        return results

    async def check_with_google_fact_check(self, claim: str) -> Dict[str, Any]:
        """
        Проверяет утверждение через Google Fact Check Tools API.

        Args:
            claim: Утверждение для проверки

        Returns:
            Результат проверки
        """
        if not self.google_api_key:
            return {
                "api": "google_fact_check",
                "status": "error",
                "message": "API ключ не настроен"
            }

        base_url = self.config.get('apis', {}).get('google_fact_check', {}).get('base_url',
                                'https://factchecktools.googleapis.com/v1alpha1/claims:search')
        timeout = self.config.get('apis', {}).get('google_fact_check', {}).get('timeout', 10)

        params = {
            'query': claim,
            'key': self.google_api_key,
            'languageCode': 'ru'  # По умолчанию используем русский язык
        }

        try:
            async with self.session.get(base_url, params=params, timeout=timeout) as response:
                response_data = await response.json()

                return {
                    "api": "google_fact_check",
                    "status": "success",
                    "data": response_data
                }
        except asyncio.TimeoutError:
            logger.warning(f"Таймаут при запросе к Google Fact Check API: {claim[:50]}...")
            return {
                "api": "google_fact_check",
                "status": "timeout",
                "message": "Превышено время ожидания ответа"
            }
        except Exception as e:
            logger.error(f"Ошибка при запросе к Google Fact Check API: {e}")
            return {
                "api": "google_fact_check",
                "status": "error",
                "message": str(e)
            }

    async def check_with_claim_review(self, claim: str) -> Dict[str, Any]:
        """
        Проверяет утверждение через ClaimReview API.

        Args:
            claim: Утверждение для проверки

        Returns:
            Результат проверки
        """
        base_url = self.config.get('apis', {}).get('claim_review', {}).get('base_url',
                              'https://example.com/api/claimreview')
        timeout = self.config.get('apis', {}).get('claim_review', {}).get('timeout', 10)

        # Этот API - заглушка, в реальной системе здесь был бы работающий API
        # Но для демонстрации имитируем ответ

        try:
            await asyncio.sleep(0.5)  # Имитация задержки API

            # Имитация ответа API
            response_data = {
                "found": False,
                "message": "ClaimReview API находится в разработке или недоступен"
            }

            return {
                "api": "claim_review",
                "status": "success",
                "data": response_data
            }
        except asyncio.TimeoutError:
            logger.warning(f"Таймаут при запросе к ClaimReview API: {claim[:50]}...")
            return {
                "api": "claim_review",
                "status": "timeout",
                "message": "Превышено время ожидания ответа"
            }
        except Exception as e:
            logger.error(f"Ошибка при запросе к ClaimReview API: {e}")
            return {
                "api": "claim_review",
                "status": "error",
                "message": str(e)
            }

    def analyze_fact_check_results(self, api_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализирует результаты проверки фактов из разных API и определяет итоговый статус.

        Args:
            api_results: Список результатов проверки из разных API

        Returns:
            Обобщенный результат проверки
        """
        sources = []
        ratings = []

        # Анализируем результаты от Google Fact Check API
        for result in api_results:
            if result.get('api') == 'google_fact_check' and result.get('status') == 'success':
                data = result.get('data', {})
                claims = data.get('claims', [])

                for claim_info in claims:
                    # Источник проверки
                    if 'claimReview' in claim_info:
                        for review in claim_info.get('claimReview', []):
                            source = {
                                "name": review.get('publisher', {}).get('name', 'Неизвестный источник'),
                                "url": review.get('url', ''),
                                "rating": review.get('textualRating', 'Не указано'),
                                "title": claim_info.get('text', '')
                            }
                            sources.append(source)

                            # Определяем числовую оценку на основе текстового рейтинга
                            rating = self.convert_textual_rating_to_numeric(review.get('textualRating', ''))
                            if rating is not None:
                                ratings.append(rating)

        # Вычисляем итоговый статус на основе найденных оценок
        status = "не проверено"
        rating = 0.5  # Нейтральная оценка по умолчанию

        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            rating = avg_rating

            if avg_rating >= 0.8:
                status = "подтверждено"
            elif avg_rating >= 0.6:
                status = "вероятно правда"
            elif avg_rating <= 0.2:
                status = "опровергнуто"
            elif avg_rating <= 0.4:
                status = "вероятно ложь"
            else:
                status = "спорно"
        elif sources:
            # Если есть источники, но нет числовых оценок
            status = "найдены проверки"

        return {
            "status": status,
            "sources": sources,
            "rating": rating
        }

    def convert_textual_rating_to_numeric(self, textual_rating: str) -> Optional[float]:
        """
        Преобразует текстовую оценку достоверности в числовую.

        Args:
            textual_rating: Текстовая оценка достоверности

        Returns:
            Числовая оценка от 0 до 1, или None если не удалось определить
        """
        textual_rating = textual_rating.lower()

        # Позитивные оценки
        if any(term in textual_rating for term in ['правда', 'верно', 'правдиво', 'достоверно', 'подтверждено']):
            return 0.9

        # Вероятно правда
        if any(term in textual_rating for term in ['вероятно правда', 'скорее правда', 'в основном правда']):
            return 0.7

        # Смешанные оценки
        if any(term in textual_rating for term in ['частично', 'смешанно', 'спорно', '50/50']):
            return 0.5

        # Вероятно ложь
        if any(term in textual_rating for term in ['вероятно ложь', 'скорее ложь', 'в основном ложь']):
            return 0.3

        # Негативные оценки
        if any(term in textual_rating for term in ['ложь', 'неправда', 'фейк', 'обман', 'недостоверно', 'опровергнуто']):
            return 0.1

        # Если не удалось определить
        return None

async def check_facts(claims: List[str], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Проверяет список утверждений через внешние API.

    Args:
        claims: Список утверждений для проверки
        config: Настройки для API-клиента

    Returns:
        Список результатов проверки
    """
    if not config:
        # Временная конфигурация для случая, когда настоящая не предоставлена
        config = {
            "apis": {
                "google_fact_check": {
                    "enabled": False,
                    "base_url": "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                    "timeout": 10
                },
                "claim_review": {
                    "enabled": False,
                    "base_url": "https://example.com/api/claimreview",
                    "timeout": 10
                }
            }
        }

    client = FactCheckAPIClient(config)

    try:
        return await client.check_facts(claims)
    finally:
        await client.close()
