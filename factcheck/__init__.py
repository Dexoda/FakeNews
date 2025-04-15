"""
Пакет для проверки фактов и поиска по открытым источникам.

Модули:
- api_client.py: Клиент для работы с внешними API фактчекинговых сервисов
- sources.py: Модуль для поиска и проверки информации по открытым источникам
"""

from .api_client import check_facts
from .sources import search_sources

__all__ = ['check_facts', 'search_sources']
