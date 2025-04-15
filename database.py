import logging
import os
from typing import Dict, Any, List, Optional, Tuple
import asyncpg
import json

logger = logging.getLogger(__name__)

class Database:
    """
    Класс для работы с базой данных PostgreSQL.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует подключение к базе данных с настройками из конфигурационного файла.

        Args:
            config: Настройки базы данных из конфигурационного файла
        """
        self.config = config
        self.pool = None

    async def connect(self):
        """
        Устанавливает соединение с базой данных.
        """
        if self.pool is not None:
            return

        try:
            # Получаем параметры подключения
            db_host = os.environ.get('DB_HOST', self.config.get('host', 'localhost'))
            db_port = int(os.environ.get('DB_PORT', self.config.get('port', 5432)))
            db_name = os.environ.get('DB_NAME', self.config.get('name', 'fakedetector'))
            db_user = os.environ.get('DB_USER', self.config.get('user', 'fakedetector'))
            db_password = os.environ.get('DB_PASSWORD', '')

            # Дополнительные параметры
            connect_timeout = self.config.get('connect_timeout', 5)
            pool_size = self.config.get('pool_size', 10)

            # Создаем пул соединений
            self.pool = await asyncpg.create_pool(
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password,
                timeout=connect_timeout,
                min_size=2,
                max_size=pool_size,
                command_timeout=10
            )

            logger.info(f"Успешное подключение к базе данных: {db_host}:{db_port}/{db_name}")

            # Инициализируем схему базы данных
            await self.init_schema()

        except Exception as e:
            logger.error(f"Ошибка при подключении к базе данных: {e}")
            raise

    async def close(self):
        """
        Закрывает соединение с базой данных.
        """
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
            logger.info("Соединение с базой данных закрыто")

    async def init_schema(self):
        """
        Инициализирует схему базы данных, создавая необходимые таблицы, если они не существуют.
        """
        try:
            async with self.pool.acquire() as conn:
                # Создаем таблицу для хранения проанализированных текстов
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS analyzed_texts (
                        id SERIAL PRIMARY KEY,
                        text TEXT NOT NULL,
                        text_hash VARCHAR(64) NOT NULL,
                        credibility_score FLOAT NOT NULL,
                        credibility_level VARCHAR(50) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(text_hash)
                    )
                ''')

                # Создаем таблицу для хранения результатов анализа
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id SERIAL PRIMARY KEY,
                        text_id INTEGER NOT NULL REFERENCES analyzed_texts(id) ON DELETE CASCADE,
                        analysis_type VARCHAR(50) NOT NULL,
                        analysis_data JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(text_id, analysis_type)
                    )
                ''')

                # Создаем таблицу для хранения результатов проверки фактов
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS factcheck_results (
                        id SERIAL PRIMARY KEY,
                        text_id INTEGER NOT NULL REFERENCES analyzed_texts(id) ON DELETE CASCADE,
                        claim TEXT NOT NULL,
                        status VARCHAR(50) NOT NULL,
                        rating FLOAT NOT NULL,
                        sources JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Создаем таблицу для хранения обратной связи пользователей
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id SERIAL PRIMARY KEY,
                        text_id INTEGER NOT NULL REFERENCES analyzed_texts(id) ON DELETE CASCADE,
                        user_id VARCHAR(100) NOT NULL,
                        rating SMALLINT NOT NULL,
                        comments TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(text_id, user_id)
                    )
                ''')

                # Создаем индексы для оптимизации запросов
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_analyzed_texts_hash ON analyzed_texts(text_hash)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_analysis_results_text_id ON analysis_results(text_id)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_factcheck_results_text_id ON factcheck_results(text_id)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_text_id ON user_feedback(text_id)')

                logger.info("Схема базы данных инициализирована")

        except Exception as e:
            logger.error(f"Ошибка при инициализации схемы базы данных: {e}")
            raise

    async def save_analysis_results(
        self,
        text: str,
        credibility_score: float,
        credibility_level: str,
        analysis_results: Dict[str, Any],
        factcheck_results: List[Dict[str, Any]]
    ) -> int:
        """
        Сохраняет результаты анализа в базу данных.

        Args:
            text: Исходный текст новости
            credibility_score: Итоговая оценка достоверности
            credibility_level: Уровень достоверности
            analysis_results: Результаты анализа текста
            factcheck_results: Результаты проверки фактов

        Returns:
            ID сохраненной записи
        """
        # Проверяем подключение к базе данных
        if self.pool is None:
            await self.connect()

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Вычисляем хеш текста для идентификации дубликатов
                    import hashlib
                    text_hash = hashlib.sha256(text.encode()).hexdigest()

                    # Проверяем, не анализировался ли этот текст ранее
                    existing_text = await conn.fetchrow(
                        'SELECT id FROM analyzed_texts WHERE text_hash = $1',
                        text_hash
                    )

                    if existing_text:
                        text_id = existing_text['id']
                        logger.info(f"Найден существующий анализ текста с ID: {text_id}")

                        # Обновляем оценку достоверности
                        await conn.execute(
                            '''
                            UPDATE analyzed_texts
                            SET credibility_score = $1, credibility_level = $2
                            WHERE id = $3
                            ''',
                            credibility_score, credibility_level, text_id
                        )
                    else:
                        # Сохраняем новый текст
                        text_id = await conn.fetchval(
                            '''
                            INSERT INTO analyzed_texts (text, text_hash, credibility_score, credibility_level)
                            VALUES ($1, $2, $3, $4)
                            RETURNING id
                            ''',
                            text, text_hash, credibility_score, credibility_level
                        )
                        logger.info(f"Создан новый анализ текста с ID: {text_id}")

                    # Сохраняем результаты разных типов анализа
                    for analysis_type, data in analysis_results.items():
                        # Проверяем, существует ли уже этот тип анализа для данного текста
                        existing_analysis = await conn.fetchrow(
                            'SELECT id FROM analysis_results WHERE text_id = $1 AND analysis_type = $2',
                            text_id, analysis_type
                        )

                        if existing_analysis:
                            # Обновляем существующий анализ
                            await conn.execute(
                                '''
                                UPDATE analysis_results
                                SET analysis_data = $1
                                WHERE id = $2
                                ''',
                                json.dumps(data), existing_analysis['id']
                            )
                        else:
                            # Вставляем новый анализ
                            await conn.execute(
                                '''
                                INSERT INTO analysis_results (text_id, analysis_type, analysis_data)
                                VALUES ($1, $2, $3)
                                ''',
                                text_id, analysis_type, json.dumps(data)
                            )

                    # Сохраняем результаты проверки фактов
                    # Сначала удаляем предыдущие результаты для этого текста
                    await conn.execute(
                        'DELETE FROM factcheck_results WHERE text_id = $1',
                        text_id
                    )

                    # Затем вставляем новые результаты
                    for factcheck in factcheck_results:
                        await conn.execute(
                            '''
                            INSERT INTO factcheck_results
                            (text_id, claim, status, rating, sources)
                            VALUES ($1, $2, $3, $4, $5)
                            ''',
                            text_id,
                            factcheck.get('claim', ''),
                            factcheck.get('status', 'не проверено'),
                            factcheck.get('rating', 0.5),
                            json.dumps(factcheck.get('sources', []))
                        )

                    return text_id

        except Exception as e:
            logger.error(f"Ошибка при сохранении результатов анализа: {e}")
            raise

    async def get_analysis_by_id(self, text_id: int) -> Optional[Dict[str, Any]]:
        """
        Получает результаты анализа по ID.

        Args:
            text_id: ID текста

        Returns:
            Результаты анализа или None, если не найдено
        """
        # Проверяем подключение к базе данных
        if self.pool is None:
            await self.connect()

        try:
            async with self.pool.acquire() as conn:
                # Получаем информацию о тексте
                text_info = await conn.fetchrow(
                    '''
                    SELECT id, text, credibility_score, credibility_level, created_at
                    FROM analyzed_texts
                    WHERE id = $1
                    ''',
                    text_id
                )

                if not text_info:
                    return None

                # Получаем результаты анализа
                analysis_results = await conn.fetch(
                    '''
                    SELECT analysis_type, analysis_data
                    FROM analysis_results
                    WHERE text_id = $1
                    ''',
                    text_id
                )

                # Получаем результаты проверки фактов
                factcheck_results = await conn.fetch(
                    '''
                    SELECT claim, status, rating, sources
                    FROM factcheck_results
                    WHERE text_id = $1
                    ''',
                    text_id
                )

                # Формируем результат
                result = {
                    'id': text_info['id'],
                    'text': text_info['text'],
                    'credibility_score': text_info['credibility_score'],
                    'credibility_level': text_info['credibility_level'],
                    'created_at': text_info['created_at'].isoformat(),
                    'analysis_results': {},
                    'factcheck_results': []
                }

                # Добавляем результаты анализа
                for analysis in analysis_results:
                    result['analysis_results'][analysis['analysis_type']] = json.loads(analysis['analysis_data'])

                # Добавляем результаты проверки фактов
                for factcheck in factcheck_results:
                    result['factcheck_results'].append({
                        'claim': factcheck['claim'],
                        'status': factcheck['status'],
                        'rating': factcheck['rating'],
                        'sources': json.loads(factcheck['sources'])
                    })

                return result

        except Exception as e:
            logger.error(f"Ошибка при получении результатов анализа: {e}")
            raise

    async def save_user_feedback(
        self,
        text_id: int,
        user_id: str,
        rating: int,
        comments: Optional[str] = None
    ) -> bool:
        """
        Сохраняет обратную связь пользователя.

        Args:
            text_id: ID проанализированного текста
            user_id: ID пользователя
            rating: Оценка от пользователя (1-5)
            comments: Дополнительные комментарии

        Returns:
            True в случае успеха, False в случае ошибки
        """
        # Проверяем подключение к базе данных
        if self.pool is None:
            await self.connect()

        try:
            async with self.pool.acquire() as conn:
                # Проверяем существование текста
                text_exists = await conn.fetchval(
                    'SELECT 1 FROM analyzed_texts WHERE id = $1',
                    text_id
                )

                if not text_exists:
                    logger.warning(f"Попытка добавить обратную связь к несуществующему тексту (ID: {text_id})")
                    return False

                # Проверяем, не оставлял ли пользователь уже обратную связь
                existing_feedback = await conn.fetchval(
                    'SELECT 1 FROM user_feedback WHERE text_id = $1 AND user_id = $2',
                    text_id, user_id
                )

                if existing_feedback:
                    # Обновляем существующую обратную связь
                    await conn.execute(
                        '''
                        UPDATE user_feedback
                        SET rating = $1, comments = $2
                        WHERE text_id = $3 AND user_id = $4
                        ''',
                        rating, comments, text_id, user_id
                    )
                else:
                    # Добавляем новую обратную связь
                    await conn.execute(
                        '''
                        INSERT INTO user_feedback (text_id, user_id, rating, comments)
                        VALUES ($1, $2, $3, $4)
                        ''',
                        text_id, user_id, rating, comments
                    )

                return True

        except Exception as e:
            logger.error(f"Ошибка при сохранении обратной связи: {e}")
            return False

    async def get_previous_analyses(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Получает список предыдущих анализов.

        Args:
            limit: Максимальное количество результатов
            offset: Смещение для пагинации

        Returns:
            Список результатов анализа
        """
        # Проверяем подключение к базе данных
        if self.pool is None:
            await self.connect()

        try:
            async with self.pool.acquire() as conn:
                # Получаем список анализов
                analyses = await conn.fetch(
                    '''
                    SELECT id, text, credibility_score, credibility_level, created_at
                    FROM analyzed_texts
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                    ''',
                    limit, offset
                )

                # Формируем результат
                result = []
                for analysis in analyses:
                    # Вычисляем среднюю оценку обратной связи
                    avg_rating = await conn.fetchval(
                        '''
                        SELECT AVG(rating)
                        FROM user_feedback
                        WHERE text_id = $1
                        ''',
                        analysis['id']
                    )

                    # Добавляем информацию о тексте
                    result.append({
                        'id': analysis['id'],
                        'text_preview': analysis['text'][:200] + ('...' if len(analysis['text']) > 200 else ''),
                        'credibility_score': analysis['credibility_score'],
                        'credibility_level': analysis['credibility_level'],
                        'created_at': analysis['created_at'].isoformat(),
                        'avg_user_rating': float(avg_rating) if avg_rating is not None else None
                    })

                return result

        except Exception as e:
            logger.error(f"Ошибка при получении списка анализов: {e}")
            return []

# Singleton-экземпляр базы данных
_db_instance = None

async def init_db(config: Dict[str, Any]):
    """
    Инициализирует подключение к базе данных.

    Args:
        config: Настройки базы данных из конфигурационного файла
    """
    global _db_instance

    if _db_instance is None:
        _db_instance = Database(config.get('database', {}))

    await _db_instance.connect()

    logger.info("База данных инициализирована")

async def get_db() -> Database:
    """
    Получает экземпляр базы данных.

    Returns:
        Экземпляр базы данных
    """
    global _db_instance

    if _db_instance is None:
        raise RuntimeError("База данных не инициализирована. Сначала вызовите init_db().")

    return _db_instance

async def close_db():
    """
    Закрывает подключение к базе данных.
    """
    global _db_instance

    if _db_instance is not None:
        await _db_instance.close()
        _db_instance = None

        logger.info("Подключение к базе данных закрыто")
