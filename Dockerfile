# --- Stage 1: Build wheels ---
FROM python:3.11-slim-bullseye AS builder

# Рабочая директория
WORKDIR /app

# Устанавливаем инструменты для компиляции C‑модулей
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gcc \
      g++ \
      make \
      libpq-dev \
      libjpeg-dev \
      zlib1g-dev \
      fonts-dejavu \
      pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Копируем только requirements и собираем колеса пакетов
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip wheel --wheel-dir=/wheels -r requirements.txt

# --- Stage 2: Runtime image ---
FROM python:3.11-slim-bullseye

# Рабочая директория
WORKDIR /app

# Передаём ID пользователя и группы для non‑root запуска
ARG USER_ID=100001
ARG GROUP_ID=100001

# Создаём пользователя tester и группу tester
RUN groupadd -g ${GROUP_ID} tester \
 && useradd -u ${USER_ID} -g tester -s /bin/bash -m tester

# Минимальные системные зависимости (библиотеки, нужные в runtime)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libpq-dev \
      libjpeg-dev \
      zlib1g-dev \
      fonts-dejavu \
 && rm -rf /var/lib/apt/lists/*

# Копируем собранные колеса и устанавливаем их без компиляции
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-index --find-links=/wheels -r requirements.txt

# Копируем остальной код приложения
COPY . .

# Настройка NLP‑моделей (не критично, если упадёт — образ всё равно соберётся)
RUN python setup_models.py || true

# Устанавливаем права на /app для пользователя tester
RUN chown -R tester:tester /app

# Переключаемся на неправа пользователя
USER tester

# По умолчанию время в контейнере
ENV TZ=Europe/Moscow

# Команда запуска FastAPI через Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
