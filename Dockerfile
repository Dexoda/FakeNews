# --- Stage 1: Build wheels ---
FROM python:3.11-slim-bullseye AS builder

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gcc g++ make libpq-dev libjpeg-dev zlib1g-dev fonts-dejavu pkg-config curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip wheel --wheel-dir=/wheels -r requirements.txt

# Ставим spaCy 3.8.0 и модель
RUN pip install spacy==3.8.0 \
 && python -m spacy download ru_core_news_md \
 && pip uninstall -y spacy

# Проверим где лежит модель
RUN find / -type d -name "ru_core_news_md" || true

# --- Stage 2: Runtime image ---
FROM python:3.11-slim-bullseye

WORKDIR /app

ARG USER_ID=100001
ARG GROUP_ID=100001

RUN groupadd -g ${GROUP_ID} tester \
 && useradd -u ${USER_ID} -g tester -s /bin/bash -m tester

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libpq-dev libjpeg-dev zlib1g-dev fonts-dejavu \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-index --find-links=/wheels -r requirements.txt

COPY --from=builder /usr/local/lib/python3.11/site-packages/ru_core_news_md /usr/local/lib/python3.11/site-packages/ru_core_news_md

COPY . .

RUN chown -R tester:tester /app

USER tester

ENV TZ=Europe/Moscow

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
