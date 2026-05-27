FROM python:3.10-slim

# Чтобы logs Python сразу выводились в docker logs, а не буферизировались.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Системные библиотеки, нужные Pillow для работы с JPEG/PNG.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libpng16-16 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала ставим зависимости — отдельным слоем, чтобы Docker мог их кэшировать
# и не переустанавливать каждый раз, когда меняется код приложения.
COPY requirements.txt .

# Зеркало pypi от Яндекса работает в РФ. Если оно когда-нибудь перестанет
# работать, можно поменять на pypi.org или другой mirror.
# Сам PyTorch ставим с официального CDN (он работает в РФ через CDN-сеть)
# и берём CPU-сборку — она меньше по размеру и не требует CUDA.
RUN pip install --upgrade pip \
 && pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://mirror.yandex.ru/mirrors/pypi/simple/ \
        torch==2.2.2 torchvision==0.17.2 \
 && pip install \
        --index-url https://mirror.yandex.ru/mirrors/pypi/simple/ \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

RUN mkdir -p models

# Порт Gradio.
EXPOSE 7860

# Точка входа: сначала пытаемся скачать веса (если ссылка задана), потом запускаем UI.
# Используется sh -c, чтобы можно было выполнить две команды подряд.
CMD ["sh", "-c", "python -m scripts.download_weights && python -m src.app"]