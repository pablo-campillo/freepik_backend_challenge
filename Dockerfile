#FROM huggingface/transformers-pytorch-gpu
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=true

ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN python -c 'from urllib.request import urlopen; print(urlopen("https://install.python-poetry.org").read().decode())' | python -
COPY src ./src
COPY poetry.lock ./poetry.lock
COPY pyproject.toml ./pyproject.toml
COPY README.md ./README.md
COPY .env ./.env
ADD git-base-textcaps ./git-base-textcaps

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN poetry install --no-interaction --no-ansi -vvv

ENV PATH=".venv/bin:$PATH"
EXPOSE 8888
ENTRYPOINT ["/entrypoint.sh"]