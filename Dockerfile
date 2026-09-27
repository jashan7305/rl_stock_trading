FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --no-cache-dir uv

COPY ./pyproject.toml ./uv.lock ./
RUN uv sync --frozen --no-dev

COPY . . 

EXPOSE 8000

CMD [ ".venv/bin/streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8000" ]
