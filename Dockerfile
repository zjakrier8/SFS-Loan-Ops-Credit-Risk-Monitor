FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir \
    streamlit \
    anthropic \
    httpx \
    plotly \
    pandas \
    sq-pysnowflake \
    block-cloud-auth \
    slack_sdk

ENV PORT=8501
ENV USE_PYSNOWFLAKE=1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]