########################################################
# Build image
FROM python:3.8-slim as base
RUN apt-get update && apt-get install -y build-essential swig

COPY ./requriements.txt /tmp/requriements.txt
RUN pip install --user -r /tmp/requriements.txt

########################################################
# Proper image
FROM python:3.8-slim

COPY --from=base /root/.local /root/.local
COPY ./ /app

# set path to our python api file
ENV PATH=/root/.local:/root/.local/bin:$PATH
ENV MODULE_NAME="app.main"

LABEL agents-bar-env=v0.1.0

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--app-dir", "/app"]
