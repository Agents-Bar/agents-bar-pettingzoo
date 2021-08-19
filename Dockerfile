FROM python:3.8-slim as base
RUN apt-get update && apt-get install -y build-essential

RUN pip install --user pip==21.2.4 \
                uvicorn~=0.13.4 fastapi~=0.63.0 pydantic~=1.7.3 requests~=2.25.1 \
                pillow==8.3.1
RUN pip install --user pettingzoo[all]~=1.11.0


FROM python:3.8-slim

# Install packages directly here to limit rebuilding on code changes

COPY --from=base /root/.local /root/.local

COPY ./ /app

# set path to our python api file
ENV PATH=/root/.local:/root/.local/bin:$PATH
ENV MODULE_NAME="app.main"
LABEL agents-bar-env=v0.1.0

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--app-dir", "/app"]
