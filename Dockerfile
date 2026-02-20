FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir .

FROM python:3.11-slim

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/nirs4all /usr/local/bin/nirs4all

WORKDIR /workspace

ENTRYPOINT ["python"]
