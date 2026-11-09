# Minimal isolated S0 runtime; building downloads public dependencies.
# Run with --network none, as documented in security_lab/README.md.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN python -m pip install --no-cache-dir "cryptography>=42"
COPY src/ /app/src/
COPY security_lab/ /app/security_lab/
USER 10001:10001
ENTRYPOINT ["python", "-m", "security_lab.s0"]
