FROM python:3.11-slim
WORKDIR /srv/infinito
COPY lab_portal/requirements.txt /tmp/lab_requirements.txt
RUN pip install --no-cache-dir -r /tmp/lab_requirements.txt
COPY . /srv/infinito
ENV PYTHONUNBUFFERED=1 LAB_DATA_DIR=/data/lab LAB_HOST=0.0.0.0 LAB_PORT=8765
RUN mkdir -p /data/lab
EXPOSE 8765
CMD ["python", "-m", "lab_portal"]
