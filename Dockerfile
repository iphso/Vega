FROM python:3.11-slim

WORKDIR /work

RUN pip install --no-cache-dir \
    datasets==3.6.0 \
    huggingface_hub==0.30.2 \
    numpy==1.26.4 \
    pyarrow==17.0.0 \
    scipy==1.13.1 \
    fastapi==0.115.6 \
    uvicorn==0.34.0

COPY scripts/ /work/scripts/

ENTRYPOINT ["python3"]
