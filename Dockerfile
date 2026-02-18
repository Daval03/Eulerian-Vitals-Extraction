FROM tensorflow/tensorflow:2.16.1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Esto ahora YA NO baja torch gigante
RUN pip install --no-cache-dir -r requirements.txt

COPY Python ./Python
WORKDIR /app/Python

CMD ["pytest", "unit_test"]
