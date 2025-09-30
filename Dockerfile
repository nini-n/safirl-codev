# ==========================
# SAFIRL - Docker (CPU-only)
# ==========================
FROM python:3.9-slim

# Sistem bağımlılıkları (headless MuJoCo + temel X/GL kütüphaneleri)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    libgl1 libegl1 libxrender1 libxext6 libxi6 libxxf86vm1 libxrandr2 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Bazı CI/Headless ortamlarda gereksiz render denemelerini kapatmak için:
ENV MUJOCO_GL=egl

WORKDIR /app

# Python bağımlılıkları
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir pytest ruff

# Proje dosyaları
COPY . .

# Hızlı sağlık kontrolü: linter + testler (testler MuJoCo simülasyonunu çalıştırmaz)
RUN set -ex; \
    ruff check . || true; \
    pytest -q || true

# Varsayılan komut: mevcut policy ile rapor varlıklarını üret
CMD ["python", "scripts/reproduce_all.py", "--skip-train"]
