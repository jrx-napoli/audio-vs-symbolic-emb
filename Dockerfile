# Używamy oficjalnego obrazu Pythona 3.10 z minimalnym systemem (slim)
FROM python:3.10-slim

# Instalacja zależności systemowych (ffmpeg, fluidsynth, libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fluidsynth \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Ustawienie katalogu roboczego aplikacji
WORKDIR /app

# Kopiowanie plików zależności Pythona i instalacja pakietów
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie pozostałych plików projektu (kodu źródłowego)
COPY src/ ./src/
COPY README.md ./

# Opcjonalnie: utworzenie katalogów na dane i wyniki wewnątrz kontenera
RUN mkdir -p data/raw data/processed data/embeddings results

# Ustawienie domyślnego polecenia na interpreter Pythona (ułatwia uruchamianie)
ENTRYPOINT ["python"]
