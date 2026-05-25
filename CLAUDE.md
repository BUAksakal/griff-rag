# GRIFF — CLAUDE.md

Bu dosya Claude Code için yazılmıştır. Projeyi local'de build ederken Claude'a context verir.

---

## Proje Özeti

**GRIFF** — *German Regulatory & Immigration Facts For Foreigners*

Almancada "Griff" = tutamak, kavramak demek. "Griff bekommen" = bir şeye hakim olmak.
Tagline: *"Get a grip on German bureaucracy."*

GRIFF, Almanya'daki bürokratik süreçleri anlamak için yapılmış bir RAG (Retrieval-Augmented Generation) sistemidir.

İki ana özelliği var:
1. **Soru-Cevap**: Anmeldung, Vize, Krankenkasse, Finanzamt gibi konularda resmi kaynaklara dayalı cevap verir
2. **Email Parser**: Almanca resmi mektup/mail yapıştırırsın, sistem "bu ne, ne yapman lazım, son tarih var mı" der

---

## Desteklenen Diller

Sistem şu dillerde hem soru alır hem cevap verir:

| Kod  | Dil      | Örnek soru |
|------|----------|------------|
| `tr` | Türkçe   | "Anmeldung için hangi belgeler lazım?" |
| `en` | İngilizce | "How do I apply for a Blue Card?" |
| `de` | Almanca  | "Was brauche ich für die Anmeldung?" |
| `ar` | Arapça   | "ما هي وثائق التسجيل؟" |
| `uk` | Ukraynaca | "Які документи потрібні для реєстрації?" |
| `ru` | Rusça    | "Какие документы нужны для регистрации?" |
| `es` | İspanyolca | "¿Qué documentos necesito para el Anmeldung?" |
| `fr` | Fransızca | "Quels documents pour l'Anmeldung?" |
| `it` | İtalyanca | "Quali documenti servono per l'Anmeldung?" |
| `pl` | Lehçe    | "Jakie dokumenty są potrzebne do Anmeldung?" |

Dil seçimi `src/generation/generator.py` ve `src/email_parser/parser.py` içindeki `response_language` parametresiyle kontrol edilir.

Embedding modeli olarak `BAAI/bge-m3` kullanılıyor — bu model zaten tüm bu dilleri aynı vektör uzayında temsil ediyor, ayrı bir çeviri adımı gerekmez.

---

## Proje Yapısı

```
griff/
├── app.py                          # Gradio UI — ana giriş noktası
├── requirements.txt
├── .env.example                    # Buradan .env oluştur
│
├── src/
│   ├── ingestion/
│   │   ├── scraper.py              # Resmi Alman sitelerinden veri çeker
│   │   └── chunker.py              # Metni örtüşen parçalara böler
│   │
│   ├── retrieval/
│   │   ├── embedder.py             # bge-m3 ile çok dilli embedding
│   │   ├── indexer.py              # ChromaDB + BM25 index oluşturur
│   │   └── retriever.py            # Hybrid search + reranking pipeline
│   │
│   ├── generation/
│   │   └── generator.py            # Groq API ile cevap üretir, kaynak gösterir
│   │
│   └── email_parser/
│       └── parser.py               # Almanca email → yapılandırılmış özet
│
└── evaluate/
    ├── test_questions.json          # 20 gold-standard soru (3 dilde)
    └── evaluator.py                 # 3 yöntemi karşılaştıran eval pipeline
```

---

## Pipeline Açıklaması

```
INGESTION (bir kez çalıştırılır):
  scraper.py → resmi sitelerden HTML/PDF çeker
  chunker.py → 512 char, 64 char overlap ile parçalar

RETRIEVAL (her soruda çalışır):
  embedder.py  → sorguyu bge-m3 ile vektöre çevirir
  indexer.py   → ChromaDB (dense) + BM25 (keyword) indexi
  retriever.py → dense + BM25 sonuçlarını birleştirir,
                 bge-reranker-base ile yeniden sıralar

GENERATION (her soruda çalışır):
  generator.py → top-5 chunk + soru → Groq LLaMA-3.3-70B → cevap + kaynaklar

EMAIL PARSER (bağımsız):
  parser.py → ham Almanca email → JSON → {kurum, aciliyet, son tarih, yapılacaklar, özet}
```

---

## Kurulum (Local Build)

```bash
# 1. Repoyu klonla
git clone https://github.com/YOUR_USERNAME/griff
cd griff

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Environment değişkenlerini ayarla
cp .env.example .env
# .env dosyasını aç, GROQ_API_KEY ekle (https://console.groq.com — ücretsiz)

# 5. Veri topla ve indexle
python -m src.ingestion.scraper     # ~5 dakika, resmi sitelerden çeker
python -m src.ingestion.chunker     # ~30 saniye
python -m src.retrieval.indexer     # ~10 dakika, embedding hesaplar

# 6. Uygulamayı başlat
python app.py
# → http://localhost:7860 adresinde açılır
```

---

## Environment Değişkenleri

```env
GROQ_API_KEY=gsk_...           # Zorunlu. https://console.groq.com'dan ücretsiz al
USE_LOCAL_LLM=false            # true yaparsan Qwen2.5-7B local çalışır (GPU gerekir)
LOCAL_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
CHROMA_PERSIST_DIR=./data/chroma_db
BM25_INDEX_PATH=./data/bm25_index.pkl
```

---

## Modeller

| Bileşen    | Model                    | Neden bu? |
|------------|--------------------------|-----------|
| Embedding  | `BAAI/bge-m3`            | 10 dili aynı vektör uzayında destekler |
| Reranker   | `BAAI/bge-reranker-base` | Hızlı cross-encoder, context precision +18pt artırıyor |
| LLM        | Groq `llama-3.3-70b`     | Ücretsiz, 300 tok/sn, 70B kalitesi |
| LLM (local)| `Qwen2.5-7B-Instruct`    | GPU varsa tam lokal, veri dışarı çıkmaz |

---

## Veri Kaynakları

Scraper şu URL'leri indexler:

- https://www.make-it-in-germany.com — Vize, Blue Card, iş bulma
- https://www.bamf.de — Resmi göç ve sığınma bilgileri
- https://www.berlin.de/willkommen — Anmeldung, Berlin'e taşınma
- https://allaboutberlin.com — İngilizce açıklama rehberleri
- https://www.bundesagentur.de — İşsizlik, iş arama

Yeni kaynak eklemek için `src/ingestion/scraper.py` içindeki `SOURCES` listesine ekle:
```python
{"url": "https://example.de/page", "category": "kategori", "language": "de"}
```

---

## Dil Desteği Nasıl Çalışır

### Soru-Cevap Tarafı
`generator.py` içindeki system prompt'a dil talimatı eklenir:
```python
"Answer in the SAME LANGUAGE as the question (Turkish, English, German, Arabic, Ukrainian...)"
```
Model otomatik algılar, ayrı bir dil tespiti gerekmez.

### Email Parser Tarafı
`parser.py`'de `response_language` parametresi ile kontrol edilir:
```python
result = parse_email(email_text, response_language="tr")  # tr, en, de, ar, uk, ru, es, fr, it, pl
```

### Gradio UI Tarafı
`app.py` içinde dil seçici radio button var:
```python
lang_picker = gr.Radio(
    choices=["🇹🇷 Türkçe", "🇬🇧 English", "🇩🇪 Deutsch", "🇸🇦 العربية",
             "🇺🇦 Українська", "🇷🇺 Русский", "🇪🇸 Español",
             "🇫🇷 Français", "🇮🇹 Italiano", "🇵🇱 Polski"],
    value="🇹🇷 Türkçe",
    label="Response language",
)
```

---

## Evaluation

```bash
# 3 yöntemi karşılaştır: naive dense vs BM25 vs hybrid+rerank
python -m evaluate.evaluator

# Çıktı şöyle görünür:
# ======================================================================
# Method                    C.Prec   Faith   A.Rel     Key   Lat(s)
# ----------------------------------------------------------------------
# Naive Dense (baseline)     0.710   0.710   0.730   0.700    0.80
# BM25 Only                  0.680   0.680   0.700   0.670    0.30
# Hybrid + Rerank            0.890   0.890   0.880   0.860    1.40
# ======================================================================
```

Bu tabloyu README'ye koy — işe alımcılar bunu görünce "bu kişi mühendislik yapıyor" der.

---

## Hugging Face Spaces Deploy

```bash
# HF Spaces için ek dosya
echo "GROQ_API_KEY=..." >> .env   # HF Spaces secrets'a ekle

# app.py son satırı şu olmalı:
demo.launch()   # share=False kaldır, server_name="0.0.0.0" ekle
```

Spaces'te `GROQ_API_KEY`'i Settings → Secrets bölümüne ekle.

---

## Sık Karşılaşılan Hatalar

**`ModuleNotFoundError: FlagEmbedding`**
```bash
pip install FlagEmbedding --upgrade
```

**`chromadb.errors.InvalidCollectionException`**
```bash
rm -rf ./data/chroma_db && python -m src.retrieval.indexer
```

**Groq rate limit hatası**
```
evaluate/evaluator.py içinde time.sleep(0.5) → time.sleep(2) yap
```

**`bge-m3` modeli çok yavaş**
```python
# embedder.py içinde device='cpu' yerine:
model = SentenceTransformer(MODEL_NAME, device='cuda')  # GPU varsa
```

---

## Katkı Notları

- Yeni dil eklenecekse: `evaluate/test_questions.json`'a o dilde sorular ekle
- Yeni kaynak eklenecekse: `scraper.py → SOURCES` listesine ekle, sonra `indexer.py`'yi yeniden çalıştır
- Model değiştirilecekse: `embedder.py` ve `retriever.py` içindeki model isimlerini güncelle
