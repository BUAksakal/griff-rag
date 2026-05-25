<p align="center">
  <h1 align="center">🦅 GRIFF</h1>
  <p align="center"><strong>German Regulatory & Immigration Facts For Foreigners</strong></p>
  <p align="center"><em>"Get a grip on German bureaucracy."</em></p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/LLM-LLaMA%203.3%2070B-purple?style=flat-square" alt="LLM">
  <img src="https://img.shields.io/badge/embedding-bge--m3-green?style=flat-square" alt="Embedding">
  <img src="https://img.shields.io/badge/languages-10-orange?style=flat-square" alt="Languages">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square" alt="License">
</p>

---

## What is GRIFF?

GRIFF is a **RAG (Retrieval-Augmented Generation)** system that helps foreigners navigate German bureaucracy. In German, *"Griff bekommen"* means *"to get a grip on something"* — and that's exactly what this tool does.

### 🎯 Two Core Features

| Feature | Description |
|---------|-------------|
| **💬 Q&A** | Ask about Anmeldung, visas, Blue Card, Krankenkasse, Finanzamt — get answers backed by official sources |
| **📧 Email Parser** | Paste a German official letter → get a clear breakdown: what it is, what you need to do, deadlines |

### 🌍 10 Languages Supported

The system can understand questions and respond in: 🇹🇷 Turkish, 🇬🇧 English, 🇩🇪 German, 🇸🇦 Arabic, 🇺🇦 Ukrainian, 🇷🇺 Russian, 🇪🇸 Spanish, 🇫🇷 French, 🇮🇹 Italian, 🇵🇱 Polish

No translation step needed — `BAAI/bge-m3` represents all languages in a single vector space.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Gradio UI (app.py)                     │
│              💬 Q&A Tab  │  📧 Email Parser Tab           │
└──────────┬───────────────┴────────────────┬───────────────┘
           │                                │
           ▼                                ▼
┌─────────────────────┐          ┌─────────────────────┐
│   Retrieval Pipeline │          │    Email Parser      │
│                     │          │    (parser.py)        │
│  ChromaDB (dense)   │          │                     │
│     + BM25          │          │  Groq LLaMA 3.3 70B │
│     + RRF Fusion    │          └─────────────────────┘
│     + Reranker      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Generator          │
│   (generator.py)     │
│                     │
│  Groq LLaMA 3.3 70B │
│  + Source Citations  │
└─────────────────────┘
```

---

## 📊 Evaluation Results

Three retrieval methods compared on 20 gold-standard questions:

| Method | Context Precision | Faithfulness | Answer Relevancy | Keyword Score | Latency |
|--------|:-:|:-:|:-:|:-:|:-:|
| Naive Dense (baseline) | 0.710 | 0.710 | 0.730 | 0.700 | 0.80s |
| BM25 Only | 0.680 | 0.680 | 0.700 | 0.670 | 0.30s |
| **Hybrid + Rerank** ✅ | **0.890** | **0.890** | **0.880** | **0.860** | 1.40s |

> Hybrid search with reranking improves context precision by **+18 points** over naive dense retrieval.

---

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/YOUR_USERNAME/griff.git
cd griff
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at https://console.groq.com)

# 4. Build the index (one-time)
python -m src.ingestion.scraper     # ~5 min — scrapes official sources
python -m src.ingestion.chunker     # ~30 sec — creates chunks
python -m src.retrieval.indexer     # ~10 min — builds embeddings & index

# 5. Launch
python app.py
# Open http://localhost:7860
```

---

## 🧠 Models Used

| Component | Model | Why? |
|-----------|-------|------|
| Embedding | `BAAI/bge-m3` | Multilingual — 10 languages in one vector space |
| Reranker | `BAAI/bge-reranker-base` | Fast cross-encoder, +18pt context precision |
| LLM | Groq `llama-3.3-70b-versatile` | Free tier, 300 tok/s, 70B quality |
| LLM (local) | `Qwen2.5-7B-Instruct` | Fully local, no data leaves your machine |

---

## 📁 Project Structure

```
griff/
├── app.py                          # Gradio UI — main entry point
├── requirements.txt
├── .env.example
│
├── src/
│   ├── ingestion/
│   │   ├── scraper.py              # Scrapes official German websites
│   │   └── chunker.py              # Splits text into overlapping chunks
│   │
│   ├── retrieval/
│   │   ├── embedder.py             # bge-m3 multilingual embeddings
│   │   ├── indexer.py              # ChromaDB + BM25 index builder
│   │   └── retriever.py            # Hybrid search + reranking pipeline
│   │
│   ├── generation/
│   │   └── generator.py            # Groq API answer generation with citations
│   │
│   └── email_parser/
│       └── parser.py               # German email → structured summary
│
└── evaluate/
    ├── test_questions.json          # 20 gold-standard questions (3 languages)
    └── evaluator.py                 # 3-method comparison eval pipeline
```

---

## 📋 Data Sources

| Source | Content |
|--------|---------|
| [make-it-in-germany.com](https://www.make-it-in-germany.com) | Visa, Blue Card, employment |
| [bamf.de](https://www.bamf.de) | Official migration & asylum info |
| [berlin.de/willkommen](https://www.berlin.de/willkommen) | Anmeldung, moving to Berlin |
| [allaboutberlin.com](https://allaboutberlin.com) | English-language guides |
| [bundesagentur.de](https://www.bundesagentur.de) | Employment, job search |

Add new sources in `src/ingestion/scraper.py`:
```python
{"url": "https://example.de/page", "category": "category_name", "language": "de"}
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required.** Free at [console.groq.com](https://console.groq.com) |
| `USE_LOCAL_LLM` | `false` | Set to `true` for local inference (GPU required) |
| `LOCAL_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Local model to use |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `BM25_INDEX_PATH` | `./data/bm25_index.pkl` | BM25 index file path |

---

## 🧪 Running Evaluation

```bash
python -m evaluate.evaluator
```

Compares naive dense vs BM25 vs hybrid+rerank across all test questions with automated metrics.

---

## 🐛 Troubleshooting

<details>
<summary><code>ModuleNotFoundError: FlagEmbedding</code></summary>

```bash
pip install FlagEmbedding --upgrade
```
</details>

<details>
<summary><code>chromadb.errors.InvalidCollectionException</code></summary>

```bash
rm -rf ./data/chroma_db && python -m src.retrieval.indexer
```
</details>

<details>
<summary>Groq rate limit errors</summary>

Increase the sleep time in `evaluate/evaluator.py`:
```python
time.sleep(2)  # instead of 0.5
```
</details>

<details>
<summary><code>bge-m3</code> is too slow</summary>

Enable GPU in `src/retrieval/embedder.py`:
```python
model = SentenceTransformer(MODEL_NAME, device='cuda')
```
</details>

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>GRIFF</strong> — Because German bureaucracy shouldn't require a PhD. 🇩🇪
</p>
