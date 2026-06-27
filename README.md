# OCR Through RAG Using CVIP

An end-to-end intelligent document search system that combines Computer Vision, OCR, Agentic AI, and Retrieval-Augmented Generation (RAG). Upload a PDF book and search for any text — the system returns exact **page numbers**, **paragraph numbers**, and **line numbers** where the text appears.

---

## What It Does

1. User uploads a PDF book via web interface
2. System converts each page to an image (PyMuPDF)
3. OpenCV applies binarization and noise removal
4. **4 parallel OCR workers** simultaneously detect Text, Table, Figure, and Font regions
5. **Agent 1** validates and filters garbage chunks
6. **Agent 2** corrects OCR errors using a local LLM
7. Clean text chunks are embedded and stored in ChromaDB vector database
8. User searches any text — **Agent 3** retrieves matches and returns formatted results with exact locations

---

## Architecture

```
PDF Upload
    │
    ▼
PDF → Images (PyMuPDF, 2x zoom)
    │
    ▼
Preprocessing (OpenCV)
├── Grayscale conversion
├── Adaptive Binarization
└── Noise Removal
    │
    ▼
Parallel Segmentation (4 workers simultaneously)
├── Worker 1: Text regions    (PSM 3)
├── Worker 2: Table regions   (PSM 6)
├── Worker 3: Figure regions  (Contour detection)
└── Worker 4: Font regions    (PSM 7)
    │
    ▼
Agent 1 — Segmentation QA
(Validates chunks, filters noise)
    │
    ▼
Agent 2 — OCR Correction
(Fixes low-confidence text using LLM)
    │
    ▼
ChromaDB Vector Store
(Text + metadata: book, page, paragraph, line)
    │
    ▼
Agent 3 — Query & Response
(Retrieves matches, formats answer with LLM)
    │
    ▼
Gradio Web Interface
```

---

## Tech Stack

| Component | Technology |
|---|---|
| PDF Processing | PyMuPDF (fitz) |
| Computer Vision | OpenCV |
| OCR Engine | Tesseract (pytesseract) |
| Parallel Execution | Python ThreadPoolExecutor |
| AI Agents | LangChain (LCEL) |
| Local LLM | Ollama — Gemma2:2b |
| Vector Database | ChromaDB |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Web Interface | Gradio |
| Runtime | Google Colab |

---

## Project Structure

```
cvip_project/
├── app.py                  ← Gradio web interface
├── pipeline/
│   ├── __init__.py
│   ├── ocr.py              ← PDF→Image→Preprocess→Parallel OCR
│   ├── rag.py              ← ChromaDB store and search
│   └── agents.py           ← Agent 1, 2, 3 definitions
├── uploads/                ← Uploaded PDFs
└── chroma_db/              ← Vector database (auto-generated)
```

---

## Setup & Installation

### Option A — Google Colab (Recommended)

Open the provided `.ipynb` notebook in Google Colab and run cells in order. Each cell is self-contained with instructions.

**Cell order:**
1. System dependencies (Tesseract, poppler)
2. Python libraries
3. Ollama + Gemma2:2b model
4. Folder structure
5. Write `ocr.py`
6. Write `rag.py`
7. Write `agents.py`
8. Write `app.py`
9. Verify files
10. Launch app (Gradio public URL)

### Option B — Windows Local

**Prerequisites:**
- Python 3.10+
- [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
- [Ollama](https://ollama.com/download)

```bash
# 1. Clone / create project folder
mkdir cvip_project && cd cvip_project

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install libraries
pip install streamlit langchain langchain-community langchain-core langchain-ollama chromadb pytesseract opencv-python pymupdf pillow sentence-transformers gradio

# 4. Pull LLM model
ollama pull gemma2:2b

# 5. Run app
python app.py
```

Update Tesseract path in `ocr.py` line 8 if different:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

---

## Usage

1. Open the Gradio URL in browser
2. Go to **Upload & Index** tab
3. Upload a PDF file and click **Process & Index**
4. Wait for indexing to complete
5. Go to **Search** tab
6. Type any text (e.g. `CVIP is TOP`)
7. Enable **Agent 3** for LLM-formatted answer
8. Results show: Book name, Page No., Paragraph No., Line No.

---

## Three AI Agents

**Agent 1 — Segmentation QA**
Runs after OCR. Uses rules + LLM to validate each text chunk. High confidence chunks (≥70%) pass automatically. Low confidence chunks go to LLM for verdict. Garbage chunks are filtered out before indexing.

**Agent 2 — OCR Correction**
Runs on low-confidence chunks only (confidence < 75%). Uses Gemma2:2b to contextually fix common OCR errors: `l→1`, `O→0`, `rn→m`, broken hyphenated words, extra spaces.

**Agent 3 — Query & Response**
Runs on every search. Takes user query → retrieves top-k chunks from ChromaDB → builds context with metadata → asks LLM to format a precise answer with exact book/page/paragraph/line locations.

---

## Key Concepts

**Binarization:** Converting a grayscale image to pure black and white. Makes text clearly distinguishable from background. Adaptive thresholding is used to handle uneven lighting in scanned documents.

**Parallel OCR:** All 4 segmentation workers run simultaneously on the same page image using Python's `ThreadPoolExecutor`. This reduces per-page processing time from ~20s (sequential) to ~5s (parallel).

**RAG (Retrieval-Augmented Generation):** Instead of asking the LLM to recall information from memory (which causes hallucinations), we first retrieve relevant document chunks from ChromaDB, then provide them as context to the LLM for accurate, grounded answers.

**Vector Embeddings:** Text is converted to numerical vectors (384 dimensions) using SentenceTransformers. Semantically similar texts have numerically close vectors, enabling meaning-based search rather than exact keyword matching.

---

## Known Limitations

- Agent pipeline is slow on CPU — agents skip for high-confidence chunks to compensate
- Gemma2:2b requires minimum 1.5GB available RAM
- Large PDFs (>50 pages) take significant time to process on Colab free tier
- Figure regions are detected by location only, not OCR'd

---

## Subject

Computer Vision & Image Processing (CVIP) — Semester Project
