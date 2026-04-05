# SupportDesk-RAG: A Support Ticket Retrieval & Troubleshooting Assistant

## Hands-On RAG Workshop with OpenAI

### Workshop Overview
This comprehensive workshop teaches you to build a production-ready Retrieval-Augmented Generation (RAG) system using OpenAI embeddings and language models. By the end, you'll have a working assistant that answers incident queries using retrieved ticket context, with strong safeguards against hallucinations.

### Learning Objectives
- ✅ Generate and work with OpenAI embeddings
- ✅ Master chunking strategies for optimal retrieval  
- ✅ Compare 5 different indexing strategies (LlamaIndex)
- ✅ Implement a complete RAG pipeline with LangChain
- ✅ Evaluate with two-layer metrics (retrieval + generation)
- ✅ Deploy anti-hallucination safeguards
- ✅ Build agentic RAG systems with multi-step reasoning

---

## 🚀 Quick Start

### 1. Install Python 3.12 (one-time)

> 💡 **Alternatively**, you can use [`uv`](https://docs.astral.sh/uv/) to skip the manual Python install — it downloads the correct version automatically. See [Step 3, Option A](#option-a-using-uv-auto-downloads-python-312) for setup details.

> ⚠️ **Python 3.13 and 3.14 are not supported** — `chromadb` depends on Pydantic V1 internals that were removed in Python 3.13+.

**Windows (PowerShell with winget):**
```powershell
winget install Python.Python.3.12
```
Restart your terminal after installation.

**Windows (Manual installer):**
1. Download Python 3.12 from https://www.python.org/downloads/release/python-3129/
2. Run the installer
3. ✅ **Check "Add python.exe to PATH"** at the bottom of the first screen
4. Click **"Install Now"**

**Verify installation (Windows):**
```powershell
py -3.12 --version
```

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python@3.12

# Verify installation
python3.12 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv

# Verify installation
python3.12 --version
```

### 2. Clone or open this repo
```bash
# If you already have the repo, skip this step
git clone https://github.com/singhsidhukuldeep/SupportDesk-RAG-Workshop.git
cd SupportDesk-RAG-Workshop
```

### 3. Create a virtual environment (recommended)

#### Option A: Using `uv` (auto-downloads Python 3.12)

[`uv`](https://docs.astral.sh/uv/) can fetch the correct Python version automatically — no need to install Python 3.12 yourself (skip Step 1).

Install `uv` if you don't have it:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Create the virtual environment:
```bash
uv venv --python 3.12 .venv
```

#### Option B: Using `conda` (auto-downloads Python 3.12)

If you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed — no need to install Python 3.12 yourself (skip Step 1).

```bash
conda create -n supportdesk-rag python=3.12 -y
```

#### Option C: Using `venv` (manual)

Requires Python 3.12 to be installed first (Step 1).

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
```

**macOS/Linux:**
```bash
python3.12 -m venv .venv
```

### 4. Activate the virtual environment

**If you used `uv` or `venv` (Options A/C):**

**Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```bat
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**If you used `conda` (Option B):**
```bash
conda activate supportdesk-rag
```

### 5. Install dependencies

**If you used `uv` (Option A):**
```bash
uv pip install -r requirements.txt
```

**If you used `conda` or `venv` (Options B/C):**
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Configure OpenAI API

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS/Linux:**
```bash
cp .env.example .env
```

Then edit `.env` and set:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### 7. Run a smoke test
```bash
cd modules/1_embeddings
python demo.py
```

If Module 1 runs, your environment is ready.

---

## Workshop Modules

### Module 1: Embeddings (`modules/1_embeddings/`)
**Learn:**
- Generate embeddings using OpenAI API
- Compute semantic similarity scores
- Visualize similarity relationships with heatmaps

**Run:**
```bash
cd modules/1_embeddings
python demo.py
```

---

### Module 2: Chunking (`modules/2_chunking/`)
**Learn:**
- Fixed-size vs recursive vs semantic chunking
- Structure-aware splitting (Markdown/HTML)
- Build vector stores with Chroma

**Run:**
```bash
cd modules/2_chunking
python demo.py
```

---

### Module 3: Indexing Strategies (`modules/3_indexing/`)
**Learn:**
- Vector Index - Semantic similarity search (most common)
- Summary Index - High-level document summaries
- Tree Index - Hierarchical retrieval patterns
- Keyword Table Index - Traditional keyword matching
- Hybrid Retrieval - Combining multiple strategies

**Technologies:** LlamaIndex for clean indexing abstractions

**Run:**
```bash
cd modules/3_indexing
python demo.py
```

---

### Module 4: RAG Pipeline (`modules/4_rag_pipeline/`)
**Learn:**
- Complete RAG architecture
- LangChain integration
- Prompt engineering for grounded responses
- Anti-hallucination strategies

**Run:**
```bash
cd modules/4_rag_pipeline
python demo.py
```

---

### Module 5: Evaluation (`modules/5_evaluation/`)
**Learn:**
- Two-layer evaluation approach (Retrieval + Generation)
- Retrieval metrics (Precision@K, Recall@K, F1)
- Generation metrics (Groundedness, Completeness)
- LLM-as-judge for generation evaluation
- Creating comprehensive evaluation reports

**Technologies:** FAISS, LLM-as-Judge evaluation

**Run:**
```bash
cd modules/5_evaluation
python demo.py
```

---

### Module 6: Agentic RAG (`modules/6_agentic_rag/`)
**Learn:**
- Creating custom tools for LangChain agents
- Building agents with OpenAI function calling
- Implementing conversation memory
- Multi-step reasoning with tool selection
- Comparing agentic vs direct RAG approaches

**Technologies:** LangChain Agents, OpenAI Function Calling

**Run:**
```bash
cd modules/6_agentic_rag
python demo.py
```

---

### Run All Modules

To run all module demos sequentially from the project root:

**Windows (PowerShell):**
```powershell
$modules = @("1_embeddings", "2_chunking", "3_indexing", "4_rag_pipeline", "5_evaluation", "6_agentic_rag")
foreach ($module in $modules) {
    Write-Host "`n=== Running Module: $module ===" -ForegroundColor Cyan
    Push-Location "modules/$module"
    python demo.py
    Pop-Location
}
```

**macOS/Linux:**
```bash
for module in 1_embeddings 2_chunking 3_indexing 4_rag_pipeline 5_evaluation 6_agentic_rag; do
    echo -e "\n=== Running Module: $module ==="
    cd modules/$module
    python demo.py
    cd ../..
done
```

---

## 📁 Repository Structure

```
SupportDesk-RAG-Workshop/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                # Environment template
├── POST_CLASS_GUIDE.md         # Post-workshop learning guide
├── data/
│   └── synthetic_tickets.json  # Sample support tickets
└── modules/
    ├── 1_embeddings/
    │   ├── demo.py             # Working demo code
    │   ├── notes.md            # Instructor notes
    │   └── exercises.md        # Practice exercises
    ├── 2_chunking/
    │   ├── demo.py
    │   ├── notes.md
    │   └── exercises.md
    ├── 3_indexing/
    │   ├── demo.py
    │   ├── notes.md
    │   └── exercises.md
    ├── 4_rag_pipeline/
    │   ├── demo.py
    │   ├── notes.md
    │   └── exercises.md
    ├── 5_evaluation/
    │   ├── demo.py
    │   ├── notes.md
    │   ├── exercises.md
    │   ├── solutions.py
    │   └── evaluation_queries.json
    └── 6_agentic_rag/
        ├── demo.py
        ├── notes.md
        ├── exercises.md
        ├── solutions.py
        ├── tools.py
        ├── test_setup.py
        └── README.md
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (defaults shown)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### Model Options

**Embeddings:**
- `text-embedding-3-small` (1536 dims, recommended)
- `text-embedding-3-large` (3072 dims, highest quality)

**Chat:**
- `gpt-4o-mini` (recommended for cost/performance)
- `gpt-4o` (most capable)

---

## 💰 Cost Estimate

Running all modules: **< $0.10**
- Embeddings: ~$0.01 (20 tickets + queries)
- Chat completions: ~$0.05 (RAG pipeline demos)

See [OpenAI Pricing](https://openai.com/pricing) for current rates.

---

## 🎯 Prerequisites

- Python 3.12.x — **not 3.13/3.14**
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Basic understanding of Python
- Familiarity with APIs (helpful but not required)

---

## 🛠️ Troubleshooting

### Python 3.13 / 3.14 — `chromadb` crashes on import
`chromadb` uses Pydantic V1 internally, which Python 3.13+ broke. You will see:
```
pydantic.v1.errors.ConfigError: unable to infer type for attribute "chroma_server_nofile"
```
**Fix:** use Python 3.12. If you have it installed alongside 3.14, recreate the venv:
```bash
py -3.12 -m venv .venv
```
Then re-run the install steps.

### Virtual Environment Activation Fails (Windows PowerShell)
If you see an execution policy error:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### `python` Command Not Found (Windows)
Use:
```powershell
py --version
py -3.12 -m venv .venv
```

### OpenAI API Errors
- Verify API key in `.env` file
- Check credits: https://platform.openai.com/usage
- Rate limits: Wait 60s if you get 429 errors

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

If issues persist, confirm your venv is active and reinstall cleanly:
```bash
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
```

### Path Issues
- Always run demos from their module directory
- Ensure `data/synthetic_tickets.json` exists

---

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Chroma Documentation](https://docs.trychroma.com/)

---

## 🤝 Contributing

Found a bug or have suggestions? Open an issue or submit a pull request!

---

## 📄 License

MIT License - Feel free to use for learning and teaching!
