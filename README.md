# Trading RAG — SMC Assistant

A terminal chatbot that answers questions about Smart Money Concepts (SMC) trading
strategies. It only answers from your own documents — no hallucination, no made-up
strategies. Everything it knows comes from the `.md` files you put in `/data/strategies/`.

---

## Table of Contents

1. [How it works](#how-it-works)
2. [Prerequisites](#prerequisites)
3. [What is a virtual environment?](#what-is-a-virtual-environment)
4. [Setup (step by step)](#setup-step-by-step)
5. [Running the bot](#running-the-bot)
6. [Adding your own documents](#adding-your-own-documents)
7. [Project structure](#project-structure)
8. [Configuration reference](#configuration-reference)
9. [Quick command reference](#quick-command-reference)

---

## How it works

The bot uses a pattern called **RAG (Retrieval-Augmented Generation)**. Instead of
relying on the AI model's training data, it first searches your own documents for
relevant content, then passes that content to Gemini to generate a grounded answer.

```
You type a question
        │
        ▼
[Embeddings] — converts your question into a vector (list of numbers)
        │
        ▼
[ChromaDB] — searches the vector store for the 4 most similar document chunks
        │
        ▼
[Prompt builder] — packages the question + retrieved chunks into a prompt
        │
        ▼
[Gemini 2.5 Flash] — reads only the retrieved context and writes an answer
        │
        ▼
Answer printed in your terminal
```

### The three stages in detail

**Stage 1 — Ingestion** *(runs automatically on first launch)*

Every `.md` file in `data/strategies/` is read and split into overlapping 600-token
chunks. Each chunk is then converted into a vector using `gemini-embedding-001` and
saved into a local ChromaDB database (`./vectorstore/`). This only happens once —
subsequent runs load the existing database.

**Stage 2 — Retrieval**

Your question is embedded with the same model. ChromaDB compares that vector against
all stored chunk vectors using cosine similarity and returns the 4 best matches.
These are the chunks the AI will actually read.

**Stage 3 — Generation**

The 4 retrieved chunks are injected into a strict system prompt. Gemini 2.5 Flash
is told to answer only from those chunks — if the context isn't enough to answer,
it says so instead of guessing.

---

## Prerequisites

Before you start, make sure you have:

- **Python 3.10 or higher** — check with `python --version`
- **A Google AI Studio API key** — get one free at [aistudio.google.com](https://aistudio.google.com)
  (sign in → click "Get API key" → create one)

---

## What is a virtual environment?

When you install Python packages with `pip install`, they go into your global Python
installation and are shared across every project on your computer. This causes problems:
different projects need different versions of the same library, and they end up
conflicting with each other.

A **virtual environment (venv)** is an isolated copy of Python that belongs only to
this project. When the venv is active, any `pip install` goes into a folder called
`venv/` inside your project — not into your global Python. Other projects are
completely unaffected.

**You should always use a venv for Python projects.** It is the standard practice.

---

## Setup (step by step)

### Step 1 — Get the project

```bash
git clone <repo-url>
cd test_rag
```

---

### Step 2 — Create the virtual environment

Run this once inside the project folder. It creates a `venv/` directory:

```bash
python -m venv venv
```

You will not see much output. That is normal — it just created the folder.

---

### Step 3 — Activate the virtual environment

You must activate the venv every time you open a new terminal to work on this project.

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

After activation, your terminal prompt will change to show `(venv)` at the start:

```
(venv) PS C:\Users\you\test_rag>
```

That `(venv)` prefix is your confirmation that the venv is active. If you do not see
it, the venv is not active and you should run the activate command again.

> To deactivate the venv when you are done, type `deactivate` and press Enter.
> Your prompt will go back to normal.

---

### Step 4 — Install dependencies

With the venv active (you see `(venv)` in the prompt), install all required packages:

```bash
pip install -r requirements.txt
```

This installs LangChain, ChromaDB, the Google GenAI SDK, and everything else the
project needs — all inside `venv/`, not your global Python. This step only needs
to be done once.

---

### Step 5 — Add your API key

Open the `.env` file in the project root and replace the placeholder with your
real Google AI Studio API key:

```
GOOGLE_API_KEY=your_actual_key_here
```

Save the file. Do not share this file or commit it to git — it is already listed
in `.gitignore`.

---

## Running the bot

Every time you want to use the bot:

**1. Open a terminal in the project folder**

**2. Activate the venv:**

```powershell
# Windows
.\venv\Scripts\Activate.ps1
```
```bash
# Mac / Linux
source venv/bin/activate
```

**3. Run:**

```bash
python main.py
```

**What happens on first run:**

The bot detects there is no vector store yet and ingests your documents automatically
before starting the chat:

```
=== Trading RAG SMC Assistant ===

[INFO] No vector store found. Running ingestion now...
[INGEST] Loaded 3 documents from .../data/strategies
[INGEST] Split into 9 chunks
[INGEST] Building Chroma store at .../vectorstore
[INGEST] Vector store built and persisted.

Type your trade idea or question. Type 'exit' to quit.

You: what is a BOS and how do I enter after one?

Bot: A Break of Structure (BOS) confirms a directional shift...
```

On all subsequent runs, ingestion is skipped and the chat starts immediately.

Type `exit` or `quit` to stop the bot.

---

## Adding your own documents

1. Create a `.md` file with your strategy or notes
2. Drop it into `data/strategies/`
3. Re-run ingestion to rebuild the vector store (with venv active):

```bash
python -m rag.ingest
```

4. Run the bot as normal — it will now answer from your new document too

---

## Project structure

```
test_rag/
│
├── data/
│   └── strategies/          # Your knowledge base — add .md files here
│       ├── smc_bos.md
│       ├── smc_fvg.md
│       └── smc_liquidity_grab.md
│
├── rag/
│   ├── config.py            # Loads .env, defines model names and paths
│   ├── embeddings.py        # Wraps the Gemini embedding API for LangChain
│   ├── ingest.py            # Loads docs → splits → embeds → saves to ChromaDB
│   ├── retriever.py         # Loads the saved ChromaDB and returns a retriever
│   └── chat.py              # Retrieves context → builds prompt → calls Gemini
│
├── venv/                    # Virtual environment (gitignored, do not commit)
├── vectorstore/             # ChromaDB vector database (auto-created, gitignored)
├── main.py                  # Entry point: checks vectorstore, runs chat loop
├── requirements.txt         # All Python dependencies
├── .env                     # Your API key (gitignored, do not commit)
└── .env.example             # Template showing what goes in .env
```

---

## Configuration reference

All settings live in `.env`. The bot works with defaults — the only required field
is your API key.

| Variable          | Default                       | Description                                                    |
|-------------------|-------------------------------|----------------------------------------------------------------|
| `GOOGLE_API_KEY`  | *(none)*                      | **Required.** Your Google AI Studio key.                       |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Model used to embed documents and queries. Use `models/gemini-embedding-2` for the latest. |
| `CHAT_MODEL`      | `gemini-2.5-flash`            | Gemini model used to generate answers.                         |
| `VECTORSTORE_DIR` | `./vectorstore`               | Where ChromaDB saves the vector database on disk.              |

---

## Quick command reference

```powershell
# ── First-time setup ──────────────────────────────────────────
python -m venv venv                   # create the virtual environment
.\venv\Scripts\Activate.ps1           # activate it  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt       # install all packages into the venv

# ── Every time you use the bot ────────────────────────────────
.\venv\Scripts\Activate.ps1           # activate the venv first
python main.py                        # run the bot

# ── After adding new documents ────────────────────────────────
.\venv\Scripts\Activate.ps1
python -m rag.ingest                  # rebuild the vector store
python main.py                        # then run the bot

# ── When you are done ─────────────────────────────────────────
deactivate                            # exit the venv
```
