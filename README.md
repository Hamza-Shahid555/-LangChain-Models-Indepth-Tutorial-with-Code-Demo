# 🦜🔗 LangChain Models — Complete Tutorial



---

## 📚 Table of Contents

- [What is the Model Component?](#what-is-the-model-component)
- [1. Language Models](#1-language-models)
  - [Legacy LLMs vs Chat Models](#legacy-llms-vs-chat-models)
  - [OpenAI](#openai)
  - [Anthropic Claude](#anthropic-claude)
  - [Google Gemini](#google-gemini)
  - [Key Parameters](#key-parameters)
- [2. Open Source Models](#2-open-source-models)
  - [Hugging Face Inference API](#hugging-face-inference-api)
  - [Local Execution](#local-execution)
- [3. Embedding Models](#3-embedding-models)
  - [OpenAI Embeddings](#openai-embeddings)
  - [Sentence Transformers](#sentence-transformers)
  - [Semantic Search Demo](#semantic-search-demo)
- [Setup & Installation](#setup--installation)
- [Folder Structure](#folder-structure)

---

## What is the Model Component?

In LangChain, the **Model** component acts as a **unified interface** to interact with a wide variety of AI models — whether they are closed-source APIs (OpenAI, Anthropic, Google) or open-source models (Hugging Face, local LLMs).

```
Your Code
   │
   ▼
LangChain Model Interface
   │
   ├──► Language Models (Text Generation)
   │       ├── Legacy LLMs
   │       └── Chat Models  ✅ (recommended)
   │
   └──► Embedding Models (Text → Vectors)
           ├── OpenAI Embeddings
           └── Sentence Transformers
```

The key benefit: **swap any model provider with minimal code changes.**

---

## 1. Language Models

### Legacy LLMs vs Chat Models

| Feature | Legacy LLM | Chat Model |
|---|---|---|
| Input | Plain text string | List of messages |
| Output | Plain text string | `AIMessage` object |
| LangChain support | `LLM` base class | `ChatModel` base class |
| Recommended? | ❌ Being phased out | ✅ Yes |

LangChain **encourages Chat Models** as the standard going forward. Legacy LLMs still work but Chat Models offer structured message handling (System, Human, AI roles).

```python
# Legacy LLM style (not recommended)
llm.invoke("Tell me a joke")

# Chat Model style (recommended)
from langchain_core.messages import HumanMessage, SystemMessage

chat.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me a joke")
])
```

---

### OpenAI

```python
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=500
)

response = model.invoke("What is the capital of France?")
print(response.content)
```

📄 Full code: [`src/language_models/openai_demo.py`](src/language_models/openai_demo.py)

---

### Anthropic Claude

```python
from langchain_anthropic import ChatAnthropic
import os

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.5,
    max_tokens=1024
)

response = model.invoke("Explain quantum computing simply.")
print(response.content)
```

📄 Full code: [`src/language_models/anthropic_demo.py`](src/language_models/anthropic_demo.py)

---

### Google Gemini

```python
from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ["GOOGLE_API_KEY"] = "your-api-key"

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.3
)

response = model.invoke("What are the laws of thermodynamics?")
print(response.content)
```

📄 Full code: [`src/language_models/gemini_demo.py`](src/language_models/gemini_demo.py)

---

### Key Parameters

#### 🌡️ Temperature
Controls the **randomness/creativity** of the model output.

| Temperature | Behavior | Use Case |
|---|---|---|
| `0.0` | Deterministic, factual | Q&A, coding, math |
| `0.3–0.7` | Balanced | General purpose |
| `1.0+` | Creative, unpredictable | Brainstorming, stories |

```python
# Very precise output
model = ChatOpenAI(model="gpt-4o", temperature=0.0)

# Creative output
model = ChatOpenAI(model="gpt-4o", temperature=1.2)
```

#### 🔤 Tokenization
- Models don't read raw text — they read **tokens** (chunks of characters).
- ~1 token ≈ 4 characters in English.
- `max_tokens` controls the **output length limit**.
- API pricing is based on tokens used.

```
"Hello, world!" → ["Hello", ",", " world", "!"]  → 4 tokens
```

---

## 2. Open Source Models

### Hugging Face Inference API

No GPU required — runs inference **on Hugging Face's servers** via API.

```python
from langchain_huggingface import HuggingFaceEndpoint
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your-hf-token"

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    temperature=0.7,
    max_new_tokens=256
)

response = llm.invoke("What is machine learning?")
print(response)
```

📄 Full code: [`src/open_source_models/hf_inference_api.py`](src/open_source_models/hf_inference_api.py)

---

### Local Execution

Download the model and run **100% offline** — no API needed.

```python
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Download and run locally
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=pipe)
response = llm.invoke("Explain neural networks in simple terms.")
print(response)
```

> ⚠️ First run downloads the model weights (~600MB for TinyLlama). Subsequent runs are instant.

📄 Full code: [`src/open_source_models/local_model.py`](src/open_source_models/local_model.py)

---

## 3. Embedding Models

Embedding models **convert text into numerical vectors**. These vectors capture the semantic meaning of the text and are used for:
- Semantic search
- Document retrieval
- Clustering / classification
- RAG (Retrieval-Augmented Generation)

```
"The cat sat on the mat"  →  [0.23, -0.14, 0.89, 0.01, ...]
"A kitten rested on a rug" →  [0.21, -0.12, 0.87, 0.03, ...]
                              # ↑ Similar vectors = similar meaning
```

---

### OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector = embeddings.embed_query("Hello world")
print(f"Dimensions: {len(vector)}")   # 1536 dimensions
print(f"Sample values: {vector[:5]}")
```

---

### Sentence Transformers

Free, open-source, runs locally.

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embeddings.embed_query("Hello world")
print(f"Dimensions: {len(vector)}")   # 384 dimensions
```

📄 Full code: [`src/embeddings/sentence_transformers_demo.py`](src/embeddings/sentence_transformers_demo.py)

---

### Semantic Search Demo

Find the **most relevant document** for a user query using cosine similarity.

```python
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Python is a high-level programming language known for its simplicity.",
    "The Eiffel Tower is located in Paris, France.",
    "Machine learning is a subset of artificial intelligence.",
    "Football is the most popular sport in the world.",
]

query = "Tell me about AI and data science"

# Embed all documents and the query
doc_vectors = embeddings.embed_documents(documents)
query_vector = embeddings.embed_query(query)

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine_similarity(query_vector, dv) for dv in doc_vectors]
best_match_idx = np.argmax(scores)

print(f"Query: {query}")
print(f"Best match: {documents[best_match_idx]}")
print(f"Similarity score: {scores[best_match_idx]:.4f}")
```

📄 Full code: [`src/embeddings/semantic_search.py`](src/embeddings/semantic_search.py)

---

## Setup & Installation

```bash
# Clone this repository
git clone https://github.com/your-username/langchain-models-tutorial.git
cd langchain-models-tutorial

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
HUGGINGFACEHUB_API_TOKEN=hf_...
```

---

## Folder Structure

```
langchain-models-tutorial/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable template
│
├── src/
│   ├── language_models/
│   │   ├── openai_demo.py             # OpenAI ChatGPT example
│   │   ├── anthropic_demo.py          # Anthropic Claude example
│   │   └── gemini_demo.py             # Google Gemini example
│   │
│   ├── open_source_models/
│   │   ├── hf_inference_api.py        # Hugging Face API (TinyLlama)
│   │   └── local_model.py             # Offline local model execution
│   │
│   └── embeddings/
│       ├── openai_embeddings.py       # OpenAI embedding vectors
│       ├── sentence_transformers_demo.py  # Free local embeddings
│       └── semantic_search.py         # Full semantic search example
│
├── notebooks/
│   └── langchain_models_walkthrough.ipynb  # Interactive Jupyter notebook
│
└── docs/
    ├── language_models.md             # Deep-dive notes on language models
    ├── open_source_models.md          # Open-source model guide
    └── embeddings.md                  # Embeddings & vector search guide
```

---

## 📌 Key Takeaways

1. **LangChain's Model layer** abstracts all providers under a common interface.
2. **Use Chat Models** — not legacy LLMs — they're the current standard.
3. **Temperature** is your creativity dial: 0 = precise, 1+ = creative.
4. **Embeddings** convert text to numbers so computers can understand meaning.
5. **Semantic search** uses vector similarity — not keyword matching — to find relevant content.
6. **Open-source models** are free: use HF API for quick tests, run locally for privacy/offline use.

---

*Tutorial source: CampusX — LangChain Models Component*
