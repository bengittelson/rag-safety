# RAG Makes Command-R Less Safe

*Ben Gittelson — CSCI E-222 Foundations of Large Language Models, Harvard Extension School, Spring 2026*

## Introduction and Problem Statement

Retrieval-augmented generation (RAG) is widely used in LLM applications in safety-critical domains such as finance, medicine, and law. An et al. (2025) found that RAG makes 8 of 11 popular LLMs *less* safe — even when retrieved documents are themselves safe — because models repurpose benign information to construct harmful responses. However, An et al. did not test RAG-optimized models. This project asks: do their findings extend to Cohere's **Command-R**, a model explicitly fine-tuned for grounded RAG generation?

- **RQ1:** Is Command-R safer with or without RAG when responding to adversarial prompts?
- **RQ2:** What explains any difference in safety rates?

## Data, Models, and Methods

| Component | Details |
|---|---|
| Generator | `command-r-08-2024` (32B, bfloat16), grounded generation prompt template, temp=0.3, max tokens=256 |
| Adversarial prompts | Red-Teaming Resistance Dataset (Haize Labs) — 1,639 of 5,192 prompts sampled across 15 harm categories |
| RAG index | June 2024 Wikipedia dump; 1M paragraphs embedded with BGE-M3 (1024-dim), indexed via FAISS cosine similarity |
| Safety judge | `Llama-Guard-3-8B` — outputs safe/unsafe label and harm category per response |

## Results

Adding RAG increased Command-R's unsafe response rate from **18.9%** (309/1,639) to **21.0%** (345/1,639), confirming An et al.'s findings extend to RAG-optimized models. The largest increases were in the hate (+62%), suicide & self-harm (+38%), and nonviolent crimes (+16%) categories. The model often repurposes benign Wikipedia content to construct harmful answers. Unsafe response rate also rose with the number of cited documents, suggesting grounding itself amplifies risk.

## Strengths, Limitations, and Future Work

**Strengths:** Extends An et al.'s safety analysis to a RAG-specific model; demonstrates meaningful unsafe response rates both with and without RAG.

**Limitations:** Random Wikipedia sampling reduces retrieval relevance; Llama Guard judgments are not human-validated.

**Future work:** Test hybrid/graph/agentic RAG architectures; evaluate on domain-specific query sets and indices (financial, legal, medical).

## Running the Code

The notebook (`rag_safety.ipynb`) is designed to run on **Google Colab with an A100 GPU** (~80 GB VRAM). It requires access to two gated Hugging Face models (`CohereLabs/c4ai-command-r-08-2024` and `meta-llama/Llama-Guard-3-8B`).

```bash
# 1. Open in Colab (badge in notebook) or clone and open locally
# 2. Authenticate with Hugging Face
huggingface-cli login   # or set HF_TOKEN in the config cell

# 3. Install dependencies (first notebook cell)
#    Restart the runtime after installation if prompted

# 4. Set paths in the config cell (FAISS_INDEX_PATH, output CSVs, etc.)
#    Default paths assume Google Drive mount at /content/drive/MyDrive/

# 5. Run cells sequentially:
#    - Build/load the FAISS index
#    - Run RAG and no-RAG inference
#    - Run Llama Guard safety judgments
#    - Run analysis and visualization cells
```

Key hyperparameters are consolidated in the environment config cell: `USE_4BIT`, `SUBSET_SIZE`, `TOP_K`, `TEMPERATURE`, and file paths.
