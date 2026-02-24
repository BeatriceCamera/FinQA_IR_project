# Financial-Opinion-Retrieval
Information Retrieval project investigating the impact of **sentiment-controlled LLM-based query expansion** within a **hybrid financial retrieval system (BM25 + Bi-Encoder)**.


## Investigating Polarity Manipulation in Financial IR
- **Language:** Python  
- **Libraries:**  
  - **IR Framework:** PyTerrier  
  - **Neural Retrieval & Embeddings:** Sentence-Transformers (Sentence-BERT), PyTorch  
  - **LLM & Transformers:** Hugging Face Transformers  
  - **Sentiment Analysis:** FinBERT  
  - **NLP & Processing:** spaCy  
  - **Data Handling & Utilities:** numpy, pandas, json, requests  


## Overview  
This project studies how **Large Language Models (LLMs)** influence ranking behavior in an opinion-oriented financial search setting.

Using the **FinQA retrieval dataset**, we analyze whether injecting controlled sentiment (neutral, positive/bullish, negative/bearish) into short financial queries improves or degrades retrieval effectiveness.

The system follows a **hybrid retrieval architecture**:
1. **BM25 lexical retrieval** to generate candidate documents.  
2. **Bi-Encoder neural reranking** based on embedding similarity.  

The core research question is to what extent does strategically inducing polarity in LLM-generated Query Expansion alter ranking behaviour and introduce measurable bias in retrieval metrics within a hybrid financial IR system.


## Methodology  

The study follows a two-phase structure:

### Phase I — Baselines
- **TF-IDF**
- **BM25**
- **BM25 + RM3 pseudo-relevance feedback**
- **Hybrid BM25 + Bi-Encoder reranking**

These systems establish reference performance under identical indexing and evaluation settings.

### Phase II — Sentiment-Controlled Query Expansion
For each original query, three expanded variants are generated through prompt engineering:

- **Neutral Expansion** — domain enrichment without sentiment  
- **Positive Expansion** — bullish financial language  
- **Negative Expansion** — risk-oriented vocabulary  

Expansions are generated offline and validated using **FinBERT** to ensure controlled polarity differences.

All query variants are evaluated using the same hybrid retrieval pipeline to isolate the effect of sentiment manipulation.


## Evaluation  

Retrieval effectiveness is measured using standard IR metrics:

- P@1, P@5, P@10  
- R@5, R@10  
- nDCG@5, nDCG@10  
- MAP  

All experiments are conducted under controlled and reproducible settings using PyTerrier.


## Results and Insights  

- The **original hybrid baseline** achieves the highest performance across all metrics.  
- All LLM-based query expansions lead to performance degradation.  
- Performance drops are primarily explained by **query drift**, especially in a sparse-relevance setting.  
- **Negative (bearish) expansions are more resilient than positive (bullish) ones**, suggesting that risk-oriented vocabulary aligns better with financial opinion content.  

The results highlight the fragility of hybrid retrieval systems to linguistic steering and emphasize the need for precision-preserving integration of LLMs in financial IR.


## View the Full Notebook  
`Financial_Question_Answering.ipynb`  

For a static preview:  
[Open in nbviewer](https://nbviewer.org/github/BeatriceCamera/REPO_NAME/blob/main/Financial_Question_Answering.ipynb)


## Authors  
Beatrice Camera  
Daria Miele  
Zofia Pempera  

B.Sc. Artificial Intelligence @ University of Pavia, University of Milan, University of Milano-Bicocca
