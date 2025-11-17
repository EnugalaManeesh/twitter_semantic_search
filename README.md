# twitter_semantic_search

This project allows you to perform **semantic search on Twitter data** using a Postman collection for Twitter API v2. It builds an index from the API endpoints and enables quick retrieval of relevant tweets or posts using semantic similarity.

---

## Features

- Build an **index** from Twitter API v2 Postman collection.
- Perform **semantic search** over tweets or API responses.
- Uses **Sentence Transformers** for embedding text.
- Compatible with **CPU-only** and **PyTorch** environments.

---

## Requirements

- Python 3.10+
- Packages:
  - `sentence-transformers`
  - `torch`, `torchvision`, `torchaudio`
  - `scikit-learn`
  - `scipy`
  - `ujson`
  - `tqdm`
  - `snscrape`

You can install dependencies with:

```bash
pip install -r requirements.txt
Setup

Clone the repository:

git clone https://github.com/EnugalaManeesh/twitter_semantic_search.git
cd twitter_semantic_search


Copy your Postman collection file (e.g., Twitter API v2.postman_collection.json) into the project folder.

Build the semantic index:

python3 semantic_search.py --rebuild --collection "Twitter API v2.postman_collection.json" --index_dir ./index_data


Run semantic search:

python3 semantic_search.py --query "python programming" --index_dir ./index_data
