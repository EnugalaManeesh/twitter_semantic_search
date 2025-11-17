#!/usr/bin/env python3
"""
semantic_search.py

Usage:
  python semantic_search.py --query "How do I fetch tweets with expansions?"
  python semantic_search.py --rebuild
"""

import argparse
import os
import json
import ujson
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm
import hashlib

# -----------------------
# Utilities
# -----------------------
def slug(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]

def read_postman_collection(path):
    with open(path, 'r', encoding='utf-8') as f:
        return ujson.load(f)

def extract_items(collection):
    """
    Flatten Postman collection items recursively.
    Each 'item' may be a folder (with 'item' children) or a request.
    Returns list of dicts with at least: name, description, request, responses, path
    """
    results = []
    def walk(items, path):
        for it in items:
            name = it.get('name','')
            cur_path = path + [name] if name else path
            if 'item' in it:  # folder
                walk(it['item'], cur_path)
            else:
                # request item
                entry = {
                    'name': name,
                    'path': cur_path,
                    'request': it.get('request', {}),
                    'responses': it.get('response', []),
                    'description': it.get('description') or it.get('request',{}).get('description') or ""
                }
                results.append(entry)
    root_items = collection.get('item', [])
    walk(root_items, [])
    return results

def build_text_blob(item):
    parts = []
    parts.append(f"Name: {item.get('name','')}")
    parts.append("Path: " + " > ".join(item.get('path',[])))
    req = item.get('request',{})
    # URL
    url = ""
    try:
        url = req.get('url',{}).get('raw') or req.get('url',{})
    except Exception:
        url = ""
    if url:
        parts.append("URL: " + str(url))
    # Method
    method = req.get('method')
    if method:
        parts.append("Method: " + method)
    # Description
    desc = item.get('description','') or req.get('description','')
    if desc:
        parts.append("Description: " + desc)
    # Body
    body = req.get('body',{})
    if body:
        # Try to extract raw or formdata/text
        raw = ""
        if 'raw' in body:
            raw = body.get('raw','')
        elif isinstance(body.get('mode'), str) and body.get(body.get('mode',''), None):
            raw = str(body.get(body.get('mode',''), ''))
        if raw:
            parts.append("Request Body: " + raw)
    # Responses
    responses = item.get('responses', [])
    for r in responses:
        status = r.get('name','') or r.get('status','')
        rbody = r.get('body','')
        if status or rbody:
            parts.append(f"Response ({status}): {rbody}")
    return "\n\n".join([p for p in parts if p])

def chunk_text(text, max_chars=900, overlap=200):
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= L:
            break
    return chunks

# -----------------------
# Index building
# -----------------------
def build_index(collection_path, index_dir, model_name='all-MiniLM-L6-v2', max_chars=900, overlap=200):
    os.makedirs(index_dir, exist_ok=True)
    coll = read_postman_collection(collection_path)
    items = extract_items(coll)
    print(f"Extracted {len(items)} request items from collection.")
    model = SentenceTransformer(model_name)
    embeddings = []
    metadata = []
    for it in tqdm(items, desc="Processing items"):
        blob = build_text_blob(it)
        chunks = chunk_text(blob, max_chars=max_chars, overlap=overlap)
        for i, c in enumerate(chunks):
            emb = model.encode(c, show_progress_bar=False)
            embeddings.append(emb)
            meta = {
                'chunk_id': f"{slug(it.get('name',''))}-{i}",
                'item_name': it.get('name',''),
                'path': it.get('path',[]),
                'text': c,
                'source_file': os.path.basename(collection_path)
            }
            metadata.append(meta)
    if len(embeddings) == 0:
        raise RuntimeError("No chunks to index.")
    X = np.vstack(embeddings).astype('float32')
    dim = X.shape[1]
    # Normalize for cosine similarity (optional). We'll use inner product on normalized vectors.
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(dim)  # Inner product on normalized vectors => cosine similarity
    index.add(X)
    # Save index and metadata & model name
    faiss.write_index(index, os.path.join(index_dir, "twitter_postman.index"))
    np.save(os.path.join(index_dir, "meta.npy"), np.array(metadata, dtype=object))
    with open(os.path.join(index_dir, "index_info.json"), 'w', encoding='utf-8') as f:
        json.dump({'model_name': model_name, 'dim': dim, 'n_chunks': len(metadata)}, f, indent=2)
    print(f"Index built with {len(metadata)} chunks. Saved to {index_dir}")

# -----------------------
# Query
# -----------------------
def load_index(index_dir):
    idx_path = os.path.join(index_dir, "twitter_postman.index")
    meta_path = os.path.join(index_dir, "meta.npy")
    info_path = os.path.join(index_dir, "index_info.json")
    if not os.path.exists(idx_path):
        raise FileNotFoundError("Index file not found. Run with --rebuild to create it.")
    index = faiss.read_index(idx_path)
    metadata = list(np.load(meta_path, allow_pickle=True))
    info = json.load(open(info_path,'r',encoding='utf-8'))
    model_name = info.get('model_name','all-MiniLM-L6-v2')
    return index, metadata, model_name

def query_index(query, index_dir, top_k=5):
    index, metadata, model_name = load_index(index_dir)
    model = SentenceTransformer(model_name)
    qv = model.encode(query)
    qv = np.asarray([qv.astype('float32')])
    faiss.normalize_L2(qv)
    D, I = index.search(qv, top_k)
    # D: similarity scores (cosine) since we normalized and used inner product
    results = []
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        if idx < 0:
            continue
        meta = metadata[idx]
        results.append({
            'rank': rank,
            'score': float(score),
            'chunk_id': meta.get('chunk_id'),
            'item_name': meta.get('item_name'),
            'path': meta.get('path'),
            'text': meta.get('text'),
            'source_file': meta.get('source_file')
        })
    out = {
        'query': query,
        'model': model_name,
        'top_k': top_k,
        'results': results
    }
    return out

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Semantic search over Twitter Postman collection")
    parser.add_argument('--collection', default="Twitter API v2.postman_collection.json", help="Path to Postman collection JSON")
    parser.add_argument('--index_dir', default="index_data", help="Directory to store index & metadata")
    parser.add_argument('--rebuild', action='store_true', help="Rebuild the index from the collection")
    parser.add_argument('--query', type=str, help="Query text to search")
    parser.add_argument('--top_k', type=int, default=5, help="Top-k results to return")
    args = parser.parse_args()

    if args.rebuild:
        print("Building index...")
        build_index(args.collection, args.index_dir)
        print("Index build complete.")

    if args.query:
        out = query_index(args.query, args.index_dir, top_k=args.top_k)
        print(ujson.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
