import os
import json
import hashlib
import shutil
from glob import glob
from tqdm import tqdm
from pymilvus import MilvusClient
from transformers import AutoTokenizer
import ollama

# === Configuration ===
DEFAULT_MODEL = "mxbai-embed-large"
COLLECTION_NAME = "Trideum_Collection"
CHUNK_SIZE = 400
HASH_MANIFEST = "file_hash_manifest.json"
CORPUS_DIR = "corpus"

# === Utility Functions ===
def clean_text(text):
    return text.replace("\n", " ").strip().lower()

def tokenize_file(file_path, chunk_size=CHUNK_SIZE):
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        file_text = file.read()
    tokens = tokenizer.encode(file_text, add_special_tokens=False)
    for i in range(0, len(tokens), chunk_size):
        yield {
            "text": clean_text(tokenizer.decode(tokens[i:i + chunk_size])),
            "filename": file_path
        }

def file_hash(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_hash_manifest():
    if os.path.exists(HASH_MANIFEST):
        with open(HASH_MANIFEST, "r") as f:
            return json.load(f)
    return {}

def update_hash_manifest(manifest):
    with open(HASH_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

def get_new_files(corpus_dir, existing_hashes):
    new_files = []
    new_hashes = {}
    for file_path in glob(os.path.join(corpus_dir, "**/*.txt"), recursive=True):
        h = file_hash(file_path)
        if file_path not in existing_hashes or existing_hashes[file_path] != h:
            new_files.append(file_path)
        new_hashes[file_path] = h
    return new_files, new_hashes

# === Milvus + Embeddings ===
def emb_text(text, model=DEFAULT_MODEL):
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def init_milvus_client():
    host = os.getenv("MILVUS_HOST", "milvus")
    port = os.getenv("MILVUS_PORT", "19530")
    uri = f"http://{host}:{port}"
    return MilvusClient(uri=uri)

def setup_collection(client: MilvusClient, collection_name: str, dimension: int):
    if not client.has_collection(collection_name):
        index_params = client.prepare_index_params()
        index_params.add_index("id", index_type="STL_SORT")
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            index_params=index_params,
            metric_type="IP",
            consistency_level="Bounded",
            enable_dynamic_field=True
        )

def clear_corpus_and_milvus():
    shutil.rmtree(CORPUS_DIR, ignore_errors=True)
    os.makedirs(CORPUS_DIR, exist_ok=True)
    if os.path.exists(HASH_MANIFEST):
        os.remove(HASH_MANIFEST)
    client = init_milvus_client()
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

# === Querying ===
def query_collection(client: MilvusClient, collection_name: str, query: str, limit=10):
    if not client.has_collection(collection_name):
        print(f"[WARN] Collection '{collection_name}' not found.")
        return []
    vector = emb_text(query)
    results = client.search(
        collection_name=collection_name,
        data=[vector],
        limit=limit,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text", "filename"],
    )
    return [{"filename": hit["entity"]["filename"], "text": hit["entity"]["text"]} for hit in results[0]]

def build_context(results):
    return "\n\n".join(
        f"[Source: {item['filename']}]\n{item['text']}"
        for item in results)

# === Streaming Corpus Processing ===
def process_and_index_new_files(corpus_dir: str, collection_name="Trideum_Collection", batch_size=100):
    existing_hashes = load_hash_manifest()
    new_files, updated_hashes = get_new_files(corpus_dir, existing_hashes)
    if not new_files:
        print("No new files to process.")
        return

    client = init_milvus_client()
    setup_collection(client, collection_name, dimension=1024)

    batch = []
    index = 0

    for file_path in new_files:
        for chunk in tokenize_file(file_path):
            # 1️⃣ embed+append in its own try/except
            try:
                vector = emb_text(chunk["text"].strip())
            except Exception as e:
                print(f"Skipping chunk (embed error) from {file_path}: {e}")
                continue

            batch.append({
                "id": index,
                "vector": vector,
                "text": chunk["text"],
                "filename": chunk["filename"]
            })
            index += 1

            # 2️⃣ when batch is full, insert in its own try/finally
            if len(batch) >= batch_size:
                try:
                    client.insert(collection_name=collection_name, data=batch)
                except Exception as e:
                    print(f"Failed to insert batch up to index {index}: {e}")
                finally:
                    batch.clear()

    # 3️⃣ leftover batch
    if batch:
        try:
            client.insert(collection_name=collection_name, data=batch)
        except Exception as e:
            print(f"Failed to insert final batch: {e}")
        finally:
            batch.clear()

    update_hash_manifest(updated_hashes)
