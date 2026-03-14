import warnings
import json
import os
import re
import hashlib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ANALYST_PATH = "../ANALYST"

ps = PorterStemmer()

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "as", "be", "was",
    "are", "were", "been", "has", "have", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "so", "if", "up", "out", "about", "into", "than",
    "then", "that", "this", "these", "those", "what", "which", "who",
    "how", "when", "where", "why", "all", "any", "both", "each"
}

def tokenize(text):
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return [ps.stem(token) for token in tokens if token not in STOPWORDS]

def make_ngrams(tokens, n=2):
    return ['_'.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def process_document(file_path, doc_id):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Exact duplicate detection
    content_hash = hashlib.md5(data['content'].encode('utf-8')).hexdigest()
    if content_hash in seen_hashes:
        return None, None
    seen_hashes.add(content_hash)

    url = data['url']

    try:
        soup = BeautifulSoup(data['content'], 'html.parser')
    except Exception:
        return url, {}

    # Extract important and normal text
    title_text = soup.title.get_text() if soup.title else ""
    h_text = " ".join(tag.get_text() for tag in soup.find_all(['h1','h2','h3']))
    bold_text = " ".join(tag.get_text() for tag in soup.find_all(['b','strong']))
    all_text = soup.get_text()

    normal_tokens = tokenize(all_text)
    important_tokens = set(tokenize(title_text + " " + h_text + " " + bold_text))

    # Build word positions
    positions = defaultdict(list)
    for pos, token in enumerate(normal_tokens):
        positions[token].append(pos)

    # Generate bigrams and trigrams
    bigrams = make_ngrams(normal_tokens, 2)
    trigrams = make_ngrams(normal_tokens, 3)
    all_tokens = normal_tokens + bigrams + trigrams

    tf = Counter(all_tokens)

    # Build postings with positions
    postings = {}
    for token, freq in tf.items():
        postings[token] = {
            'doc_id': doc_id,
            'tf': freq,
            'important': token in important_tokens,
            'positions': positions.get(token, [])
        }

    return url, postings

seen_hashes = set()
duplicates_skipped = 0

# Main indexing loop
inverted_index = defaultdict(list)
doc_id_map = {}
doc_id = 0

folders = [f for f in os.listdir(ANALYST_PATH) if not f.startswith('.')]

for folder in folders:
    folder_path = os.path.join(ANALYST_PATH, folder)
    if not os.path.isdir(folder_path):
        continue
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        url, postings = process_document(file_path, doc_id)
        if url is None:
            duplicates_skipped += 1
            continue
        doc_id_map[doc_id] = url
        for token, posting in postings.items():
            inverted_index[token].append(posting)
        doc_id += 1
        if doc_id % 100 == 0:
            print(f"Processed {doc_id} documents...")

print(f"Done! Indexed {doc_id} documents")
print(f"Duplicates skipped: {duplicates_skipped}")
print(f"Unique tokens: {len(inverted_index)}")

# Save index to disk
os.makedirs("../index", exist_ok=True)

with open("../index/index.json", 'w') as f:
    json.dump(inverted_index, f)

with open("../index/doc_id_map.json", 'w') as f:
    json.dump(doc_id_map, f)

index_size = os.path.getsize("../index/index.json") / 1024
print(f"Index size on disk: {index_size:.2f} KB")