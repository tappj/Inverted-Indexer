import json
import os
import re
import math
from nltk.stem import PorterStemmer

INDEX_PATH = "../index/index.json"
DOC_MAP_PATH = "../index/doc_id_map.json"
HITS_PATH = "../index/hits_scores.json"

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

print("Loading index...")
with open(INDEX_PATH, 'r') as f:
    inverted_index = json.load(f)

with open(DOC_MAP_PATH, 'r') as f:
    doc_id_map = json.load(f)

with open(HITS_PATH, 'r') as f:
    hits_scores = json.load(f)

print("Index loaded!")

def tokenize(text):
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return [ps.stem(token) for token in tokens if token not in STOPWORDS]

def make_ngrams(tokens, n=2):
    return ['_'.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def is_low_quality_url(url):
    patterns = [
        r'/news/page/\d+',
        r'/page/\d+',
        r'/\d{4}/\d{2}/',
        r'/\d{4}/$',
        r'#more-\d+',
        r'/news/$',
        r'/news$',
        r'/faculty/$',
        r'/faculty$',
        r'/events/$',
        r'/events$',
        r'/motherboard-',
        r'/huffington-post-',
        r'/rolling-stone-',
        r'/wired-',
        r'/daily-pilot-',
        r'/medium-',
    ]
    for pattern in patterns:
        if re.search(pattern, url):
            return True
    return False

def terms_are_close(positions_dict, terms, window=5):
    if len(terms) < 2:
        return False
    all_pos = [positions_dict.get(t, []) for t in terms]
    if any(len(p) == 0 for p in all_pos):
        return False
    for p1 in all_pos[0]:
        for p2 in all_pos[1]:
            if abs(p1 - p2) <= window:
                return True
    return False

while True:
    query = input("\nEnter query (or 'quit' to exit): ").strip()
    if query.lower() == 'quit':
        break

    terms = tokenize(query)
    print(f"Stemmed query terms: {terms}")

    bigrams = make_ngrams(terms, 2)
    trigrams = make_ngrams(terms, 3)
    all_query_terms = terms + bigrams + trigrams

    if not terms:
        print("No valid terms in query.")
        continue

    # AND logic on base terms only
    posting_lists = []
    for term in terms:
        if term in inverted_index:
            doc_ids = set(str(p['doc_id']) for p in inverted_index[term])
            posting_lists.append(doc_ids)
        else:
            posting_lists.append(set())

    if posting_lists:
        matching_docs = posting_lists[0]
        for pl in posting_lists[1:]:
            matching_docs = matching_docs.intersection(pl)
    else:
        matching_docs = set()

    # Build positions lookup per doc
    doc_positions = {}
    for term in terms:
        if term not in inverted_index:
            continue
        for posting in inverted_index[term]:
            doc_id = str(posting['doc_id'])
            if doc_id in matching_docs:
                if doc_id not in doc_positions:
                    doc_positions[doc_id] = {}
                doc_positions[doc_id][term] = posting.get('positions', [])

    total_docs = len(doc_id_map)
    scores = {}

    for term in all_query_terms:
        if term not in inverted_index:
            continue
        postings = inverted_index[term]
        df = len(postings)
        idf = math.log(total_docs / (df + 1))
        is_ngram = '_' in term
        ngram_boost = 2.0 if is_ngram else 1.0

        for posting in postings:
            doc_id = str(posting['doc_id'])
            if doc_id not in matching_docs:
                continue
            tf = posting['tf']
            tf_idf = (1 + math.log(tf)) * idf * ngram_boost
            if posting['important']:
                tf_idf *= 1.5

            # Proximity boost
            if len(terms) > 1 and doc_id in doc_positions:
                if terms_are_close(doc_positions[doc_id], terms):
                    tf_idf *= 1.5

            url = doc_id_map[doc_id]
            penalty = 0.3 if is_low_quality_url(url) else 1.0

            # HITS authority boost
            authority = hits_scores.get(doc_id, {}).get('authority', 0)
            hits_boost = 1.0 + authority

            scores[doc_id] = scores.get(doc_id, 0) + tf_idf * penalty * hits_boost

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    seen_urls = set()
    clean_results = []
    for doc_id, score in ranked:
        url = doc_id_map[doc_id].split('#')[0]
        if url not in seen_urls:
            seen_urls.add(url)
            clean_results.append((url, score))

    print(f"\nTop 5 results for '{query}':")
    for url, score in clean_results[:5]:
        print(f"  {url}")