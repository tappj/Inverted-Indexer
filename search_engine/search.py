import json
import os
import re
from nltk.stem import PorterStemmer

INDEX_PATH = "../index/index.json"
DOC_MAP_PATH = "../index/doc_id_map.json"

ps = PorterStemmer()

# Load index into memory
print("Loading index...")
with open(INDEX_PATH, 'r') as f:
    inverted_index = json.load(f)

with open(DOC_MAP_PATH, 'r') as f:
    doc_id_map = json.load(f)

print("Index loaded!")

def tokenize(text):
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return [ps.stem(token) for token in tokens]

# Search prompt
while True:
    query = input("\nEnter query (or 'quit' to exit): ").strip()
    if query.lower() == 'quit':
        break
    terms = tokenize(query)
    print(f"Stemmed query terms: {terms}")

    # AND logic: find doc_ids that appear in ALL terms' postings
    if not terms:
        print("No valid terms in query.")
        continue

    # Get posting lists for each term
    posting_lists = []
    for term in terms:
        if term in inverted_index:
            doc_ids = set(str(p['doc_id']) for p in inverted_index[term])
            posting_lists.append(doc_ids)
        else:
            posting_lists.append(set())  # term not found, empty set

    # AND = intersection of all sets
    if posting_lists:
        matching_docs = posting_lists[0]
        for pl in posting_lists[1:]:
            matching_docs = matching_docs.intersection(pl)
    else:
        matching_docs = set()

    # Score each matching document using tf-idf
    import math
    total_docs = len(doc_id_map)
    scores = {}

    for term in terms:
        if term not in inverted_index:
            continue
        postings = inverted_index[term]
        df = len(postings)  # how many docs contain this term
        idf = math.log(total_docs / (df + 1))  # +1 to avoid division by zero

        for posting in postings:
            doc_id = str(posting['doc_id'])
            if doc_id not in matching_docs:
                continue
            tf = posting['tf']
            tf_idf = (1 + math.log(tf)) * idf
            # Boost score if term appeared in important text
            if posting['important']:
                tf_idf *= 1.5
            scores[doc_id] = scores.get(doc_id, 0) + tf_idf

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 5 results for '{query}':")
    for doc_id, score in ranked[:5]:
        print(f"  {doc_id_map[doc_id]}")
