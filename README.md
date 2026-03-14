# Search Engine — Inverted Indexer

Built for INF121 Assignment 3 at UCI.

## What it does
A two-part search engine that indexes ICS UCI web pages and retrieves 
ranked results for user queries.

- indexer.py — builds an inverted index from crawled HTML pages
- search.py — accepts console queries and returns ranked results

## Features
- Porter stemming and stopword filtering
- tf-idf scoring with important word boosting (title, headings, bold)
- AND query logic
- Bigram and trigram indexing for phrase matching
- Word position tracking with proximity boosting
- Anchor text indexing
- Exact duplicate page detection (MD5 hashing)
- Low quality URL penalization

## How to run
1. Place the ANALYST folder in the root of the project
2. Install dependencies: pip3 install beautifulsoup4 nltk
3. Build the index: python3 indexer.py
4. Run search: python3 search.py

## Output
Saves index files to the index/ folder.