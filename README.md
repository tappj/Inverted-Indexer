# Search Engine — Inverted Indexer
This is the first part of a search engine I built for INF121 Assignment 3: M1.

## What it does
Reads through a collection of ICS web pages and builds an inverted index,
which maps each word to every document it appears in along with how often.

## How to run
1. Place the ANALYST folder in the root of the project
2. Install dependencies: pip3 install beautifulsoup4 nltk
3. Run: python3 indexer.py

## Output
Saves the index to the index/ folder as two JSON files.
