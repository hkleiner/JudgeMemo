import json
import re
import numpy as np
import os
from multiprocessing import cpu_count
from datasets import Dataset


# Custom exception for invalid data source usage
class UnknownSourceExeption(Exception):
    def __init__(self):
        self.message = "At least one of the parameters [corpus_path] or [dataset_name] has to be defined!"
        super().__init__(self.message)


class DatasetAnalyser:
    def __init__(self, min_tokens, max_tokens, corpus_path=None, dataset=None, clean=False):
        self.MIN_TOKENS = min_tokens  # 8000
        self.MAX_TOKENS = max_tokens  # max__tokens
        self.short_stories_u_16, self.short_stories_8_16 = [], []
        self.corpus = corpus_path if corpus_path else None

        if clean:
            # Preprocess and deduplicate dataset
            self.dataset = dataset.map(self.extract_gutenberg_text, num_proc=cpu_count())
            self.remove_duplicates()
        else:
            self.dataset = dataset if dataset else None

    @staticmethod
    def fast_tokenize(text):
        # Lightweight whitespace-based tokenizer
        return re.split(r'\s+', text.strip())

    @staticmethod
    def extract_idx(file):
        # Extract numeric index from filename
        filename = file[4:-4]
        return int(filename)

    def analyze_books(self, idx_key, content_key):
        # Token count stats across dataset or corpus
        token_counts = []

        if self.corpus:
            for file in os.listdir(self.corpus):
                file_path = self.corpus + file
                file_idx = self.extract_idx(file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    token_counts.append((file_idx, len(self.fast_tokenize(content))))
        elif self.dataset:
            if not idx_key:
                token_counts = [(idx, len(self.fast_tokenize(sample[content_key]))) for idx, sample in enumerate(self.dataset) if
                                sample[content_key]]
            else:
                token_counts = [(sample[idx_key], len(self.fast_tokenize(sample[content_key]))) for sample in self.dataset if sample[content_key]]
        else:
            raise UnknownSourceExeption

        if not token_counts:
            return None, None, 0, {}

        token_counts_array = np.array([count[1] for count in token_counts])

        shortest_book = min(token_counts, key=lambda x: x[1])
        longest_book = max(token_counts, key=lambda x: x[1])
        avg_tokens = np.mean(token_counts_array)

        return shortest_book, longest_book, avg_tokens, token_counts

    def get_short_stories(self, token_counts):
        # Classify documents into short story buckets
        for idx, tok_c in token_counts:
            if tok_c <= self.MAX_TOKENS:
                self.short_stories_u_16.append((idx, tok_c))
                if tok_c >= self.MIN_TOKENS:
                    self.short_stories_8_16.append((idx, tok_c))

    @staticmethod
    def analyze_short_stories(token_counts):
        # Min, max, avg stats for filtered short stories
        token_counts_array = np.array([count[1] for count in token_counts])

        shortest_book = min(token_counts, key=lambda x: x[1])
        longest_book = max(token_counts, key=lambda x: x[1])
        avg_tokens = np.mean(token_counts_array)

        return shortest_book, longest_book, avg_tokens

    def remove_duplicates(self):
        # Remove duplicate entries based on ID
        seen_ids = set()
        cleaned_dataset = [entry for entry in self.dataset if entry['id'] not in seen_ids and not seen_ids.add(entry['id'])]
        self.dataset = Dataset.from_list(cleaned_dataset)

    @staticmethod
    def extract_gutenberg_text(example):
        # Parse Project Gutenberg metadata (title, author, release date)
        """Step 1: Extract Title and Author"""
        title_match = re.search(r"Title:\s*(.*?)\s*\n", example['text'])
        author_match = re.search(r"Author:\s*(.*?)\s*\n", example['text'])
        release_date_match = re.search(r"Release Date:\s*(.*?)\s*\n", example['text'])

        title = title_match.group(1).strip() if title_match else "Unknown"
        author = author_match.group(1).strip() if author_match else "Unknown"
        release_date = release_date_match.group(1).strip() if release_date_match else "Unknown"

        """Step 2: Extract text between START and END markers"""
        match = re.search(
            r'\*\*\*START OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*\n(.*?)\n\*\*\*END OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*',
            example['text'], re.DOTALL
        )
        text = match.group(1).strip() if match else ""

        """Step 3: Remove everything before the first 6 consecutive line breaks"""
        split_match = re.search(r'(?:\s*\n){6}(.*)', text, re.DOTALL)
        example['text'] = split_match.group(1).strip() if split_match else ""

        # Enrich entry with extracted metadata
        example["title"] = title
        example["author"] = author
        example["release_date"] = release_date

        return example

    def run_analysis(self, idx_key='id', content_key='text', path_min_max_stories=None):
        # Run full pipeline: analyze, filter, summarize, and optionally export
        shortest, longest, avg_tokens, token_counts = self.analyze_books(idx_key=idx_key, content_key=content_key)
        print(f"{len(token_counts)} documents\n")

        print(f"Shortest Book -> ID: {shortest[0]}, Tokens: {shortest[1]}")
        print(f"Longest Book  -> ID: {longest[0]}, Tokens: {longest[1]}")
        print(f"Average Token Count: {avg_tokens:.2f}\n")

        # Get short stories
        self.get_short_stories(token_counts)
        print(f"Found {len(self.short_stories_u_16)} short stories under {self.MAX_TOKENS}K tokens.")
        print(f"Found {len(self.short_stories_8_16)} short stories between {self.MIN_TOKENS}K and {self.MAX_TOKENS}K tokens.\n")

        if path_min_max_stories:
            # Save filtered short story metadata
            with open(path_min_max_stories, "w", encoding="utf-8") as f:
                json.dump(self.short_stories_8_16, f, ensure_ascii=False, indent=4)

        # Report stats on filtered subset
        shortest, longest, avg_tokens = self.analyze_short_stories(self.short_stories_8_16)
        print(f"Shortest Book ({self.MIN_TOKENS}K-{self.MAX_TOKENS}K)-> ID: {shortest[0]}, Tokens: {shortest[1]}")
        print(f"Longest Book ({self.MIN_TOKENS}K-{self.MAX_TOKENS}K) -> ID: {longest[0]}, Tokens: {longest[1]}")
        print(f"Average Token Count ({self.MIN_TOKENS}K-{self.MAX_TOKENS}K): {avg_tokens:.2f}")
