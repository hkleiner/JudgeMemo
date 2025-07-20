import json
import os
import spacy
from transformers import AutoTokenizer

from huggingface_hub import login
login()

# Load spaCy English tokenizer
nlp = spacy.load("en_core_web_sm")

# Load LLaMA 3.3 tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

# Directory with text files
META_FILE = "../data/project_gutenberg/gold_pg-1900_meta_full.json"


def whitespace_tokenizer(text):
    return text.split()


def spacy_tokenizer(text):
    return [token.text for token in nlp(text)]


def llama33_tokenizer(text):
    return llama_tokenizer.encode(text, add_special_tokens=False)


def count_paragraphs(text):
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    return len(paragraphs)


def count_sentences(text):
    doc = nlp(text)
    return len(list(doc.sents))


def process_documents(meta_file):
    token_counts = {
        "whitespace": [],
        "spacy": [],
        "llama": [],
    }
    paragraph_counts = []
    sentence_counts = []
    character_counts = []

    with open(meta_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
        print(len(content))

        for entry in content:
            filepath = "." + entry["gold_text"]
            with open("."+filepath, "r", encoding="utf-8") as file:
                text = file.read()

                char_count = len(text)

                ws_tokens = whitespace_tokenizer(text)

                if len(ws_tokens) < 8000:  # only documents larger than 8000 tokens (simple tokenizer)
                    print(entry['id'])
                    continue

                spacy_tokens = spacy_tokenizer(text)
                llama_tokens = llama33_tokenizer(text)

                # Paragraphs and Sentences
                paragraph_count = count_paragraphs(text)
                sentence_count = count_sentences(text)

                token_counts["whitespace"].append(len(ws_tokens))
                token_counts["spacy"].append(len(spacy_tokens))
                token_counts["llama"].append(len(llama_tokens))
                paragraph_counts.append(paragraph_count)
                sentence_counts.append(sentence_count)
                character_counts.append(char_count)

                print(f"File: {entry['gold_text']}")
                print(f"  Characters        : {char_count}")
                print(f"  Paragraphs        : {paragraph_count}")
                print(f"  Sentences         : {sentence_count}")
                print(f"  Whitespace tokens: {len(ws_tokens)}")
                print(f"  spaCy tokens     : {len(spacy_tokens)}")
                print(f"  LLaMA 3.3 tokens : {len(llama_tokens)}")
                print()

        print("=== AVERAGE TOKEN COUNTS ===")
        for method, counts in token_counts.items():
            print("Number of documents: ", len(counts))
            avg = sum(counts) / len(counts) if counts else 0
            print(f"{method.capitalize()} average: {avg:.2f}")
        avg_paragraphs = sum(paragraph_counts) / len(paragraph_counts) if paragraph_counts else 0
        avg_sentences = sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0
        avg_chars = sum(character_counts) / len(character_counts) if character_counts else 0
        print(f"Paragraphs average: {avg_paragraphs:.2f}")
        print(f"Sentences average : {avg_sentences:.2f}")
        print(f"Character average : {avg_chars:.2f}")


if __name__ == "__main__":
    process_documents(META_FILE)
