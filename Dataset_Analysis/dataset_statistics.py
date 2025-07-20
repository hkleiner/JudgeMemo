import json
import spacy
from transformers import AutoTokenizer
from huggingface_hub import login

# Authenticate with Hugging Face (required for some model downloads)
login()

# Load NLP tools
nlp = spacy.load("en_core_web_sm")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

# Metadata file with document references
META_FILE = "../data/project_gutenberg/gold_pg-1900_meta_full.json"


def whitespace_tokenizer(text):
    return text.split()


def spacy_tokenizer(text):
    return [token.text for token in nlp(text)]


def llama33_tokenizer(text):
    return llama_tokenizer.encode(text, add_special_tokens=False)


def count_paragraphs(text):
    return len([p for p in text.split("\n\n") if p.strip()])


def count_sentences(text):
    return len(list(nlp(text).sents))


def process_documents(meta_file):
    # Stores token counts for different tokenizers
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
        print(len(content))  # Number of documents in metadata

        for entry in content:
            filepath = "." + entry["gold_text"]
            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()
                char_count = len(text)

                ws_tokens = whitespace_tokenizer(text)

                # Skip short documents (under 8000 whitespace tokens)
                if len(ws_tokens) < 8000:
                    print(entry['id'])
                    continue

                # Apply all tokenizers
                spacy_tokens = spacy_tokenizer(text)
                llama_tokens = llama33_tokenizer(text)

                # Count structural features
                paragraph_count = count_paragraphs(text)
                sentence_count = count_sentences(text)

                # Collect stats
                token_counts["whitespace"].append(len(ws_tokens))
                token_counts["spacy"].append(len(spacy_tokens))
                token_counts["llama"].append(len(llama_tokens))
                paragraph_counts.append(paragraph_count)
                sentence_counts.append(sentence_count)
                character_counts.append(char_count)

                # Print per-file stats
                print(f"File: {entry['gold_text']}")
                print(f"  Characters        : {char_count}")
                print(f"  Paragraphs        : {paragraph_count}")
                print(f"  Sentences         : {sentence_count}")
                print(f"  Whitespace tokens: {len(ws_tokens)}")
                print(f"  spaCy tokens     : {len(spacy_tokens)}")
                print(f"  LLaMA 3.3 tokens : {len(llama_tokens)}")
                print()

        # Print averages across all processed documents
        print("=== AVERAGE TOKEN COUNTS ===")
        for method, counts in token_counts.items():
            print("Number of documents: ", len(counts))
            avg = sum(counts) / len(counts) if counts else 0
            print(f"{method.capitalize()} average: {avg:.2f}")
        print(f"Paragraphs average: {sum(paragraph_counts) / len(paragraph_counts):.2f}")
        print(f"Sentences average : {sum(sentence_counts) / len(sentence_counts):.2f}")
        print(f"Character average : {sum(character_counts) / len(character_counts):.2f}")


if __name__ == "__main__":
    process_documents(META_FILE)
