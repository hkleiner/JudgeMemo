import re
from .utils import get_formatted_idx, get_pg_url


class PGPreprocessor:
    """
    Cleans and structures Project Gutenberg-style texts for downstream processing.
    """
    def __init__(self, text, idx, title, author, out_path_dir):
        self.idx = idx if '-' not in idx else get_formatted_idx(idx)
        self.out_filename = self._get_out_path(out_path_dir)
        self.title = title
        self.author = author
        self.url = get_pg_url(self.idx)
        self.processed_text = self.preprocess(text)

    def _get_out_path(self, out_path_dir):
        return f"{out_path_dir}PG-{self.idx}.txt"

    def get_meta_data(self, our_id):
        # Returns structured metadata for the processed document
        gold_source = {
            "dataset": "Project Gutenberg",
            "source_id": self.idx,
            "title": self.title,
            "author": self.author,
            "year": '',
            "gutenberg": {
                "url": self.url,
                "num": int(self.idx)
            },
        }
        entry = {
            "id": our_id,
            "gold_text": self.out_filename[2:],
            "gold_source": gold_source
        }
        return entry

    @staticmethod
    def _remove_bracket_content(text):
        # Remove content in {} or []
        # Regular expression to match content between {} or []
        cleaned_text = re.sub(r'[{\[][^{}\[\]]*[}\]]', '', text)
        return cleaned_text

    @staticmethod
    def _remove_lines_with_stars(input_text):
        # Remove lines with asterisks and clean empty lines
        cleaned_text = re.sub(r'^.*\*.*$', '', input_text, flags=re.MULTILINE)
        # remove any leftover empty lines
        cleaned_text = "\n".join(line for line in cleaned_text.splitlines() if line.strip())
        return cleaned_text

    @staticmethod
    def _insert_line_breaks(text):
        # Add paragraph breaks after punctuation, avoid splitting abbreviations
        abbreviations = r'(M|Mr|Ms|Mrs|Dr|Prof|St|Lt|Col|Capt|Jr|Sr|vs|etc|e\.g|i\.e)'

        regex = re.compile(r'([”!?.\'"])(\n)')

        lines = text.splitlines(keepends=True)  # split into lines while preserving \n
        modified_lines = []

        for line in lines:
            # skip adding a new line break if the line ends with an abbreviation
            if re.search(rf'\b{abbreviations}.$', line.strip()):
                modified_lines.append(line)
            else:
                # add the line with modified line breaks
                modified_line = regex.sub(r'\1\n\n', line)
                modified_lines.append(modified_line)

        # join the modified lines back into the final text
        return ''.join(modified_lines)

    @staticmethod
    def _find_chapters(text):
        # Prefix potential chapter headings for easier navigation
        lines = text.split('\n')
        # Clean overly aggressive matches
        for i, line in enumerate(lines):
            line = line.strip()
            if (
                    line.startswith('CHAPTER') or
                    (line[:3].isupper() and line[:3].isalpha())
            ):
                lines[i] = "CHAPTER: " + line
        # CHECK AFTERWARDS MANUALLY!!
        # too many CHAPTERS possible
        for i, line in enumerate(lines):
            if line.startswith('CHAPTER: '):
                sub_line = line[9:]
                if sub_line.startswith('“'):
                    lines[i] = sub_line
        return '\n'.join(lines)

    @staticmethod
    def _merge_false_splits(text):
        # Fix false paragraph breaks (e.g. split mid-sentence)
        split_text = text.split('\n\n')
        for i in range(len(split_text)):
            if split_text[i][0].islower():
                split_text[i-1] += '\n' + split_text[i]
                split_text[i] = ''
        split_text = [par for par in split_text if par]  # remove empty strings
        merged_text = '\n\n'.join(split_text)
        return merged_text

    def preprocess(self, text):
        # Full cleaning pipeline
        no_brackets_text = self._remove_bracket_content(text)
        no_stars_text = self._remove_lines_with_stars(no_brackets_text)
        no_underscores_text = no_stars_text.replace('_', '')
        text_paragraphs = self._insert_line_breaks(no_underscores_text)
        merged_text = self._merge_false_splits(text_paragraphs)
        chaptered_text = self._find_chapters(merged_text)
        return chaptered_text

    def save_processed_text(self):
        # Write cleaned text to disk
        with open(self.out_filename, 'w', encoding='utf-8') as out:
            out.write(self.processed_text)
