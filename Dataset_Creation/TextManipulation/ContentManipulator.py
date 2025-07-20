import re
import string
from spacy.tokens import Doc
from Dataset_Creation.TextManipulation.utils import _get_char_ranges, _find_valid_paragraph_sections
from Dataset_Creation.TextManipulation.exceptions import NoChapterException
from difflib import SequenceMatcher


class ContentManipulator:
    """
    Parent class for repetitive processes
    """
    def __init__(self, text):
        self.text = text
        self.char_ranges = []

    def _prepare_data_range(self, range_type: str, min_len: int, min_chap: int = 2) -> tuple[list[str], list[int], str, list[tuple[list[str], dict[tuple[int, int], list[list[int]]]]] | None]:
        """
        Prepares the text and identifies valid ranges for manipulation based on the specified range type.

        Depending on the level of granularity (`range_type`), this function splits the input text into
        paragraphs, sections, or chapters, filters them by length and content criteria, and computes their
        character index ranges for downstream manipulation.

        Args:
            range_type (str):
                The level of text segmentation. Must be one of:
                - "paragraph": Treats each paragraph (split by double newlines) as a unit.
                - "section": Groups consecutive paragraphs into variable-length sections with a minimum total length.
                - "chapter": Extracts blocks starting with "CHAPTER:" and treats them as logical units.

            min_len (int):
                The minimum number of characters a paragraph or group of paragraphs (section) must have to be considered valid.
                Ignored for "chapter" mode.

            min_chap (int, optional):
                The minimum number of chapters required in "chapter" mode. Defaults to 2.

        Returns:
            tuple[list[str], list[int], str, list | None]:
                - segments (list[str]): The text split into paragraphs, sections, or chapters.
                - valid_ranges (list[int]): Indices of segments eligible for manipulation.
                - delimiter (str): The delimiter used for splitting/reconstructing the text.
                - possible_sections (list | None): Only used in "section" mode. A list of tuples:
                  (section_paragraphs, {(start_idx, end_idx): char_ranges}). Returns `None` for other modes.

        Raises:
            NoChapterException: If `range_type` is "chapter" and fewer than `min_chap` chapter markers are found.

        Notes:
            - "paragraph" mode filters out any paragraph shorter than `min_len` or containing "CHAPTER:".
            - "section" mode dynamically builds sections by appending paragraphs until `min_len` is reached.
            - "chapter" mode relies on the presence of "CHAPTER:" markers in the text and includes any leading text
              before the first chapter as a pseudo-chapter for reconstruction purposes.
        """
        if range_type == "paragraph":
            delimiter = "\n\n"
            paragraphs = self.text.split(delimiter)
            # extract character ranges and filter valid paragraphs
            self.char_ranges = _get_char_ranges(paragraphs, delimiter)

            valid_ranges = [i for i, paragraph in enumerate(paragraphs) if len(paragraph.strip()) >= min_len and
                             "CHAPTER: " not in paragraph]

            return paragraphs, valid_ranges, delimiter, None
        elif range_type == "section":
            # here: min_len default 3000
            delimiter = "\n\n"
            paragraphs = self.text.split(delimiter)
            self.char_ranges = _get_char_ranges(paragraphs, delimiter)

            possible_sections = _find_valid_paragraph_sections(paragraphs=paragraphs, char_ranges=self.char_ranges, min_chars=min_len)

            valid_ranges = []
            for i in range(len(possible_sections)):
                if all("CHAPTER: " not in paragraph for paragraph in possible_sections[i][0]):
                    valid_ranges.append(i)

            return paragraphs, valid_ranges, delimiter, possible_sections
        elif range_type == "chapter":
            delimiter = "\n\n\nCHAPTER: "
            matches = list(re.finditer(r'CHAPTER:\s', self.text))

            if len(matches) < min_chap:
                raise NoChapterException()

            chapters = []
            for i, match in enumerate(matches):
                start_idx = match.end()  # content that directly follows "CHAPTER: "
                end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(self.text)
                chapter_content = self.text[start_idx:end_idx].strip()
                chapters.append(chapter_content)
                # extract chapter character ranges
                self.char_ranges.append([start_idx, end_idx])

            valid_ranges = [r for r in self.char_ranges]
            # add non-chapters to chapters and char_ranges to be able to reconstruct the text after the manipulation
            s, _ = self.char_ranges[0]
            self.char_ranges.insert(0, [0, s-len(delimiter)])  # everything that comes before the first chapter
            text = list(self.text)[:s-len(delimiter)]
            chapters.insert(0, ''.join(text))

            valid_chapters = [i for i, r in enumerate(self.char_ranges) if r in valid_ranges]

            return chapters, valid_chapters, delimiter, None
        else:
            raise ValueError("Unsupported range type. Supported types: 'paragraph', 'section', 'chapter'.")

    def _prepare_data_document(self) -> tuple[list[str], list[tuple[int, list[int]]], str]:
        """
        Prepares token-level data from the document for fine-grained manipulation.

        This method splits the text into tokens based on whitespace, extracts character
        ranges for each token, and filters out tokens that contain punctuation, uppercase
        letters, or newline characters.

        Returns:
            tuple[list[str], list[tuple[int, list[int]]], str]:
                - processed_text (list[str]): The tokenized version of the text, split by whitespace.
                - no_punct_tok (list[tuple[int, list[int]]]): A list of tuples where each tuple contains:
                    - the token's index in the original list
                    - its corresponding character range [start, end]
                  Only tokens without punctuation, uppercase letters, or newlines are included.
                - delimiter (str): The delimiter used for tokenization (" ").
        """
        delimiter = " "
        processed_text = self.text.split(delimiter)
        print(processed_text)

        self.char_ranges = _get_char_ranges(processed_text, delimiter)

        # only return tokens that do not contain any punctuation, are upper case or have new lines
        no_punct_tok = [(i, self.char_ranges[i])
                        for i, tok in enumerate(processed_text)
                        if not any(t in string.punctuation or t == "\n" or t.isupper()
                                   for t in tok)]

        return processed_text, no_punct_tok, delimiter

    @staticmethod
    def _update_paragraphs(insert_ranges, paragraphs):
        """
        Inserts repeated or new paragraph segments into the existing paragraph list
        at specified positions, maintaining correct order by applying changes in reverse.

        This method ensures that multiple insertions do not interfere with each other
        by sorting the insertion positions in descending order and applying them from
        the end of the text toward the beginning.

        Args:
            insert_ranges (dict[tuple[int, int], list[str]]):
                A dictionary where each key is a tuple of the form (insert_pos, i), where:
                    - insert_pos (int): The index in `paragraphs` after which the content should be inserted.
                    - i (int): An auxiliary index to differentiate between multiple insertions at the same position.
                Each value is a list of paragraph strings to insert.
            paragraphs (list[str]):
                The list of original paragraph strings that will be updated.

        Returns:
            list[str]: The updated list of paragraphs after all insertions have been applied.
        """
        # sort descending (insert_at/insert_pos)
        sorted_insert_ranges = dict(sorted(insert_ranges.items(), key=lambda item: item[0][0], reverse=True))
        # update ranges afterwards from the end to the beginning of the text to ensure the correct order is maintained
        for (pos, _), para in sorted_insert_ranges.items():
            paragraphs = (paragraphs[:pos] +
                          para +
                          paragraphs[pos:])
        return paragraphs

    @staticmethod
    def _reconstruct_text(paragraphs: list[str], delimiter: str) -> str:
        """
        Reconstructs the full text from a list of text segments using the specified delimiter.

        Args:
            paragraphs (list[str]): The list of text segments (e.g., paragraphs, sections, or tokens) to join.
            delimiter (str): The delimiter used to concatenate the segments.

        Returns:
            str: The reconstructed text.
        """
        return delimiter.join(paragraphs)

    @staticmethod
    def _filter_non_overlapping(indices: list[int], sections, occupied_paragraphs: set[int]) -> list[int]:
        """
        Filters out section indices that overlap with already-used paragraphs.

        Args:
            indices (list[int]): Current valid section indices.
            sections (list[tuple]): All possible sections with ranges.
            occupied_paragraphs (set[int]): Paragraph indices that must not overlap.

        Returns:
            list[int]: Filtered list of valid section indices.
        """
        return [
            idx for idx in indices
            if occupied_paragraphs.isdisjoint(set(range(*list(sections[idx][1].keys())[0])))
        ]

    @staticmethod
    def _get_diff_spans_seqMatcher(text1: str, text2: str, doc: Doc) -> list[list[int]]:
        """
        Computes the character-level differences between two strings and returns spans of changed tokens
        in the original text.

        Uses difflib's SequenceMatcher to identify non-equal segments, and then expands those
        segments to fully include the affected tokens in `text1`.

        Args:
            text1 (str): The original string to compare.
            text2 (str): The modified string to compare against the original.

        Returns:
            list[list[int]]: A list of [start, end] index pairs, where each pair marks a token-aligned character
                             span in `text1` that differs from `text2`.
        """
        matcher = SequenceMatcher(None, text1, text2)
        # extract character index ranges for each token in the original text
        token_bounds = [(token.idx, token.idx + len(token)) for token in doc]

        spans = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":  # only process differences (not equal parts)
                continue

            # collect all tokens from original text that overlap with the differing span [i1, i2)
            matching_tokens = [
                (start, end) for start, end in token_bounds
                if (start < i2 and end > i1)  # this token overlaps the diff span
            ]
            # if any overlapping tokens found, create a span from the first to the last
            if matching_tokens:
                span_start = matching_tokens[0][0]
                span_end = matching_tokens[-1][1]
                # only add valid, non-zero-length spans
                if span_end > span_start:
                    spans.append([span_start, span_end])

        # remove duplicates and sort by start index
        spans = set(tuple(span) for span in spans)

        # convert back to list of lists (instead of list of tuples)
        return [list(span) for span in spans]

    def _get_diff_spans(self, updated_text):
        """
        Identifies the character ranges where differences occur between the original text and the updated text.

        This method compares the original text (tokenized) with the updated version (split by spaces), and tracks
        the character ranges that were modified. It assumes that the text is tokenized in a similar way during
        both the comparison and the update.

        Args:
            updated_text (str): The updated version of the original text after some manipulation or change.

        Returns:
            list[list[int]]: A list of character ranges [start, end] in the original text, where differences
                             between the original and updated texts occur. Each range corresponds to the position
                             of the affected text in the original text.

        Notes:
            - This method compares each word from the tokenized original text with the updated text, and collects the
              character ranges where discrepancies occur.
            - It assumes that tokenized text splits by spaces (as `u_text.split(" ")` is used for comparison).
        """
        delimiter = " "
        affected_ranges = []
        u_text = updated_text.split(delimiter)
        tokenized_o_text, _, delimiter = self._prepare_data_document()
        for i, (c_o, c_u) in enumerate(zip(tokenized_o_text, u_text)):
            if c_o != c_u:
                affected_ranges.append(self.char_ranges[i])

        return affected_ranges

    def _get_diff_spans_multi_token(self, updated_text, max_lookahead=5):
        """
        Finds character spans in the original text that differ from the updated version,
        handling multi-token substitutions (e.g., one token replaced by two).

        Returns:
            list[list[int]]: Character spans [start, end] in the original text where changes occurred.
        """
        delimiter = " "
        u_text = updated_text.split(delimiter)
        o_text, _, _ = self._prepare_data_document()  # Returns tokenized original and char ranges
        affected_ranges = []

        i, j = 0, 0  # Pointers for original and updated text

        while i < len(o_text) and j < len(u_text):
            if o_text[i] == u_text[j]:
                i += 1
                j += 1
            else:
                # Begin mismatch block
                start_i = i
                found_match = False
                lookahead_i = 1  # Default values to prevent unbound variable error
                lookahead_j = 1

                # Try to find a match window ahead
                for li in range(1, max_lookahead):
                    for lj in range(1, max_lookahead):
                        if i + li < len(o_text) and j + lj < len(u_text):
                            if o_text[i + li] == u_text[j + lj]:
                                lookahead_i = li
                                lookahead_j = lj
                                found_match = True
                                break
                    if found_match:
                        break

                if found_match:
                    for k in range(start_i, i + lookahead_i):
                        affected_ranges.append(self.char_ranges[k])
                    i += lookahead_i
                    j += lookahead_j
                else:
                    # No match in lookahead window; treat as individual mismatch
                    affected_ranges.append(self.char_ranges[i])
                    i += 1
                    j += 1

        # Any remaining unmatched original tokens
        while i < len(o_text):
            affected_ranges.append(self.char_ranges[i])
            i += 1

        return affected_ranges
