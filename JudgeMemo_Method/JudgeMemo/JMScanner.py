from JudgeMemo import JMUtils
import spacy


class JMScan:
    """
    Represents a scanned section of a document.

    Attributes:
        sec_nr (str): Section identifier in the format "section_XX", where XX is a zero-padded number.
        sec_text (str): Text content of the section.
        sec_start (int): Start character index of the section in the original document text.
        sec_end (int): End character index (exclusive) of the section in the original document text.
        prev_summary (str or None): Summary of the previous section, if available.
        sec_summary (str or None): Summary of this section, if generated.
        sec_evaluation (str or None): Evaluation of this section, if performed.
        prev_context (str or None): Context text from the previous section for overlapping scans.

    Methods:
        set_sec_evaluation(sec_evaluation):
            Sets the evaluation result for this section.

        set_sec_summary(sec_summary):
            Sets the summary for this section.

        set_prev_sec_summary(prev_summary):
            Sets the summary of the previous section.

        set_previous_context(prev_context):
            Sets the previous context text for overlapping scan mode.

        _create_section_key(section_number: int) -> str:
            Creates a standardized section key string from an integer section number.
    """
    def __init__(self,
                 sec_text: str,  # content of a section
                 sec_nr: int,  # section number (ascending with each section)
                 sec_start: int,
                 sec_end: int
                 ):
        self.sec_nr = self._create_section_key(sec_nr)
        self.sec_text = sec_text
        self.sec_start = sec_start
        self.sec_end = sec_end
        # set later
        self.prev_summary = None
        self.sec_summary = None
        self.sec_evaluation = None
        self.prev_context = None

    def set_sec_evaluation(self, sec_evaluation):
        """
        Set the evaluation result for this section.

        Args:
            sec_evaluation (str): The evaluation text or score for this section.
        """
        self.sec_evaluation = sec_evaluation

    def set_sec_summary(self, sec_summary):
        """
        Set the summary text for this section.

        Args:
            sec_summary (str): The summary of the section.
        """
        self.sec_summary = sec_summary

    def set_prev_sec_summary(self, prev_summary):
        """
        Set the summary of the previous section.

        Args:
            prev_summary (str): The summary of the preceding section.
        """
        self.prev_summary = prev_summary

    def set_previous_context(self, prev_context):
        """
        Set the previous context text for this section, used in overlapping scans.

        Args:
            prev_context (str): Text from the previous section to provide context.
        """
        self.prev_context = prev_context

    @staticmethod
    def _create_section_key(section_number: int) -> str:
        """
        Create a standardized section key string from a section number.

        Args:
            section_number (int): The numeric index of the section.

        Returns:
            str: The formatted section key (e.g., "section_01").
        """
        return f"section_{section_number:02d}"


class JMScanner:
    """
    Scans and segments a document text into sections based on token count and sentence boundaries.

    Uses a tokenizer for tokenization and spaCy for sentence splitting.
    Supports two scanning modes:
        - "hard": non-overlapping sections ending at sentence boundaries.
        - "stride": overlapping sections with previous context included.

    Attributes:
        tokenizer: Tokenizer instance loaded from JMUtils.
        text (str): The full document text to scan.
        doc_id (str): Identifier for the document.
        nlp: spaCy NLP pipeline for sentence splitting (disabled components for speed).

    Methods:
        _tokenize(text):
            Tokenizes the input text using the tokenizer.

        _detokenize(tokens):
            Converts tokens back to text.

        _split_sentences():
            Splits the full text into sentence spans (start_char, end_char).

        scan(scan_range: int, overlap_ratio: float, scan_mode: str) -> list[JMScan]:
            Splits the document into sections according to scan_range and overlap_ratio,
            respecting sentence boundaries and scan mode.

            Args:
                scan_range (int): Approximate token count per section.
                overlap_ratio (float): Overlap ratio for stride mode (0.0 - 1.0).
                scan_mode (str): "hard" for no overlap, "stride" for overlapping sections.

            Returns:
                List of JMScan objects representing each scanned section.

            Raises:
                ValueError: If overlap_ratio is not in [0.0, 1.0).
    """
    def __init__(self,
                 tokenizer: str,
                 text: str,
                 doc_id: str,
                 ):
        self.tokenizer = JMUtils.load_tokenizer(tokenizer)
        self.text = text
        self.doc_id = doc_id
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])  # just for sentence splitting
        self.nlp.enable_pipe("senter")

    def _tokenize(self, text):
        """
        Tokenize input text using the tokenizer.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[int]: List of token IDs representing the text.
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _detokenize(self, tokens):
        """
        Convert tokens back into a text string.

        Args:
            tokens (List[int]): List of token IDs.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens)

    def _split_sentences(self):
        """
        Split the full document text into sentence spans using spaCy.

        Returns:
            List[Tuple[int, int]]: List of (start_char, end_char) tuples for each sentence.
        """
        doc = self.nlp(self.text)
        return [(sent.start_char, sent.end_char) for sent in doc.sents if sent.text.strip()]

    def scan(self,
             scan_range: int,
             overlap_ratio: float,
             scan_mode: str
            ):
        """
        Segment the document text into sections of approximately `scan_range` tokens, aligned with sentence boundaries.

        Depending on `scan_mode`, sections can be non-overlapping ("hard") or overlapping ("stride").

        Args:
            scan_range (int): Target number of tokens per section.
            overlap_ratio (float): Proportion of overlap between sections in "stride" mode (0.0 <= overlap_ratio < 1.0).
            scan_mode (str): Scanning mode. Either "hard" (no overlap) or "stride" (overlapping sections).

        Returns:
            List[JMScan]: List of JMScan objects, each representing a scanned section.

        Raises:
            ValueError: If `overlap_ratio` is not between 0.0 (inclusive) and 1.0 (exclusive).
        """
        if overlap_ratio >= 1.0 or overlap_ratio < 0.0:
            raise ValueError("The overlap_ratio has to be between 0.0 and 1.0!")

        sentence_spans = self._split_sentences()
        scans = []
        section_chars = []
        total_tokens = 0
        sec_nr = 1

        i = 0
        while i < len(sentence_spans):
            start_char, end_char = sentence_spans[i]
            sentence_text = self.text[start_char:end_char]
            token_count = len(self._tokenize(sentence_text))

            if total_tokens + token_count <= scan_range or not section_chars:
                section_chars.append((start_char, end_char))
                total_tokens += token_count
                i += 1
            else:
                sec_start = section_chars[0][0]
                sec_end = section_chars[-1][1]
                sec_text = self.text[sec_start:sec_end]

                scan = JMScan(
                    sec_text=sec_text,
                    sec_nr=sec_nr,
                    sec_start=sec_start,
                    sec_end=sec_end
                )

                if scan_mode == "stride":
                    if scans:
                        prev_scan = scans[-1]
                        prev_tokens = self._tokenize(prev_scan.sec_text)
                        overlap_tokens = int(scan_range * overlap_ratio)
                        prev_context_tokens = prev_tokens[-overlap_tokens:]
                        prev_context = self.tokenizer.decode(prev_context_tokens)
                        scan.set_previous_context('...' + prev_context)
                    else:
                        scan.set_previous_context('[no preceding section - beginning of text]')
                scans.append(scan)
                sec_nr += 1
                section_chars = []
                total_tokens = 0

        # Handle remaining sentences
        if section_chars:
            sec_start = section_chars[0][0]
            sec_end = section_chars[-1][1]
            sec_text = self.text[sec_start:sec_end]

            scan = JMScan(
                sec_text=sec_text,
                sec_nr=sec_nr,
                sec_start=sec_start,
                sec_end=sec_end
            )

            if scan_mode == "stride" and scans:
                prev_scan = scans[-1]
                prev_tokens = self._tokenize(prev_scan.sec_text)
                overlap_tokens = int(scan_range * overlap_ratio)
                prev_context_tokens = prev_tokens[-overlap_tokens:]
                prev_context = self.tokenizer.decode(prev_context_tokens)
                scan.set_previous_context("..." + prev_context)

            scans.append(scan)

        return scans


if __name__ == "__main__":
    text = JMUtils.load_prompt_template("../data/project_gutenberg/gold_dataset/PG-642.txt")

    # Create a scanner instance
    scanner = JMScanner(
        tokenizer="meta-llama/Llama-3.3-70B-Instruct",
        text=text,
        doc_id="test_doc_001"
    )

    # Scan with stride technique and overlap
    scan_results = scanner.scan(
        scan_range=2000,
        overlap_ratio=0.1,
        scan_mode="stride"  # or "hard"
    )

    # Print results
    for scan_res in scan_results:
        print(f"\n--- {scan_res.sec_nr} ---")
        print(f"Text: {scan_res.sec_text}")
        print(f"Start: {scan_res.sec_start}, End: {scan_res.sec_end}")
        if scan_res.prev_context:
            print(f"Prev context: {scan_res.prev_context}")