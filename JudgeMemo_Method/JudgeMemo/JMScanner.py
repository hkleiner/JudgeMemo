from JudgeMemo import JMUtils
import spacy


class JMScan:
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
        self.sec_evaluation = sec_evaluation

    def set_sec_summary(self, sec_summary):
        self.sec_summary = sec_summary

    def set_prev_sec_summary(self, prev_summary):
        self.prev_summary = prev_summary

    def set_previous_context(self, prev_context):
        self.prev_context = prev_context

    @staticmethod
    def _create_section_key(section_number: int) -> str:
        return f"section_{section_number:02d}"


class JMScanner:
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
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _detokenize(self, tokens):
        return self.tokenizer.decode(tokens)

    def _split_sentences(self):
        doc = self.nlp(self.text)
        return [(sent.start_char, sent.end_char) for sent in doc.sents if sent.text.strip()]

    def scan(self,
             scan_range: int,
             overlap_ratio: float,
             scan_mode: str
            ):
        # TODO: scan self.text into sections of ca. scan_range tokens
        # - make sure that a section always ends with a full sentence. therefore, it is okay to have a little bit more
        # or less than scan_range tokens for a section
        # use self.tokenizer to tokenize the text
        # for each section of the text, we create an object of JMScan and save all the important infos there:
        # - the section text (not tokenized)
        # - the section number (count +1 for each section)
        # - the start character of the section (position in the text)
        # - the end character of the section (position in text) - exclusive! (technically the start of the next section)
        # if the scan_mode is "hard" (else case), there is no overlap between sections. The section ends with a
        # sentence and starts with a new consecutive one that is not part of the previous section
        # if the scan_mode is stride, we technically scan the same way as for "hard" but:
        # - we also collect the previous_context and set it for the scan
        # - the size of previous_context is defined by the overlap_ratio: it is overlap_ratio percent of the scan_range
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