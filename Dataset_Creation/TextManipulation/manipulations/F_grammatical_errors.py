import string
from Dataset_Creation.TextManipulation.ContentManipulator import ContentManipulator
import random
import spacy
import lemminflect  # indirectly used for verb_tenses (inflect())


class TypoInserter(ContentManipulator):
    """
    Randomly inserts a randomly chosen number of typos in a given range at document-level. Character replacements are
    based on a QWERTY keyboard layout.

    Idea adapted from Matthew Anderson (Jul 5, 2019 at 22:33)
    @https://stackoverflow.com/questions/56908331/python-automatically-introduce-slight-word-typos-into-phrases
    """
    def __init__(self, text):
        super().__init__(text)
        self.near_by_keys = {
            'a': ['q', 'w', 's', 'x', 'z'],
            'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'e', 'r', 'f', 'c', 'x'],
            'e': ['w', 's', 'd', 'r'],
            'f': ['d', 'r', 't', 'g', 'v', 'c'],
            'g': ['f', 't', 'y', 'h', 'b', 'v'],
            'h': ['g', 'y', 'u', 'j', 'n', 'b'],
            'i': ['u', 'j', 'k', 'o'],
            'j': ['h', 'u', 'i', 'k', 'n', 'm'],
            'k': ['j', 'i', 'o', 'l', 'm'],
            'l': ['k', 'o', 'p'],
            'm': ['n', 'j', 'k', 'l'],
            'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'k', 'l', 'p'],
            'p': ['o', 'l'],
            'q': ['w', 'a', 's'],
            'r': ['e', 'd', 'f', 't'],
            's': ['w', 'e', 'd', 'x', 'z', 'a'],
            't': ['r', 'f', 'g', 'y'],
            'u': ['y', 'h', 'j', 'i'],
            'v': ['c', 'f', 'g', 'v', 'b'],
            'w': ['q', 'a', 's', 'e'],
            'x': ['z', 's', 'd', 'c'],
            'y': ['t', 'g', 'h', 'u'],
            'z': ['a', 's', 'x'],
            ' ': ['c', 'v', 'b', 'n', 'm']
        }

    def execute(self, start: int = 10, end: int = 10, dense: bool = False) -> tuple[str, list[list[int]]]:
        """
        Randomly modifies characters in the text by replacing them with nearby QWERTY keyboard characters.

        Args:
            start (int, optional): The minimum percentage (in %) of characters to modify. Defaults to 10 (10%).
            end (int, optional): The maximum percentage (in %) of characters to modify. Defaults to 10 (10%).
            dense (bool, optional): If False, typos are inserted randomly at document-level. If True, a certain range
            to insert the typos in is calculated.

        Returns:
            tuple[str, list[list[int]]]:
                A tuple containing:
                - updated_text (str): The modified version of the original text.
                - affected_chars (list[list[int]]): A list of affected character positions in the format
                  [[start_index, end_index], ...].

        Notes:
            - The method first determines how many characters should be altered.
            - It converts all characters to lowercase while recording their original capitalization.
            - It selects random character positions and replaces them with nearby keyboard characters.
            - Original capitalization is restored before reconstructing the final text.
            - The function ensures that only valid alphanumeric characters are altered.
        """
        tokens = self.text.split(' ')
        percentage = random.randint(start, end)  # get ratio

        message = list(self.text)
        capitalization = [False] * len(message)  # is a letter capitalized?
        # make all characters lowercase & record uppercase
        for i in range(len(message)):
            text = message[i]
            capitalization[i] = text.isupper()
            message[i] = text.lower()

        # get how many characters are affected based on the number of tokens in the text
        n_chars_to_flip = round((len(tokens) * percentage) / 100)
        print(
            f"Number of selected characters: {n_chars_to_flip} out of {len(tokens)} tokens (percentage: {percentage}%)")

        affected_chars = []  # list of characters that will be flipped
        if dense:
            sec_len = round(n_chars_to_flip * (5 * 2.5))  # average english word: 5 letters
            print(
                f"Inserted {n_chars_to_flip} typos in a section of {sec_len} characters")
            # randomly choose the start char for the typo-section
            section_start = random.randint(sec_len, len(message) - sec_len - 1)
            section_end = section_start + sec_len  # a section has 3000 characters
            for i in range(n_chars_to_flip):
                selected_char = random.sample(range(section_start, section_end - 1), k=1)[0]
                while message[selected_char] not in self.near_by_keys.keys():
                    selected_char = random.sample(range(section_start, section_end - 1), k=1)[0]
                affected_chars.append(selected_char)
        else:
            for i in range(n_chars_to_flip):
                selected_char = random.randint(0, len(message) - 1)
                while message[selected_char] not in self.near_by_keys.keys():
                    selected_char = random.randint(0, len(message) - 1)
                affected_chars.append(selected_char)

        for pos in affected_chars:
            nearby_chars = self.near_by_keys[message[pos]]
            message[pos] = random.choice(nearby_chars)  # choose one

        # reinsert capitalization
        for i in range(len(message)):
            if capitalization[i]:
                message[i] = message[i].upper()

        # recombine the message into a string
        updated_text = self._reconstruct_text(message, '')

        # maybe change this -> just to keep the same format as for other changes
        affected_chars = [[num, num] for num in affected_chars]

        return updated_text, affected_chars


class VerbTenseChanger(ContentManipulator):
    """
    A content manipulator that detects and modifies the tense of verb and auxiliary verb tokens in a text.

    This class uses spaCy for part-of-speech tagging and Lemminflect (via spaCy extensions)
    to perform tense inflection. It randomly selects a specified number of verbs (including auxiliaries)
    from the input text and toggles their tense between past and present.
    """
    def __init__(self, text):
        super().__init__(text)
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.TENSES = {
            "present_1": "VBP",
            "present_3": "VBZ",
            "past": "VBD"
        }

    def execute(self, start: int = 5, end: int = 5, dense: bool = False) -> tuple[str, list[list[int]]]:
        """
        Changes the tense of `n` randomly selected verbs or auxiliary verbs in the text.

        Verbs currently in past tense will be changed to present tense,
        and verbs in present tense will be changed to past tense.

        Args:
            start (int): The minimum percentage (in %) of verb tense changes to apply. Defaults to 5 (5%).
            end (int): The maximum percentage (in %) of verb tense changes to apply. Defaults to 5 (5%).
            dense (bool, optional): If False, verb tenses are changed randomly at document-level. If True, a certain range
            to change the verb tenses in is calculated.

        Returns:
            tuple[str, list[list[int]]]:
                - updated_text: The text with `n` verbs modified in tense.
                - affected_ranges: List of [start, end] character ranges in the original text where changes were made.
        """
        # tokenize the text using spaCy
        doc = self.spacy_nlp(self.text)
        verb_tokens = [token for token in doc if token.pos_ in {"VERB", "AUX"}]

        percentage = random.randint(start, end)  # get ratio
        # get how many verbs are affected based on the number of verbs in the text
        n = round((len(verb_tokens) * percentage) / 100)
        print(f"Number of selected verbs {n} out of {len(verb_tokens)} (percentage: {percentage}%)")

        if dense:
            # Step 1: Gather all sentences
            sents = list(doc.sents)
            valid_sections = []

            for i in range(len(sents)):
                verb_count = 0
                span_start = sents[i].start_char
                span_end = sents[i].end_char

                for j in range(i, len(sents)):
                    sent_verb_tokens = [token for token in sents[j] if token.pos_ in {"VERB", "AUX"}]
                    verb_count += len(sent_verb_tokens)
                    span_end = sents[j].end_char

                    if verb_count >= n:
                        valid_sections.append((span_start, span_end))
                        break  # stop growing this window once it's big enough

            if not valid_sections:
                raise ValueError(f"Couldn't find a dense section with at least {n} verbs. No changes applied.")

            # Step 2: Randomly pick one valid section
            span_start, span_end = random.choice(valid_sections)
            span_text = self.text[span_start:span_end]
            span_doc = self.spacy_nlp(span_text)
            span_verb_tokens = [token for token in span_doc if token.pos_ in {"VERB", "AUX"}]

            # Randomly sample n verbs within the span
            tokens_to_change = random.sample(span_verb_tokens, n)

            new_tokens = []
            for token in span_doc:
                if token in tokens_to_change:
                    tense = token.morph.get("Tense")
                    person = token.morph.get("Person")

                    if "Past" in tense:
                        new_tense = token._.inflect(self.TENSES["present_3"]) if "3" in person else token._.inflect(
                            self.TENSES["present_1"])
                    else:
                        new_tense = token._.inflect(self.TENSES["past"])

                    new_tokens.append((new_tense or token.text) + token.whitespace_)
                else:
                    new_tokens.append(token.text_with_ws)

            updated_text = self.text[:span_start] + self._reconstruct_text(new_tokens, "") + self.text[span_end:]
        else:
            # randomly sample n verb tokens
            tokens_to_change = random.sample(verb_tokens, n)

            new_tokens = []
            for token in doc:
                if token in tokens_to_change:
                    tense = token.morph.get("Tense")
                    person = token.morph.get("Person")

                    if "Past" in tense:
                        new_tense = token._.inflect(self.TENSES["present_3"]) if "3" in person else token._.inflect(
                            self.TENSES["present_1"])
                    else:
                        new_tense = token._.inflect(self.TENSES["past"])

                    # if inflection fails, fallback to original token
                    new_tokens.append((new_tense or token.text) + token.whitespace_)
                else:
                    new_tokens.append(token.text_with_ws)

            updated_text = self._reconstruct_text(new_tokens, "")
        affected_ranges = self._get_diff_spans(updated_text)

        return updated_text, affected_ranges


class WordOrderSwapper(ContentManipulator):
    """
    Randomly performes pair-wise swaps of a randomly chosen number of words in a given range at document-level.
    """
    def __init__(self, text):
        super().__init__(text)
        self.spacy_nlp = spacy.load("en_core_web_sm")

    def execute(self, start: int = 1, end: int = 1, dense: bool = False) -> tuple[str, list[list[int]]]:
        """
        Randomly swaps words within a sentence in the text. Applied to multiple sentences in the text based on 'start'
        and 'end'.

        Args:
            start (int, optional): The minimum percentage of sentences to perform word swaps in. Defaults to 1.
            end (int, optional): The maximum percentage of sentences to perform word swaps in. Defaults to 1.
            dense (bool, optional): If False, words are swapped randomly in sentences at document-level. If True, a
            certain sentence range to change the words in is calculated.

        Returns:
            tuple[str, list[list[int]]]:
                A tuple containing:
                - updated_text (str): The modified version of the original text with swapped words.
                - affected_ranges (list[list[int]]): A list of the start and end positions of the affected word pairs in the format
                  [[[start_index_1, end_index_1], [start_index_2, end_index_2]], ...].

        Notes:
            - This function introduces lexical noise by swapping two randomly chosen non-punctuation words
              within selected sentences of the text.
            - The number of affected sentences is determined by a randomly selected percentage between
              `start` and `end`.
            - If `dense` is True, the affected sentences are concentrated in a contiguous section of the
              text; otherwise, they are spread randomly across the document.
            - Whitespace and punctuation are preserved during the swap to maintain readability.
            - The function uses `self._get_diff_spans_multi_token(updated_text, 20)` to compute the ranges
              of swapped word pairs, returned as character index spans.
        """
        # process document with spaCy
        delimiter = "\n\n"
        paragraphs = self.text.split(delimiter)
        shortened_text = delimiter.join(paragraphs[1:])  # paragraph splitting
        doc = self.spacy_nlp(shortened_text)

        percentage = random.randint(start, end)  # get ratio
        # get how many sentences are affected based on the number of sentences in the text
        number_sents = len(list(doc.sents))
        number_affected_sents = round((number_sents * percentage) / 100)
        print(f"Number of selected sentences {number_affected_sents} out of {number_sents} sentences (percentage: {percentage}%)")

        affected_sents = []
        if dense:
            section_start = random.randint(number_affected_sents, number_sents - number_affected_sents - 1)
            section_end = section_start + number_affected_sents
            for _ in range(number_affected_sents):
                sent_no = random.sample(range(section_start, section_end - 1), k=1)[0]
                while len(list(doc.sents)[sent_no]) < 3 and sent_no not in affected_sents:
                    sent_no = random.choice(range(section_start, section_end - 1))
                affected_sents.append(sent_no)
        else:  # spread over the whole document
            for i in range(number_affected_sents):
                sent_no = random.randint(0, number_sents - 1)
                while len(list(doc.sents)[sent_no]) < 3 and sent_no not in affected_sents:
                    sent_no = random.randint(0, number_sents - 1)
                affected_sents.append(sent_no)

        updated_text, affected_ranges = "", []
        for i, sent in enumerate(doc.sents):
            tokens = list(sent)
            if i in affected_sents:
                affected_ranges.append([sent[1].idx, sent[-1].idx + len(sent[-1]) - 2])
                valid_indices = [j for j, tok in enumerate(tokens)
                                 if 0 < j < len(tokens)-2 and not tok.is_punct and "\n" not in tok.text]  # leave first and last token in place
                random.shuffle(valid_indices)

                new_tokens = ["" for _ in range(len(tokens))]
                c = 0
                for t, tok in enumerate(tokens):
                    if t not in valid_indices:  # first and last token of a sentence; punctuation
                        new_tokens[t] = tok.text + tok.whitespace_
                    else:  # shuffled tokens
                        token = tokens[valid_indices[c]]
                        new_tokens[t] = token.text + tok.whitespace_
                        c += 1

                text = "".join(new_tokens)
                updated_text += text
            else:
                updated_text += "".join([tok.text + tok.whitespace_ for tok in tokens])

        updated_text = paragraphs[0] + delimiter + updated_text

        return updated_text, affected_ranges


class PunctuationRemover(ContentManipulator):
    """
    Randomly removes a randomly chosen number of characters in a given range at document-level.
    """
    def execute(self, start: int = 40, end: int = 40, dense: bool = False) -> tuple[str, list[list[int]]]:
        """
        Randomly removes punctuation characters from the text.

        Args:
            start (int, optional): The minimum percentage (in %) of punctuation characters to remove. Defaults to 40 (40%).
            end (int, optional): The maximum percentage (in %) of punctuation characters to remove. Defaults to 40 (40%).
            dense (bool, optional):

        Returns:
            tuple[str, list[list[int]]]:
                A tuple containing:
                - updated_text (str): The modified version of the original text with removed punctuation.
                - affected_chars (list[list[int]]): A list of affected character positions in the format
                  [[start_index, end_index], ...].

        Notes:
            - The method identifies punctuation characters in the text.
            - A random selection of punctuation characters (within the given range) is removed.
            - The remaining text is reconstructed after removal.
            - Only punctuation characters are affected; letters and numbers remain unchanged.
        """
        message = list(self.text)

        # find punctuation in original text
        possible_chars_idx = [i for i, char in enumerate(message) if char in string.punctuation]

        percentage = random.randint(start, end)  # get ratio
        # get how many punctuation chars are affected based on the number of punctuation chars in the text
        n_chars_to_remove = round((len(possible_chars_idx) * percentage) / 100)
        print(f"Number of selected punct-chars {n_chars_to_remove} out of {len(possible_chars_idx)} (percentage: {percentage}%)")

        affected_chars = []
        if dense:
            sec_len = round(n_chars_to_remove * (5 * 2.5))  # average english word: 5 letters

            # Ensure valid section boundaries
            if len(message) < sec_len:
                section_start = 0
                section_end = len(message)
            else:
                section_start = random.randint(0, len(message) - sec_len)
                section_end = section_start + sec_len

            # Filter punctuation indices within the selected section
            section_punct_indices = [i for i in possible_chars_idx if section_start <= i < section_end]

            if len(section_punct_indices) < n_chars_to_remove:
                n_chars_to_remove = len(section_punct_indices)  # prevent errors

            affected_chars = random.sample(section_punct_indices, k=n_chars_to_remove)
        else:
            affected_chars = random.sample(possible_chars_idx, k=n_chars_to_remove)

        for i in affected_chars:
            message[i] = ''  # set empty string in place of punctuation

        affected_chars = [[num, num] for num in affected_chars]
        updated_text = self._reconstruct_text(message, '')

        return updated_text, affected_chars


class WordRemover(ContentManipulator):
    """
    Randomly removes a randomly chosen number of words in a given range at document-level. Only words consisting of
    pure alphanumeric characters are considered.
    """
    def execute(self, start: int = 5, end: int = 5, dense: bool = False) -> tuple[str, list[list[int]]]:
        """
        Randomly removes words from the text.

        Args:
            start (int, optional): The minimum percentage of tokens to remove. Defaults to 5.
            end (int, optional): The maximum percentage of tokens to remove. Defaults to 5.
            dense (bool, optional): If False, words are removed randomly at document-level. If True, a
            certain range to remove words in is calculated.

        Returns:
            tuple[str, list[list[int]]]:
                A tuple containing:
                - updated_text (str): The modified version of the original text with removed words.
                - affected_words (list[list[int]]): A list of affected word positions in the format
                  [[start_index, end_index], ...].

        Notes:
            - The method extracts words from the text while removing punctuation.
            - A random selection of words (within the given range) is removed.
            - Words are identified by their start and end indices, and replaced with an empty string.
            - The final text is reconstructed after word removal.
        """
        percentage = random.randint(start, end)
        _, no_punct_toks, delimiter = self._prepare_data_document()
        total_words = len(no_punct_toks)
        n_words_to_remove = round((total_words * percentage) / 100)
        print(f"Number of words to remove: {n_words_to_remove} out of {total_words} (percentage: {percentage}%)")

        if total_words == 0 or n_words_to_remove == 0:
            return self.text, []

        updated_text = list(self.text)

        if dense:
            avg_word_len = 5  # including space
            sec_len = round(n_words_to_remove * avg_word_len * 2.5)
            print(f"Targeting {n_words_to_remove} words in a section of {sec_len} characters")

            if len(updated_text) < sec_len:
                section_start = 0
                section_end = len(updated_text)
            else:
                section_start = random.randint(0, len(updated_text) - sec_len)
                section_end = section_start + sec_len

            # Filter tokens in the dense section
            dense_tokens = [
                (tok, span) for tok, span in no_punct_toks
                if section_start <= span[0] < section_end
            ]

            if len(dense_tokens) < n_words_to_remove:
                n_words_to_remove = len(dense_tokens)

            affected_words = random.sample(dense_tokens, k=n_words_to_remove)
        else:
            affected_words = random.sample(no_punct_toks, k=n_words_to_remove)

        # Sort affected words in reverse order to avoid shifting text
        affected_words = sorted(affected_words, key=lambda x: x[1][0], reverse=True)

        for (_, (s, e)) in affected_words:
            for i in range(s, e + 1):
                if i < len(updated_text):
                    updated_text[i] = ''

        updated_text = self._reconstruct_text(updated_text, '')
        affected_words = [span for _, span in affected_words]

        return updated_text, affected_words
