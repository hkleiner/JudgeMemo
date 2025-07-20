import random
from Dataset_Creation.TextManipulation.exceptions import NoChapterException
from Dataset_Creation.TextManipulation.ContentManipulator import ContentManipulator


class SwapContent(ContentManipulator):
    """
    A content manipulation class that performs random swaps of text segments.

    This class inherits from `ContentManipulator` and implements logic to swap randomly
    selected text ranges—paragraphs, sections, or chapters—ensuring constraints like
    minimum length and non-overlapping rules.
    """
    def execute(self, range_type: str, min_len: int = None, n: int = 2) -> tuple[str, list[list[list[int]]]]:
        """
        Randomly performs n swaps of text segments based on the specified range type.

        Args:
            range_type (str): The type of text range to operate on.
            min_len (int, optional): The minimum character length required for a range. Defaults:
                                     - 3000 for "section"
                                     - 50 for other types.
            n (int, optional): The number of swaps to perform. Each involves two unique, non-overlapping ranges.

        Returns:
            tuple[str, list[list[list[int]]]]:
                - updated_text (str): Text after performing swaps.
                - affected_words (list[list[list[int]]]): Character ranges of swapped segments.

        Raises:
            NoChapterException: If "chapter" range is specified but no chapters exist.
        """
        try:
            if min_len is None:
                min_len = 3000 if range_type == "section" else 50

            paragraphs, valid_indices, delimiter, possible_sections = self._prepare_data_range(range_type, min_len, n * 2)

            if len(valid_indices) < 2 * n:
                raise ValueError(f"Not enough valid {range_type}s to perform {n} swaps. Minimum required: {2 * n}.")

            affected_ranges = []

            for _ in range(n):
                if range_type == "section":
                    idx1 = random.choice(valid_indices)
                    valid_indices.remove(idx1)
                    idx1_range = list(possible_sections[idx1][1].keys())[0]
                    idx1_paragraphs = set(range(*idx1_range))
                    # Filter out overlapping sections
                    valid_indices = self._filter_non_overlapping(valid_indices, possible_sections, idx1_paragraphs)

                    if not valid_indices:
                        break  # no valid second index

                    idx2 = random.choice(valid_indices)
                    valid_indices.remove(idx2)
                    idx2_range = list(possible_sections[idx2][1].keys())[0]
                    idx2_paragraphs = set(range(*idx2_range))
                    # Update valid_indices again to ensure non-overlapping with idx2
                    valid_indices = self._filter_non_overlapping(valid_indices, possible_sections, idx2_paragraphs)

                    # Swap sections
                    selected_sections = [possible_sections[idx1], possible_sections[idx2]]
                    change_idx = []
                    affected_range = []
                    for _, ranges in selected_sections:
                        for keys, vals in ranges.items():
                            affected_range.append([vals[0][0], vals[-1][-1]])
                            change_idx.append(keys)

                    # Perform swap
                    hold = paragraphs[change_idx[0][0]:change_idx[0][1]]
                    paragraphs[change_idx[0][0]:change_idx[0][1]] = paragraphs[change_idx[1][0]:change_idx[1][1]]
                    paragraphs[change_idx[1][0]:change_idx[1][1]] = hold

                    affected_ranges.append(affected_range)
                else:  # paragraph or chapter
                    idx1, idx2 = random.sample(valid_indices, 2)
                    valid_indices.remove(idx1)
                    valid_indices.remove(idx2)

                    paragraphs[idx1], paragraphs[idx2] = paragraphs[idx2], paragraphs[idx1]
                    affected_ranges.append([self.char_ranges[idx1], self.char_ranges[idx2]])

            updated_text = self._reconstruct_text(paragraphs, delimiter)

            return updated_text, affected_ranges
        except NoChapterException:
            raise


class RemoveContent(ContentManipulator):
    """
    A content manipulation class that removes random text segments based on the specified range type.

    This class inherits from `ContentManipulator` and implements the `execute` method to remove
    specific ranges (e.g., paragraphs, sections, or chapters) from the text while ensuring certain
    constraints like minimum character length and non-overlapping removals.
    """
    def execute(self, range_type: str, min_len: int = None, n: int = 2) -> tuple[str, list[list[list[int]]]]:
        """
        Randomly removes `n` text segments (paragraphs, sections, or chapters) from the content.

        For range_type "section", it ensures that each removed section is non-overlapping with others
        already removed. It also respects a minimum character length requirement for valid removal.

        Args:
            range_type (str):
                The type of text range to remove from. One of {"paragraph", "section", "chapter"}.
            min_len (int, optional):
                The minimum combined character length a range must have to be considered removable.
                Defaults to:
                    - 3000 for "section"
                    - 50 for all other types
            n (int, optional):
                The number of text segments to remove. Defaults to 2.

        Returns:
            tuple[str, list[list[list[int]]]]:
                - updated_text (str): The resulting text after removals.
                - affected_ranges (list[list[list[int]]]): A list of removed character ranges,
                  where each item is a list containing [start, end] indices of the removed span.

        Raises:
            ValueError:
                If there are not enough valid elements to perform `n` removals.
            NoChapterException:
                If the content expects chapters but none can be identified.

        Notes:
            - Removed ranges are selected randomly from the available valid indices.
            - In "section" mode, sections overlapping with already removed ones are excluded from future sampling.
            - The method modifies the internal paragraph list and reconstructs the full text after deletion.
        """
        try:
            if min_len is None:
                min_len = 3000 if range_type == "section" else 50

            paragraphs, valid_indices, delimiter, possible_sections = self._prepare_data_range(range_type, min_len, n)

            if len(valid_indices) < 1 * n:
                raise ValueError(f"No valid {range_type} to remove. A minimum of {1 * n} is required!")

            affected_ranges = []  # give the character range that is going to be removed
            for _ in range(n):
                idx = random.choice(valid_indices)
                valid_indices.remove(idx)
                if range_type == "section":
                    idx_range = list(possible_sections[idx][1].keys())[0]
                    idx_paragraphs = set(range(*idx_range))
                    # Filter out overlapping sections
                    valid_indices = self._filter_non_overlapping(valid_indices, possible_sections, idx_paragraphs)

                    selected_section = possible_sections[idx]
                    for key, val in selected_section[1].items():
                        paragraphs = paragraphs[:key[0]] + paragraphs[key[1]:]
                        affected_ranges.append([val[0][0], val[-1][-1]])
                else:  # paragraph and chapter
                    # give the character range that is going to be removed
                    affected_ranges.append(self.char_ranges[idx])
                    paragraphs = paragraphs[:idx] + paragraphs[idx+1:]

            updated_text = self._reconstruct_text(paragraphs, delimiter)

            return updated_text, affected_ranges
        except NoChapterException:
            raise


class InsertContent(ContentManipulator):
    """
    A content manipulation class that inserts new text segments into an existing text body.

    This class inherits from `ContentManipulator` and supports inserting entire sections,
    paragraphs, or chapters from a separate text into the current content at random positions.
    """
    def execute(self, range_type: str, text_to_insert: str, min_len: int = None, n: int = 2) -> tuple[str, list[list[list[int]]]]:
        """
        Inserts `n` segments of the specified `range_type` from a given insertion text
        into random positions in the main text.

        Args:
            range_type (str):
                The type of content range to insert. One of {"paragraph", "section", "chapter"}.
            text_to_insert (str):
                The source text from which the insertable content is drawn.
            min_len (int, optional):
                Minimum character length for a valid range to be eligible for insertion.
                Defaults to:
                    - 3000 for "section"
                    - 50 for other types.
            n (int, optional):
                Number of insertions to perform. Defaults to 2.

        Returns:
            tuple[str, list[list[list[int]]]]:
                - updated_text (str): The full text after insertions.
                - affected_ranges (list[list[list[int]]]): A list of character index pairs
                  indicating where each insertion took place. Both start and end are equal,
                  representing the insertion point.

        Raises:
            ValueError:
                If the insertion text doesn't contain enough valid segments for the operation.
            NoChapterException:
                If the range_type is "chapter" but the structure contains no chapters.

        Notes:
            - Insertions are made after randomly selected paragraph indices in the target text.
            - For "section" insertions, overlapping sections in the insertion source are filtered out.
            - This function reconstructs the full text using the internal list of paragraphs.
        """
        try:
            if min_len is None:
                min_len = 3000 if range_type == "section" else 50
            # get paragraphs for manipulation text
            paragraphs, _, delimiter, _ = self._prepare_data_range(range_type, min_len, 1)
            # prepare insertion text
            preprocessor = ContentManipulator(text=text_to_insert)
            para_insert, valid_indices_insert, delimiter, sections_insert_text = preprocessor._prepare_data_range(range_type, min_len, 1)

            if len(valid_indices_insert) < 1 * n:
                raise ValueError(f"No valid {range_type} to insert. A minimum of {1 * n} is required!")

            affected_ranges = []  # give the character range that is going to be removed
            insert_ranges = {}
            for i in range(n):
                # choose index in text_to_manipulate after which paragraph the insert_text should be inserted
                insert_after_idx = random.choice(range(len(self.char_ranges)))  # choose one
                insert_idx = random.choice(valid_indices_insert)  # choose which valid text to insert (1)
                valid_indices_insert.remove(insert_idx)
                insert_at = insert_after_idx + 1
                if range_type == "section":
                    selected_section = sections_insert_text[insert_idx]
                    idx_range = list(selected_section[1].keys())[0]
                    idx_paragraphs = set(range(*idx_range))
                    # filter out overlapping sections
                    valid_indices_insert = self._filter_non_overlapping(valid_indices_insert, sections_insert_text, idx_paragraphs)

                    insert_ranges[(insert_at, i)] = selected_section[0]  # track changes to be done
                else:  # paragraph and chapter
                    insert_ranges[(insert_at, i)] = [para_insert[insert_idx]]  # track changes to be done
                # give the index of the character after which the text snippet will be inserted
                affected_ranges.append([self.char_ranges[insert_after_idx][1], self.char_ranges[insert_after_idx][1]])

            paragraphs = self._update_paragraphs(insert_ranges, paragraphs)
            updated_text = self._reconstruct_text(paragraphs, delimiter)

            return updated_text, affected_ranges
        except NoChapterException:
            raise


class RepeatContent(ContentManipulator):
    """
    A content manipulation class that duplicates (repeats) text segments directly after themselves.
    """
    def execute(self, range_type: str, min_len: int = None, n: int = 2, repetition_factor: int = 1) -> tuple[str, list[list[list[int]]]]:
        """
        Repeats `n` text segments by duplicating each one immediately after itself.

        Args:
            range_type (str):
                The type of text range to repeat. One of {"paragraph", "section", "chapter"}.
            min_len (int, optional):
                Minimum character length for a range to be considered valid.
                Defaults to:
                    - 3000 for "section"
                    - 50 for all other types.
            n (int, optional):
                Number of distinct segments to repeat. Defaults to 2.
            repetition_factor (int, optional):
                How many times to repeat each segment. Defaults to 1.

        Returns:
            tuple[str, list[list[list[int]]]]:
                - updated_text (str): The final text after repetition.
                - affected_ranges (list[list[list[int]]]): Start-end index ranges where each repeat begins.

        Raises:
            ValueError:
                If not enough valid ranges are available for repetition.
            NoChapterException:
                If "chapter" is specified but no chapters exist.
        """
        try:
            if min_len is None:
                min_len = 3000 if range_type == "section" else 50

            paragraphs, valid_indices, delimiter, possible_sections = self._prepare_data_range(range_type, min_len, n)

            if len(valid_indices) < n:
                raise ValueError(f"Not enough valid {range_type}s to repeat. A minimum of {n} is required!")

            affected_ranges = []
            insert_ranges = {}
            for i in range(n):
                idx = random.choice(valid_indices)
                valid_indices.remove(idx)
                if range_type == "section":
                    selected_section = possible_sections[idx]
                    idx_range = list(selected_section[1].keys())[0]
                    idx_paragraphs = set(range(*idx_range))
                    # filter out overlapping sections
                    valid_indices = self._filter_non_overlapping(valid_indices, possible_sections, idx_paragraphs)

                    for key, val in selected_section[1].items():
                        section_start, insert_pos = key
                        section_text = paragraphs[section_start:insert_pos]
                        repeated = [delimiter.join(section_text) for _ in range(repetition_factor)]
                        insert_ranges[(insert_pos, i)] = repeated  # track changes to be done
                        affected_ranges.append([val[-1][-1], val[-1][-1]])
                else:  # paragraph or chapter
                    para = paragraphs[idx]
                    repeated = [para for _ in range(repetition_factor)]
                    insert_pos = idx + 1
                    insert_ranges[(insert_pos, i)] = repeated  # track changes to be done
                    affected_ranges.append([self.char_ranges[idx][1], self.char_ranges[idx][1]])

            paragraphs = self._update_paragraphs(insert_ranges, paragraphs)
            updated_text = self._reconstruct_text(paragraphs, delimiter)

            return updated_text, affected_ranges
        except NoChapterException:
            raise


class ExchangeContent(ContentManipulator):
    """
    A content manipulation class that replaces randomly selected segments of text (paragraphs, chapters, or sections)
    with segments from externally provided texts.
    """
    def execute(self, range_type: str, texts_to_insert: list, min_len: int = None, n: int = 2, ratio: bool = False) -> tuple[str, list[list[list[int]]]]:
        """
        Replaces `n` content blocks (paragraphs, sections or chapters) in the original text with blocks from other texts.

        The replacement is done at the same granularity level (`range_type`) in both the original and
        the inserted texts. The segments to be replaced and inserted are randomly selected
        from among the valid ones that meet the `min_len` criterion.

        Args:
            range_type (str): The unit of content replacement — must be one of "paragraph", "chapter", or "section".
            texts_to_insert (list[str]): A list of texts from which content blocks will be extracted and inserted.
            min_len (int, optional): Minimum character length for a content block to be considered valid for replacement.
                                     If not provided, defaults to 3000 for sections and 50 otherwise.
            n (int): Number of content blocks to exchange. Must match the number of provided texts in `texts_to_insert`.
            ratio (bool): Flag variable to set if swaps shall be performed based on a fixed number 'n' or ratio-based.
                          Default is to False (meaning fixed number of swaps).

        Returns:
            tuple[str, list[list[int]]]:
                - updated_text (str): The modified text after performing the content exchanges.
                - affected_ranges (list[list[int]]): A list of [start, end] character index pairs
                                                     marking the regions in the original text that were replaced.

        Raises:
            ValueError: If `n` is larger than the number of `texts_to_insert`.
            ValueError: If `n` is larger than the number of `valid_indices`.
            NoChapterException: If the underlying helper methods raise this exception due to missing chapter information.
        """
        try:
            if min_len is None:
                min_len = 3000 if range_type == "section" else 50

            if n > len(texts_to_insert):
                raise ValueError(f"Expected {n} texts to insert from, got {len(texts_to_insert)}!")

            # prepare insertion texts
            paras_insert, valid_indices_to_insert, delimiters, possible_sections_to_insert = [], [], [], []
            for text_to_insert in texts_to_insert:
                preprocessor = ContentManipulator(text=text_to_insert)
                para_insert, valid_indices_insert, delimiter, sections_insert_text = preprocessor._prepare_data_range(range_type, min_len, 1)

                paras_insert.append(para_insert)
                valid_indices_to_insert.append(valid_indices_insert)
                delimiters.append(delimiter)
                possible_sections_to_insert.append(sections_insert_text)

            # preprocess text to manipulate
            paragraphs, valid_indices, delimiter, possible_sections = self._prepare_data_range(range_type, min_len, 1)

            if ratio:
                if range_type != "chapter":
                    tokens = self.text.split(' ')
                    n = round((len(tokens)/1000) + 2)
                    print(f"Number of ranges to change: {n} based on {len(tokens)} tokens")
                else:
                    raise ValueError(f"{range_type} not supported in dense setting!")

            if n > len(valid_indices):
                raise ValueError(f"Expected {n} {range_type}s in the original text, got {len(valid_indices)}!")

            affected_ranges = []
            for i in range(n):
                # select remove range from original text
                remove_idx = random.choice(valid_indices)
                valid_indices.remove(remove_idx)
                # select insert range from i-th text
                insert_idx = random.choice(valid_indices_to_insert[i])
                valid_indices_to_insert[i].remove(insert_idx)
                if range_type == "section":
                    idx_range = list(possible_sections[remove_idx][1].keys())[0]
                    idx_paragraphs = set(range(*idx_range))
                    # Filter out overlapping sections
                    valid_indices = self._filter_non_overlapping(valid_indices, possible_sections, idx_paragraphs)

                    selected_section = possible_sections[remove_idx]
                    for key, val in selected_section[1].items():
                        insert_sec = possible_sections_to_insert[i][insert_idx]
                        paragraphs = paragraphs[:key[0]] + [insert_sec[0][0]] + paragraphs[key[1]:]
                        affected_ranges.append([val[0][0], val[-1][-1]])
                else:  # paragraph and chapter
                    # insert range
                    paragraphs = paragraphs[:remove_idx] + [paras_insert[i][insert_idx]] + paragraphs[remove_idx + 1:]
                    # give the character range that is going to be removed
                    affected_ranges.append(self.char_ranges[remove_idx])

            updated_text = self._reconstruct_text(paragraphs, delimiter)

            return updated_text, affected_ranges
        except NoChapterException:
            raise
