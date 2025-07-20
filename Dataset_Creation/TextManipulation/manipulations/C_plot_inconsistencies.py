import json
import os
from Dataset_Creation.TextManipulation.ContentManipulator import ContentManipulator
import random
import spacy
import re


class ExchangeEntities(ContentManipulator):
    """
    A class for replacing named entities in text with fictional or alternative names.
    """
    def __init__(self, text):
        super().__init__(text)
        self.spacy_nlp = spacy.load("en_core_web_sm")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ENT_PATH = os.path.join(base_dir, "./src/entities.json")
        self.ENTITIES = json.load(open(ENT_PATH, 'r', encoding='utf-8'))

    def execute(self, entity: str = "PERSON", start: int = 10, end: int = 10) -> tuple[str, list[list[int]]]:
        """
        Executes the entity replacement process on the input text.

        Replaces a randomly sampled percentage (between `start` and `end`) of named entities of the specified type
        with entries from the custom `ENTITIES` list.

        Args:
            entity (str): The type of named entity to swap. Must be one of spaCy's recognized NER labels.
            start (int): The minimum percentage of entities to replace (e.g. 10%).
            end (int): The maximum percentage of entities to replace (e.g. 10%).

        Returns:
            tuple[str, list[list[int]]]:
                - The updated text with replaced entities.
                - A list of character span ranges indicating where changes occurred.

        Raises:
            ValueError: If an unsupported entity type is provided.

        Notes:
            - Entity detection is based on spaCy's pre-trained NER model.
            - Only handles single-token entities.
        """
        if entity not in self.ENTITIES.keys():
            raise ValueError(f"Unsupported entity type {entity}! Only supports: {self.ENTITIES.keys()}")

        ent_list = self.ENTITIES[entity]  # get which entities shall be processed

        doc = self.spacy_nlp(self.text)

        entities = [ent[0]
                    for ent in doc.ents
                    if ent.label_ == entity and len(ent) == 1 and ent.text[0].isupper() and ent.text != "CHAPTER"]  # get single-token entities
        print(f"Detected entities: {entities}")

        percentage = random.randint(start, end)  # get ratio
        n = round((len(entities) * percentage) / 100)  # get how many entity mentions are affected
        print(f"Number of selected entities {n} out of {len(entities)} detected entities (percentage: {percentage}%)")

        tokens_to_change = random.sample(entities, n)  # sample n entities to change
        new_tokens = []
        affected_ranges = []
        for tok in tokens_to_change:
            matches = [m.start() for m in re.finditer(re.escape(tok.text), self.text)]  # returns start index of match
            selected_start = random.choice(matches)
            end = selected_start + len(tok)
            substitute = random.choice(ent_list['1'])
            while substitute in self.text:  # check if entity is in text already
                substitute = random.choice(ent_list['1'])  # if yes, choose a new one
            new_tokens.append([substitute, [selected_start, end]])
            affected_ranges.append([selected_start, end])
            self.ENTITIES[entity]['1'].remove(
                substitute)  # remove entity to avoid choosing the same over and over again

        sorted_new_tokens = sorted(new_tokens, key=lambda x: x[1][1], reverse=True)
        updated_text = list(self.text)
        for substitute in sorted_new_tokens:
            tok, (start, end) = substitute[0], (int(substitute[1][0]), int(substitute[1][1]))
            print(f"{start}-{end}: {''.join(updated_text[start:end])} substituted with {tok}")
            updated_text[start:end] = tok

        updated_text = ''.join(updated_text)

        return updated_text, affected_ranges


class SwapEntities(ContentManipulator):
    """
    A class for swapping named entities of the same type within a text.
    Ensures entities are swapped with other entities that have different text values.
    """
    def __init__(self, text):
        super().__init__(text)
        self.spacy_nlp = spacy.load("en_core_web_sm")

    def execute(self, entity: str = "PERSON", start: int = 10, end: int = 10) -> tuple[str, list[list[int]]]:
        """
        Executes the entity swapping process on the input text.

        Randomly swaps a percentage (between `start` and `end`) of named entities of the specified type
        with other *different* entities of the same type found in the text.

        Args:
            entity (str): The type of named entity to swap. Must be one of spaCy's recognized NER labels.
            start (int): The minimum percentage of entities to swap.
            end (int): The maximum percentage of entities to swap.

        Returns:
            tuple[str, list[list[int]]]:
                - The updated text with swapped entities.
                - A list of character span ranges indicating where changes occurred.

        Notes:
            - Only handles single-token entities.
            - Entities are swapped only if they have different text values.
        """
        doc = self.spacy_nlp(self.text)

        # Collect single-token entities of the specified type, grouped by their text
        entity_occurrences = {}
        for ent in doc.ents:
            if ent.label_ == entity and len(ent) == 1 and ent.text[0].isupper():
                entity_occurrences.setdefault(ent.text, []).append(ent)

        # Only keep unique text entities (to ensure different swap partners)
        unique_entities = [ents[0] for ents in entity_occurrences.values()]
        print(f"Unique detected entities: {[ent.text for ent in unique_entities]}")

        if len(unique_entities) < 2:
            raise ValueError(f"Not enough *different* entities to swap. Expected at least 2, got {len(unique_entities)}")

        percentage = random.randint(start, end)
        n = round((len(unique_entities) * percentage) / 100)
        n = max(2, n)  # ensure at least 2 to swap
        print(f"Swapping {n} unique entities out of {len(unique_entities)} detected (percentage: {percentage}%)")

        # Sample unique entities to swap
        tokens_to_swap = random.sample(unique_entities, min(n, len(unique_entities)))

        # Prepare swap pairs by shifting the list (cyclic swap)
        swap_pairs = list(zip(tokens_to_swap, tokens_to_swap[1:] + [tokens_to_swap[0]]))

        # Record substitutions and their spans
        new_tokens = []
        affected_ranges = []
        for source_ent, target_ent in swap_pairs:
            source_start = source_ent.start_char
            source_end = source_ent.end_char
            substitute = target_ent.text
            new_tokens.append([substitute, [source_start, source_end]])
            affected_ranges.append([source_start, source_end])

        # Sort by end index descending to avoid messing up spans when replacing
        sorted_new_tokens = sorted(new_tokens, key=lambda x: x[1][1], reverse=True)
        updated_text = list(self.text)
        for substitute in sorted_new_tokens:
            tok, (start, end) = substitute[0], (int(substitute[1][0]), int(substitute[1][1]))
            print(f"{start}-{end}: {''.join(updated_text[start:end])} swapped with {tok}")
            updated_text[start:end] = tok

        updated_text = ''.join(updated_text)

        return updated_text, affected_ranges


class AddTempInconsistencies(ContentManipulator):
    """
    Adds temporal inconsistencies to a text by appending predefined, neutral sentences
    referencing modern concepts to randomly selected paragraphs.
    """

    def __init__(self, text):
        super().__init__(text)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.PATH = os.path.join(base_dir, "./src/temp_inconst_neutral_sentences.txt")
        self.sentences = self._load_neutral_sentences()

    def _load_neutral_sentences(self):
        """
        Loads a list of neutral anachronistic sentences from the predefined text file.

        Returns:
            list[str]: Stripped, individual neutral sentences.
        """
        sentences = []
        with open(self.PATH, 'r', encoding='utf-8') as f:
            for sent in f:
                sentences.append(sent.strip())
        return sentences

    def execute(self, range_type: str = "paragraph", n: int = 2, min_len: int = 50, ratio: bool = True):
        """
        Appends `n` neutral anachronistic sentences to randomly selected paragraphs in the text.

        Only paragraphs of minimum length `min_len` are eligible.
        The insertion point is the end of the selected paragraph.
        Records character ranges where sentences were inserted.

        Args:
            range_type (str): Content range type; currently only "paragraph" supported.
            n (int): Number of inconsistencies to introduce (default 2).
            min_len (int): Minimum paragraph length in characters to qualify (default 50).
            ratio (bool): If True, sets `n` proportional to text length; else uses fixed `n`.

        Returns:
            tuple[str, list[list[int]]]: Modified text and list of insertion character ranges.

        Raises:
            ValueError: If `range_type` is not "paragraph".
            ValueError: If fewer than `n` eligible paragraphs exist.
        """
        if range_type != "paragraph":
            raise ValueError("Manipulation is only applicable to paragraphs.")

        paragraphs, valid_indices, delimiter, _ = self._prepare_data_range(range_type=range_type, min_len=min_len)

        if ratio:
            tokens = self.text.split()
            n = round((len(tokens) / 1000) + 1)
            print(f"Number of paragraphs to modify: {n} out of {len(tokens)} tokens")
        else:
            if len(valid_indices) < n:
                raise ValueError(f"Not enough paragraphs available. Required: {n}, found: {len(valid_indices)}.")

        selected_paragraphs = random.sample(valid_indices, k=n)
        selected_sentences = random.sample(self.sentences, k=n)

        affected_ranges = []
        for i, par_idx in enumerate(selected_paragraphs):
            paragraphs[par_idx] += f" {selected_sentences[i]}"
            end = self.char_ranges[par_idx][1]
            affected_ranges.append([end, end])

        updated_text = self._reconstruct_text(paragraphs, delimiter)

        return updated_text, affected_ranges