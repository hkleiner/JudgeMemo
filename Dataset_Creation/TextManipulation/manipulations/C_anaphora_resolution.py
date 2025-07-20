import random
import spacy
from Dataset_Creation.TextManipulation.ContentManipulator import ContentManipulator


class EntityPronounExchanger(ContentManipulator):
    """
    A text manipulation class that replaces named entities in a document with appropriate pronouns.

    This class uses spaCy's named entity recognition (NER) to identify specific types of entities
    (e.g. ORG, GPE, LOC) and substitutes a fixed number of them with predefined pronouns
    such as "it". The pronouns will be capitalized if the entity is at the start of a sentence.

    This class does not support PERSON entities yet!
    """
    def __init__(self, text):
        super().__init__(text)
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.PRONOUNS = {
            "ORG": ["it"],     # for organizations (lower)
            "GPE": ["it"],     # for geopolitical entities (countries, cities) (lower)
            "LOC": ["it"],     # for locations (lower)
            "FAC": ["it"]      # for buildings, bridges, etc.
        }

    def execute(self, entities_to_consider: list = None, start: int = 50, end: int = 50) -> tuple[str, list[list[int]]]:
        """
        Replaces a random subset of named entities in the text with pronouns and returns the modified text
        along with the character spans of the affected changes.

        Parameters:
            entities_to_consider (list, optional): A list of entity types (e.g., ["LOC", "ORG"]) to be considered for replacement.
                                                   Defaults to ["LOC", "GPE", "FAC", "ORG"].
            start (int, optional): Lower bound for the percentage of entities to replace. Defaults to 50.
            end (int, optional): Upper bound for the percentage of entities to replace. Defaults to 50.

        Returns:
            tuple[str, list[list[int]]]:
                - Modified text with selected entities replaced by pronouns.
                - A list of character spans [[start_idx, end_idx], ...] indicating where replacements occurred.

        Raises:
            ValueError: If an unsupported entity type is provided.
        """
        if entities_to_consider is None:
            entities_to_consider = ["LOC", "GPE", "FAC", "ORG"]

        for ent_type in entities_to_consider:
            if ent_type not in self.PRONOUNS.keys():
                raise ValueError(f"Unsupported entity type {ent_type}! Only supports: {self.PRONOUNS.keys()}")

        doc = self.spacy_nlp(self.text)

        entities = [ent
                    for ent in doc.ents
                    if ent.label_ in entities_to_consider and ent.text != "CHAPTER"]  # get multi-token entities

        percentage = random.randint(start, end)  # get ratio
        n = round((len(entities) * percentage) / 100)  # get how many entity mentions are affected
        print(f"Number of selected entities {n} out of {len(entities)} detected entities (percentage: {percentage}%)")

        ents_to_change = random.sample(entities, n)

        updated_text = ""
        last_idx = 0
        affected_ranges = []
        for ent in doc.ents:
            if ent in ents_to_change:
                # add text before the entity
                updated_text += self.text[last_idx:ent.start_char]
                pronoun = random.choice(self.PRONOUNS[ent.label_])  # get pronoun
                if ent[0].is_sent_start:
                    pronoun = pronoun.capitalize()
                updated_text += pronoun
                affected_ranges.append([ent.start_char, ent.end_char])
                last_idx = ent.end_char  # set new last index

        updated_text += self.text[last_idx:]

        return updated_text, affected_ranges


class EntityTermExchanger(ContentManipulator):
    """
    A text manipulation class that replaces named entities in a document with a randomly chosen replacement term from a
    given list of replacement terms, e.g. 'thing', 'stuff' ...

    This class uses spaCy's named entity recognition (NER) to identify specific types of entities
    (e.g. ORG, GPE, LOC) and substitutes a fixed number of them with the replacement term. The term
    will be capitalized if the entity is at the start of a sentence. It will also be adapted to plural entities if
    necessary.

    This class can be used for PERSON entities as well, if explicitly given in 'entities_to_change'!
    """
    def __init__(self, text):
        super().__init__(text)
        self.spacy_nlp = spacy.load("en_core_web_sm")

    def execute(self, entities_to_consider: list = None, start: int = 50, end: int = 50, replacement_terms: list = None) -> tuple[str, list[list[int]]]:
        """
        Replaces a random subset of named entities in the text with the term 'thing' and returns the modified text
        along with the character spans of the affected changes.

        Parameters:
            entities_to_consider (list, optional): A list of entity types (e.g., ["LOC", "ORG"]) to be considered for replacement.
                                                   Defaults to ["LOC", "GPE", "FAC", "ORG"].
            start (int, optional): Lower bound for the percentage of entities to replace. Defaults to 50.
            end (int, optional): Upper bound for the percentage of entities to replace. Defaults to 50.
            replacement_terms (list, optional): A list of terms to replace the entity with. Defaults to ['thing'].

        Returns:
            tuple[str, list[list[int]]]:
                - Modified text with selected entities replaced by 'thing'.
                - A list of character spans [[start_idx, end_idx], ...] indicating where replacements occurred.
        """
        if entities_to_consider is None:
            entities_to_consider = ["LOC", "GPE", "FAC", "ORG"]

        if replacement_terms is None:
            replacement_terms = ["thing"]

        doc = self.spacy_nlp(self.text)

        entities = [ent
                    for ent in doc.ents
                    if ent.label_ in entities_to_consider and ent.text != "CHAPTER"]  # get single-token entities
        print(f"Detected entities: {entities}")

        percentage = random.randint(start, end)  # get ratio
        n = round((len(entities) * percentage) / 100)  # get how many entity mentions are affected
        print(f"Number of selected entities {n} out of {len(entities)} detected entities (percentage: {percentage}%)")

        ents_to_change = random.sample(entities, n)

        updated_text = ""
        last_idx = 0
        affected_ranges = []

        for ent in doc.ents:
            if ent in ents_to_change:
                replacement = random.choice(replacement_terms)  # determine replacement term
                needs_article = True

                # check if the token before the entity is an article
                for i in range(max(0, ent.start - 2), ent.start):
                    prev_token = doc[i]
                    if replacement[0].lower() in {"a", "e" "i", "o", "u"}:  # starts with vowel
                        if prev_token.lower_ in {"an", "the"}:  # fine
                            needs_article = False
                            break
                        elif prev_token.lower_ == "a":  # replace with 'an'
                            needs_article = False
                            updated_text += self.text[last_idx:i] + "an"  # add everything before this entity
                            last_idx = i + 1
                            break
                    else:  # doe snot start with vowel
                        if prev_token.lower_ in {"a", "the"}:  # fine
                            needs_article = False
                            break
                        elif prev_token.lower_ == "an":  # replace with 'a'
                            needs_article = False
                            updated_text += self.text[last_idx:i] + "a"  # add everything before this entity
                            last_idx = i+1
                            break

                updated_text += self.text[last_idx:ent.start_char]  # add everything before this entity

                # compose replacement phrase
                if needs_article:
                    # use 'an' if replacement starts with a vowel
                    article = "an" if replacement[0].lower() in {"a", "e", "i", "o", "u"} else random.choice(["a", "the"])
                    replacement = f"{article} {replacement}"

                    if ent[0].is_sent_start:  # capitalize if at sentence start
                        replacement = replacement.capitalize()
                else:
                    replacement = replacement

                updated_text += replacement  # add replacement and store affected range
                affected_ranges.append([ent.start_char, ent.end_char])
                last_idx = ent.end_char  # move pointer forward

        updated_text += self.text[last_idx:]  # add the remaining tail of the text

        return updated_text, affected_ranges
