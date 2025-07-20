from Dataset_Creation.TextManipulation.exceptions import NoChapterException
from Dataset_Creation.TextManipulation.manipulations.F_grammatical_errors import TypoInserter, VerbTenseChanger, WordOrderSwapper, PunctuationRemover, WordRemover


class FluencyManipulator:
    def __init__(self, text, idx):
        """
        Initialize the CoherenceManipulator with the input raw text.
        """
        self.text = text
        self.idx = idx

    def apply_manipulation(self, category, manipulation_type, **kwargs):
        """
        Apply a manipulation to the text.

        Args:
            category (str): One of 'anaphora_resolution', 'grammatical_errors' or 'unnatural_phrasing'.
            manipulation_type (str): The specific manipulation type.
            **kwargs: Arguments required for the specific manipulation.

        Returns:
            str: The updated text.
        """
        """
        ADD: manipulation_types.json
            "unnatural_phrasing": [
              "word_repetition"
            ],
            "anaphora_resolution":[
              "exchange_entities_w_pronouns"
            ]
        """
        try:
            if category == "grammatical_errors":
                manipulator = self._get_grammatical_errors_manipulator(manipulation_type)
            #elif category == 'unnatural_phrasing':
                #manipulator = WordRepetition(self.text)
            else:
                raise ValueError(f"Unsupported category: {category}")

            # execute the manipulation (calls function in corresponding class) and update the text
            text, affected_ranges = manipulator.execute(**kwargs)
            return text, affected_ranges
        except NoChapterException as e:
            print(f"NoChapterException caught for {self.idx}: {e}")
            return None, None
        except ValueError as v:
            print(f"ValueError caught for {self.idx}: {v}")
            return None, None

    def _get_grammatical_errors_manipulator(self, manipulation_type):
        """Get the class for logical flow manipulations."""
        mapping = {
            "typos": TypoInserter,
            "verb_tenses": VerbTenseChanger,
            "word_order": WordOrderSwapper,
            "punctuation_removal": PunctuationRemover,
            "word_removal": WordRemover
        }

        if manipulation_type not in mapping:
            raise ValueError(f"Unsupported grammatical error manipulation: {manipulation_type}")

        # returns initialized object of manipulation type class to apply_manipulation()
        return mapping[manipulation_type](self.text)
