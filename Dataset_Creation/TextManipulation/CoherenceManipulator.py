from Dataset_Creation.TextManipulation.manipulations.C_logical_flow_disruptions import (
    SwapContent, RemoveContent, InsertContent, RepeatContent, ExchangeContent
)
from Dataset_Creation.TextManipulation.manipulations.C_plot_inconsistencies import (
    ExchangeEntities, AddTempInconsistencies, SwapEntities
)
from Dataset_Creation.TextManipulation.manipulations.C_anaphora_resolution import (EntityPronounExchanger, EntityTermExchanger)
from Dataset_Creation.TextManipulation.exceptions import NoChapterException


class CoherenceManipulator:
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
            category (str): One of 'logical_flow_disruptions' or 'plot_inconsistencies'.
            manipulation_type (str): The specific manipulation type.
            **kwargs: Arguments required for the specific manipulation.

        Returns:
            str: The updated text.
        """
        try:
            if category == "logical_flow_disruptions":
                manipulator = self._get_logical_flow_manipulator(manipulation_type)
            elif category == "plot_inconsistencies":
                manipulator = self._get_plot_inconsistencies_manipulator(manipulation_type)
            elif category == "anaphora_resolution":
                manipulator = self._get_anaphora_resolution_manipulator(manipulation_type)
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

    def _get_logical_flow_manipulator(self, manipulation_type):
        """Get the class for logical flow manipulations."""
        mapping = {
            "swap_content": SwapContent,
            "remove_content": RemoveContent,
            "insert_content": InsertContent,
            "repeat_content": RepeatContent,
            "exchange_content": ExchangeContent
        }

        if manipulation_type not in mapping:
            raise ValueError(f"Unsupported logical flow manipulation: {manipulation_type}")

        # returns initialized object of manipulation type class to apply_manipulation()
        return mapping[manipulation_type](self.text)

    def _get_plot_inconsistencies_manipulator(self, manipulation_type):
        """Get the class for plot inconsistencies manipulations."""
        mapping = {
            "exchange_entities": ExchangeEntities,
            "swap_entities": SwapEntities,
            "temporal_inconsistencies": AddTempInconsistencies
        }
        if manipulation_type not in mapping:
            raise ValueError(f"Unsupported plot inconsistency manipulation: {manipulation_type}")
        return mapping[manipulation_type](self.text)

    def _get_anaphora_resolution_manipulator(self, manipulation_type):
        mapping = {
            "exchange_entities_w_pronouns": EntityPronounExchanger,
            "exchange_entities_w_term": EntityTermExchanger
        }
        if manipulation_type not in mapping:
            raise ValueError(f"Unsupported anaphora resolution manipulation: {manipulation_type}")
        return mapping[manipulation_type](self.text)
