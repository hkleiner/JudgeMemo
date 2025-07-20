from Dataset_Creation.TextManipulation.manipulations.C_logical_flow_disruptions import (SwapContent, RemoveContent, InsertContent,
                                                                                        RepeatContent, ExchangeContent)
from Dataset_Creation.TextManipulation.manipulations.C_anaphora_resolution import EntityPronounExchanger, EntityTermExchanger
from Dataset_Creation.TextManipulation.manipulations.C_plot_inconsistencies import AddTempInconsistencies, ExchangeEntities, SwapEntities
from Dataset_Creation.TextManipulation.manipulations.F_grammatical_errors import (TypoInserter, VerbTenseChanger,
                                                                                  WordOrderSwapper, PunctuationRemover, WordRemover)
from Dataset_Creation.TextManipulation.manipulations.F_unnatural_phrasing import AwkwardPassivePhraser

text = (
    "TITLE\n"
    "\n\nCHAPTER: II\n\n\n"
    "(1) Moreover, this is a paragraph. Maria and Kim enjoy eating ice cream. Apple is a company.\n"
    "Additionally, it is quite long and takes two lines.\n\n"
    "(2) In consequence, this is another paragraph.\n"
    "John Miller and Harry Potter went to the market. John bought her apples. Mary bought oranges.\n\n"
    "(3) The third paragraph is longer. It is also quite long.\n"
    "However, the sky is blue. Blue like the bottle on the left. Random content.\n"
    "Furthermore, we like to introduce the color yellow.\n"
    "\n\nCHAPTER: III\n\n\n"
    "(4) To sum up, it was a long trip to the Pacific. John and Harry Potter went to bed in the hotel early. We really enjoyed it.\n"
    "Moreover, the sun was out all the time and the ocean was quite warm.\n\n"
    "(5) In conclusion, the Netherlands are beautiful.\n"
    "We went to the school today."
)
# text = """Emma Johnson, a senior architect at Zaha Hadid Architects, flew from New York to Berlin last week to attend a design summit at the historic Tempelhof Airport. While in Germany, she visited the Rhine River, toured the Deutsche Bank headquarters, and gave a lecture at MIT’s European Innovation Hub. Emma later met with Carlos Ramirez, an urban planner from Chile, at an Alexanderplatz, before both traveled to the Swiss Alps for a site inspection near the Matterhorn."""
text = """After landing at Heathrow Airport, the delegation from Stanford University traveled to London for the Global Innovation Summit hosted by TechBridge International. The event, held inside the historic Royal Albert Hall, attracted leaders from NATO, UNESCO, and several startups based in Silicon Valley. On the second day, attendees visited the Tower of London and took a boat ride along the Thames River before heading to meetings at the European Commission headquarters in Brussels. Representatives from Tokyo, New York, and Dubai presented case studies on urban development. In the final session, held at the World Trade Center, a panel discussed infrastructure resilience in Jakarta, Cape Town, and São Paulo, highlighting lessons learned after recent flooding near Mount Elgon and the Panama Canal."""

insert_text = (
    # "\n\nCHAPTER: INSERT II\n\n\n"
    "INSERT: (1) This is a paragraph. Bla bla bla. Text Text Text.\n"
    "It is quite long and takes two lines.\n\n"
    "INSERT: (2) This is another paragraph.\n\n"
    "INSERT: (3) The third paragraph is longer. It is also quite long.\n"
    "However, the sky is blue. Blue like the bottle on the left. Random content.\n"
    "Furthermore, we like to introduce the color yellow.\n"
    # "\n\nCHAPTER: INSERT II\n\n\n"
    "INSERT: (4) To sum up, it was a long trip. We really enjoyed it.\n"
    "Moreover, the sun was out all the time.\n\n"
    "INSERT: (5) Short."
)


def swap_content(range_type):
    manipulator = SwapContent(text)
    updated_text, affected_ranges = manipulator.execute(range_type=range_type, min_len=20, n=2)

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)
    print("Char_ranges: ", manipulator.char_ranges)


def remove_content(range_type):
    manipulator = RemoveContent(text)
    updated_text, affected_ranges = manipulator.execute(range_type=range_type, min_len=50, n=2)

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)
    print("Char_ranges: ", manipulator.char_ranges)


def insert_content(range_type):
    manipulator = InsertContent(text=text)
    updated_text, affected_ranges = manipulator.execute(range_type=range_type, min_len=60, n=3, text_to_insert=insert_text)

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)
    print("Char_ranges: ", manipulator.char_ranges)


def repeat_content(range_type):
    manipulator = RepeatContent(text=text)
    updated_text, affected_ranges = manipulator.execute(range_type=range_type, min_len=80, n=2, repetition_factor=1)

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)
    print("Char_ranges: ", manipulator.char_ranges)


def exchange_content(range_type):
    manipulator = ExchangeContent(text=text)
    updated_text, affected_ranges = manipulator.execute(range_type=range_type, min_len=20, texts_to_insert=[insert_text for _ in range(20)], ratio=True)

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)
    print("Char_ranges: ", manipulator.char_ranges)


def typos():
    manipulator = TypoInserter(text)
    updated_text, affected_ranges = manipulator.execute(start=10, end=10, dense=False)

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)


def verb_tenses():
    manipulator = VerbTenseChanger(text)
    updated_text, affected_ranges = manipulator.execute(start=25, end=25, dense=True)

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)


def word_order():
    manipulator = WordOrderSwapper(text)
    updated_text, affected_ranges = manipulator.execute(start=33, end=33, dense=False)

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)


def punctuation():
    manipulator = PunctuationRemover(text)
    updated_text, affected_ranges = manipulator.execute(start=10, end=10)

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)


def word_removal():
    manipulator = WordRemover(text)
    updated_text, affected_ranges = manipulator.execute(start=5, end=5)

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)


def entity_pronoun_exchange():
    manipulator = EntityPronounExchanger(text)
    updated_text, affected_ranges = manipulator.execute(entities_to_consider=["LOC", "GPE", "FAC", "ORG"], start=60, end=60)

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)


def entity_term_exchange():
    manipulator = EntityTermExchanger(text)
    updated_text, affected_ranges = manipulator.execute(
        entities_to_consider=["LOC", "GPE", "FAC", "ORG"],
        start=60,
        end=60,
        replacement_terms=['stuff', 'thing']
    )

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)


def add_temporal_inconsistencies():
    manipulator = AddTempInconsistencies(text)
    updated_text, affected_ranges = manipulator.execute()

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)


def exchange_entities():
    manipulator = ExchangeEntities(text)
    updated_text, affected_ranges = manipulator.execute(entity="PERSON", start=50, end=50)

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)


def entity_swaps():
    manipulator = SwapEntities(text)
    updated_text, affected_ranges = manipulator.execute(entity="PERSON", start=60, end=60)

    print("Original Text:")
    print(text)
    print("---------------------------------")
    print("Updated Text:")
    print(updated_text)
    print("\nAffected Character Ranges:")
    print(affected_ranges)


def awkward_phrasing():
    text = "The cat chased the mouse. She painted the wall. They are reading the book."
    converter = AwkwardPassivePhraser(text)
    print(converter.execute())


if __name__ == "__main__":
    # swap_content("section")
    # remove_content("section")
    # insert_content("section")
    # repeat_content("section")
    # remove_transitions('paragraph')
    # typos()
    # verb_tenses()
    # word_order()
    # punctuation()
    # word_removal()
    # entity_pronoun_exchange()
    # entity_term_exchange()
    # add_temporal_inconsistencies()
    # exchange_content("paragraph")
    exchange_entities()
    # entity_swaps()
    # awkward_phrasing()
