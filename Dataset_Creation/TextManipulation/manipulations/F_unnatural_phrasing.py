import spacy
from Dataset_Creation.TextManipulation.ContentManipulator import ContentManipulator

"""
\textit{Unnatural Phrasing} refers to manipulations that preserve grammaticality but violate expected patterns of natural, fluent language. These include awkward constructions, unusual collocations, non-idiomatic expressions, or overly literal translations that sound ``unnatural'' to a fluent speaker.
"""

class AwkwardPassivePhraser(ContentManipulator):
    def __init__(self, text):
        super().__init__(text)
        self.nlp = spacy.load("en_core_web_sm")
        self.subject_to_object_pronouns = {
            "i": "me",
            "you": "you",
            "he": "him",
            "she": "her",
            "it": "it",
            "we": "us",
            "they": "them"
        }

    def execute(self):
        doc = self.nlp(self.text)
        converted_sentences = []

        for sent in doc.sents:
            passive = self._convert_sentence(sent)
            converted_sentences.append(passive)

        return ' '.join(converted_sentences)

    def _convert_sentence(self, sent):
        subj = None
        verb = None
        dobj = None

        for token in sent:
            if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
                subj = token
                verb = token.head
            elif token.dep_ == 'dobj':
                dobj = token

        if subj and verb and dobj:
            aux_verb = "was" if dobj.tag_ in ("NN", "NNP") else "were"

            # Use proper past participle
            past_participle = verb.lemma_ + "ed" if verb.tag_ not in ("VBN",) else verb.text

            # Convert subject pronoun to object case if applicable
            subj_lower = subj.text.lower()
            agent = self.subject_to_object_pronouns.get(subj_lower, subj_lower)

            by_phrase = f"by {agent}"

            rest = [token.text for token in sent if token not in (subj, verb, dobj)]
            rest_str = ' '.join(rest).strip()

            passive_sent = f"{dobj.text} {aux_verb} {past_participle} {by_phrase}"
            if rest_str:
                passive_sent += " " + rest_str

            return passive_sent[0].upper() + passive_sent[1:] + "."
        else:
            return sent.text
