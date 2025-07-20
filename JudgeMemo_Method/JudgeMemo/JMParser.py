import os
import re
from typing import Dict, List, Union
import json
import pandas as pd

from JudgeMemo import JMUtils


class JMParser:
    def __init__(self, doc_id):
        self.doc_id = doc_id

    @staticmethod
    def _clean_text(text) -> str:
        # Replace tabs and other whitespace (except newlines) with spaces
        # Collapse multiple spaces into one
        # Retain newlines to preserve structure
        text = re.sub(r'[ \t\r\f\v]+', ' ', text)  # collapse spaces/tabs
        text = re.sub(r' *\n *', '\n', text)  # strip spaces around newlines
        return text.strip()

    @staticmethod
    def _preprocess_text(text) -> str:
        """
        If both <think> and </think> tags are present, removes all such blocks including the tags.
        If either tag is missing, returns an empty string (text is considered unusable).
        """
        if ("<think>" in text) == ("</think>" in text):  # both or neither
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return ""

    @staticmethod
    def _extract_issues_block(text: str, label: str) -> str:
        """
        Attempts multiple regex patterns to extract the issues block for a given label.
        Returns the first successful match or an empty string if none match.
        """
        patterns = [
            # Pattern 1: Inline issues separated by spaces, numbered sections use ')' or '.' after number
            rf"(?:\*\*)?{label} Issues(?:\*\*)?:\s*(.*?)(?=\s+\d+[.)]\s|\n\s*FINAL|\Z)",

            # Pattern 2: Multiline issues until next numbered section or FINAL
            rf"(?:\*\*)?{label} Issues(?:\*\*)?:\s*((?:.|\n)*?)(?=\n\s*\d+[.)]\s|\n\s*FINAL|\Z)",

            # Pattern 3: Markdown bold issues block (optional)
            rf"\*\*{label} Issues\*\*:\s*(.*?)(?=\n\s*\d+[.)]\s|\n\s*FINAL|\Z)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_issues_block_variants(text: str, label: str) -> List[str]:
        """
        Extracts blocks of issues for a given label,
        splitting by every '- [TAG]' occurrence, even if issues are on the same line.
        """
        pattern = rf"(?:\d+\)\s*)?{label} Issues?:\s*(.*?)(?=\n\d+\)|\nFINAL|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not match:
            return []

        block = match.group(1).strip()

        # Split by '- [TAG]' and '* [TAG]', but keep the tags at start of each chunk by splitting with lookahead
        issues = re.split(r'(?=[*-] \[[A-Z]+])', block)

        # Clean and filter empty strings
        return [issue.strip() for issue in issues if issue.strip()]

    @staticmethod
    def _extract_issues(issue_chunks: List[str]) -> Dict[str, List[str]]:
        issues = {}
        for chunk in issue_chunks:
            match = re.match(r'[*-] \[([A-Z]+)]\s*(.*)', chunk, flags=re.DOTALL)
            if match:
                tag = f"[{match.group(1)}]"
                desc = match.group(2).replace('\n', ' ').strip()
                if desc:
                    issues.setdefault(tag, []).append(desc)
        return issues

    @staticmethod
    def parse_final_scores(text: str, filename: str = "", json_active: bool = True):
        """
        Extracts final coherence and fluency scores in both markdown and plain text forms,
        allowing for variations in formatting (e.g., bolding, spacing, casing).
        """

        def _extract_score(label: str) -> Union[float, None]:
            # All candidate patterns
            patterns = [
                # 1. Bold entire sentence including score, optional numbering with dot or parenthesis:
                # Example: "3) **FINAL Coherence Score: 4.5**"
                rf"(?:^\s*\d+[.)]\s*)?\*\*FINAL\s+{label}\s+Score\s*[:：]\s*([\d.]+)\*\*",

                # 2. Bold label only, score outside bold, optional numbering:
                # Example: "3) **FINAL Coherence Score:** 4.5"
                rf"(?:^\s*\d+[.)]\s*)?\*\*FINAL\s+{label}\s+Score[:：]\*\*\s*([\d.]+)",

                # 3. Flexible pattern allowing optional numbering and optional bolding around label and/or score:
                # Example: "3) FINAL Coherence Score: 4.5" or "**FINAL Coherence Score:** **4.5**"
                rf"(?:^\s*\d+[.)]\s*)?(?:\*\*)?FINAL\s+{label}\s+Score(?:\*\*)?\s*[:：]\s*(?:\*\*)?([\d.]+)(?:\*\*)?",

                # 4. Pattern dropping the word "FINAL", allowing optional bolding around label and score:
                # Example: "Coherence Score: 4.5" or "**Coherence Score:** **4.5**"
                rf"(?:^\s*\d+[.)]\s*)?(?:\*\*)?{label}\s+Score(?:\*\*)?\s*[:：]\s*(?:\*\*)?([\d.]+)(?:\*\*)?",

                # 5. Double bold around label and score:
                # Example: "**FINAL Coherence Score:** **4.5**"
                rf"(?:^\s*\d+[.)]\s*)?\*\*FINAL\s+{label}\s+Score:\*\*\s*\*\*([\d.]+)\*\*",

                # 6. Plain text with optional numbering (dot or parenthesis):
                # Example: "3) FINAL Coherence Score: 4.5"
                rf"(?:^\s*\d+[.)]\s*)?FINAL\s+{label}\s+Score\s*[:：]\s*([\d.]+)",

                # 7. Markdown bold around label only, followed by colon and score:
                # Example: "**Coherence**: 4.5"
                rf"\*\*{label}\*\*\s*[:：]\s*([\d.]+)",

                # 8. Numbered line, bold label with "Score", colon, then score:
                # Example: "3. **COHERENCE Score:** 4"
                rf"(?:^\s*\d+[.)]\s*)?\*\*{label}\s+Score\*\*\s*[:：]\s*([\d.]+)",

                # 9. Label bolded on one line, score bolded on next line:
                # Example: "**COHERENCE**\n**4**"
                rf"\*\*{label}\*\*\s*\n\s*\*\*?([\d.]+)\*\*?",

                # 10. Plain text label with colon followed by score:
                # Example: "COHERENCE: 4"
                rf"{label}\s*[:：]\s*([\d.]+)",

                # 11. Score enclosed in brackets, optional numbering:
                # Example: "3) FINAL Coherence Score: [3]"
                rf"(?:^\s*\d+[.)]\s*)?FINAL\s+{label}\s+Score\s*[:：]\s*\[([\d.]+)\]",
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip().rstrip('.,;:!?')
                    return float(value)
            return None

        scores = {
            "coherence": _extract_score("Coherence"),
            "fluency": _extract_score("Fluency")
        }

        if json_active:
            return {k: v for k, v in scores.items() if v is not None}
        else:
            base_filename = filename.split('/')[-1]
            column_name = os.path.splitext(base_filename)[0]
            df = pd.DataFrame(scores, index=[column_name]).T
            df.columns = [column_name]
            return df

    def parse_scores_and_issues(self, text: str) -> Dict[str, Union[Dict[str, float], Dict[str, List[str]]]]:
        scores = self.parse_final_scores(text)

        fluency_chunks = self._extract_issues_block_variants(text, "Fluency")
        coherence_chunks = self._extract_issues_block_variants(text, "Coherence")

        issues = {}
        if fluency_chunks:
            issues["fluency"] = self._extract_issues(fluency_chunks)
        if coherence_chunks:
            issues["coherence"] = self._extract_issues(coherence_chunks)

        return {
            "scores": scores,
            "issues": issues
        }

    def parse(self,
              input_data: str,
              is_raw_text: bool = False,
              scores_only: bool = False
              ):
        """
        Parses final scores from either a file or raw string, and saves the result to a JSON file.

        Args:
            input_data (str): Path to a file or raw text content.
            is_raw_text (bool): Whether the input_data is raw text instead of a file path.
            scores_only (bool): Whether only scores or scores and issues shall be extracted from the input_data.
            """
        # Use the raw string or read from file
        text = input_data if is_raw_text else JMUtils.read_file(input_data)
        text = self._clean_text(text)
        text = self._preprocess_text(text)

        if scores_only:
            # Parse only the scores
            scores = self.parse_final_scores(text, input_data, False)
        else:
            # Parse only the scores
            scores = self.parse_scores_and_issues(text)

        print(f"Successfully parsed {self.doc_id}.")
        return scores


if __name__ == "__main__":
    text = """
Evaluation Form:  
1) Fluency Issues:  
- [SYNTAX] Occasionally complex, dense sentences (e.g., "melted it in the crucible of his own thought").  
- [LEXICON] Rare archaic/photic phrases (e.g., "bedizened," "battened").  

2) Coherence Issues:  
- [STRUCTURE] Abrupt shift from Gypsy narrative (Chapter 1) to unrelated art history (Chapter 2).  
- [TRANSITION] Unexplained tangent about François Boucher in Chapter 2.  
- [LOGIC] Disjointed juxtaposition of Liverpool Gypsies and Reynolds’ biography.  

3) FINAL Coherence Score: **3**  

4) FINAL Fluency Score: **4.5**
        """
    parser = JMParser(doc_id="123_test")

    scores = parser.parse(text, is_raw_text=True, scores_only=False)
    print(json.dumps(scores, indent=2))
