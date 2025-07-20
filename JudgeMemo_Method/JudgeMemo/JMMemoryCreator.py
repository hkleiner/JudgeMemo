from JudgeMemo import JMUtils
from JudgeMemo.JMParser import JMParser


class JMMemoryCreator:
    """
    JMMemoryCreator processes evaluated document sections to build a structured memory representation
    of issues and scores, and can generate a formatted report summarizing these evaluations.

    Attributes:
        doc_id (str): Identifier of the document being processed.
        parser (JMParser): Parser instance to extract structured info from evaluation text.
        memory (dict): Stores the structured evaluation memory for all sections.

    Methods:
        create_memory(scanner_results, memory_file, is_raw_text=False):
            Processes evaluation results from scanned sections, converts them to structured memory,
            and optionally saves the memory to a JSON file.

        _create_memory_entry(sec_eval, sec_nr, sec_summary, sec_start, sec_end, is_raw_text=False) -> dict:
            Parses a single section's evaluation text and builds a memory entry dictionary.

        _build_memory_dict(section_key, issues_and_scores, section_start_char_incl,
                           section_end_char_excl, section_summary) -> dict:
            Builds the dictionary structure for one section's memory entry including issues, scores, and metadata.

        create_report(report_file, sec2tag=True) -> str:
            Generates a textual evaluation report from the stored memory.
            Supports two modes:
                - sec2tag (True): Reports by section, listing scores and categorized issues by tag.
                - sec2tag (False): Reports by tag, listing all sections that have issues with that tag,
                                   followed by section scores summary.
            Optionally saves the report to a file.
    """
    def __init__(self, doc_id):
        """
        Initializes JMMemoryCreator with a document identifier and sets up parser and memory container.

        Args:
            doc_id (str): Document identifier for which memory is being created.
        """
        self.doc_id = doc_id
        self.parser = JMParser(doc_id=doc_id)
        self.memory = dict()

    def create_memory(self, 
                      scanner_results, 
                      memory_file, 
                      is_raw_text: bool = False
                      ):
        """
        Processes a list of scan results to build and store the memory structure.
        Optionally saves the memory dictionary as a JSON file.

        Args:
            scanner_results (list): List of section scan objects containing evaluation data.
            memory_file (str): Path to save the memory JSON file. If empty or None, memory is not saved.
            is_raw_text (bool): Whether the evaluation text is raw or already parsed (default False).
        """
        for scan in scanner_results:
            entry = self._create_memory_entry(
                sec_eval=scan.sec_evaluation,
                sec_nr=scan.sec_nr,
                sec_summary=scan.sec_summary,
                sec_start=scan.sec_start,
                sec_end=scan.sec_end,
                is_raw_text=is_raw_text
            )
            self.memory.update(entry)

        print(f"Successfully created memory for document {self.doc_id}: {memory_file}")
        if memory_file:  # save to file
            JMUtils.save_to_json(self.memory, memory_file)

    def _create_memory_entry(self,
                             sec_eval: str,
                             sec_nr: str,
                             sec_summary: str,
                             sec_start: int,
                             sec_end: int,
                             is_raw_text: bool = False
                             ) -> dict:
        """
        Parses a single section's evaluation text and creates a memory entry dictionary
        including issues and scores.

        Args:
            sec_eval (str): The evaluation text of the section.
            sec_nr (str): Section number or key.
            sec_summary (str): Summary text of the section.
            sec_start (int): Start character index of the section in the document (inclusive).
            sec_end (int): End character index of the section in the document (exclusive).
            is_raw_text (bool): Whether the evaluation text is raw or already parsed (default False).

        Returns:
            dict: Memory entry for the section keyed by section number.
        """
        sec_scores_issues = self.parser.parse(input_data=sec_eval, is_raw_text=is_raw_text, scores_only=False)

        return self._build_memory_dict(
            section_key=sec_nr,
            issues_and_scores=sec_scores_issues,
            section_start_char_incl=sec_start,
            section_end_char_excl=sec_end,
            section_summary=sec_summary
        )

    @staticmethod
    def _build_memory_dict(section_key: str,
                           issues_and_scores: dict,
                           section_start_char_incl: int,
                           section_end_char_excl: int,
                           section_summary: str) -> dict:
        """
        Constructs the structured memory dictionary for a section given its parsed issues and scores.

        Args:
            section_key (str): Identifier/key for the section.
            issues_and_scores (dict): Parsed dictionary with keys "issues" and "scores".
            section_start_char_incl (int): Inclusive start character index of section.
            section_end_char_excl (int): Exclusive end character index of section.
            section_summary (str): Summary text for the section.

        Returns:
            dict: A dictionary mapping section_key to its detailed memory entry.
        """
        raw_issues = issues_and_scores.get("issues", {})
        raw_scores = issues_and_scores.get("scores", {})

        structured_issues = {
            "fluency": raw_issues.get("fluency", {}),
            "coherence": raw_issues.get("coherence", {})
        }
        structured_scores = {
            "fluency": raw_scores.get("fluency", 0),
            "coherence": raw_scores.get("coherence", 0)
        }

        return {
            section_key: {
                "issues": structured_issues,
                "scores": structured_scores,
                "section_start_char_incl": section_start_char_incl,
                "section_end_char_excl": section_end_char_excl,
                "section_summary": section_summary
            }
        }

    def create_report(self,
                      report_file: str,
                      sec2tag: bool = True
                      ) -> str:
        """
        Generates a human-readable report summarizing issues and scores stored in memory.
        Two modes are supported:
          - sec2tag=True: The report is organized by section, with issues categorized by metric and tag.
          - sec2tag=False: The report is organized by tags, listing all sections with that issue tag,
                           followed by a summary of scores per section.

        Args:
            report_file (str): Path to save the generated report text file.
                               If empty or None, report is not saved.
            sec2tag (bool): Whether to generate the report by section (True) or by tag (False).

        Raises:
            ValueError: If the memory is empty when attempting to create a report.

        Returns:
            str: The full textual report.
        """
        if not self.memory:
            raise ValueError("Memory is empty. Please run `create_memory` first.")

        report_lines = []
        report_lines.append("=== Evaluation Report ===\n")

        if sec2tag:
            for sec, data in self.memory.items():
                report_lines.append(
                    f"Section {sec} (Chars {data['section_start_char_incl']}–{data['section_end_char_excl']}):")
                if data['section_summary']:
                    report_lines.append(f"Section Summary: {data['section_summary']}")
                report_lines.append("Scores:")
                report_lines.append(f"  - Fluency: {data['scores']['fluency']}")
                report_lines.append(f"  - Coherence: {data['scores']['coherence']}")
                report_lines.append("Issues:")

                for metric in ['fluency', 'coherence']:
                    issues = data["issues"].get(metric, {})
                    if issues:
                        report_lines.append(f"  {metric.capitalize()} Issues:")
                        for tag, descriptions in issues.items():
                            report_lines.append(f"    - {tag}:")
                            for desc in descriptions:
                                report_lines.append(f"        • {desc}")
                    else:
                        report_lines.append(f"  {metric.capitalize()} Issues: None")

                report_lines.append("")  # blank line between sections

        else:
            # tag2seq mode

            # aggregate tags
            tag2sections = {
                "fluency": {},
                "coherence": {}
            }
            section_scores = {}

            for sec, data in self.memory.items():
                section_scores[sec] = data["scores"]
                for metric in ['fluency', 'coherence']:
                    for tag, descriptions in data["issues"].get(metric, {}).items():
                        if tag not in tag2sections[metric]:
                            tag2sections[metric][tag] = []
                        tag2sections[metric][tag].append((sec, descriptions))

            for metric in ['fluency', 'coherence']:
                report_lines.append(f"{metric.capitalize()} Tags:\n")
                for tag, occurrences in tag2sections[metric].items():
                    report_lines.append(f"  - {tag}:")
                    for sec, descriptions in occurrences:
                        report_lines.append(f"      Section {sec}:")
                        for desc in descriptions:
                            report_lines.append(f"        • {desc}")
                report_lines.append("")  # blank line

            report_lines.append("=== Section Scores ===\n")
            for sec, scores in section_scores.items():
                report_lines.append(f"Section {sec}: Fluency = {scores['fluency']}, Coherence = {scores['coherence']}")

        report = "\n".join(report_lines)
        
        if report_file:
            JMUtils.save_to_text(report, report_file)
            print(f"Report saved to {report_file}")

        return report
