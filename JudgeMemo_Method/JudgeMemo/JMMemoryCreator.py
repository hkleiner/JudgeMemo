from JudgeMemo import JMUtils
from JudgeMemo.JMParser import JMParser


class JMMemoryCreator:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.parser = JMParser(doc_id=doc_id)
        self.memory = dict()

    def create_memory(self, 
                      scanner_results, 
                      memory_file, 
                      is_raw_text: bool = False
                      ):
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
                             ):
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
                      ):
        # TODO: build the final report from the before created memory that contains all information about the in sections
        #  evaluated document and its issues
        # the report is given to the model as part of the user prompt and shall have a nice structured format that you
        # can define as you think it suits best for an LLM
        # There are two modes:
        # - sec2tag (True): For each Section, create a sub-report containing the issues of a section and the fluency and
        # coherence scores it got assigned; issues are categorized in fluency and coherence issues and further
        # subcategorized by tag -> it is important to keep this information
        # - sec2tag (False) -> tag2seq: Collect all fluency and coherence tags that appear in the memory (keep the
        # categorization by metric). For each tag, name the sections that had issues falling into the tag category.
        # In the very end, give the fluency and coherence scores for each section.
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
