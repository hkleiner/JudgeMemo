from JudgeMemo import JMUtils
from JudgeMemo.JMScanner import JMScanner
from JudgeMemo.JMSummarizer import JMSummarizer
from JudgeMemo.JMMemoryCreator import JMMemoryCreator
from JudgeMemo.JMJudge import JMJudge


class JMProcessor:
    """
    JMProcessor orchestrates the JudgeMemo pipeline by integrating text scanning,
    summarization, evaluation, memory creation, and final report generation.

    It processes a given document text using the specified model and parameters,
    generating section-level evaluations, optional summaries, and a final report
    with an overall quality rating.

    Attributes:
        doc_id (str): Unique identifier for the document.
        text (str): Input text content to be processed.
        THINK (bool): Flag to control inclusion of internal reasoning in prompts.
        scanner (JMScanner): Instance for scanning the document into sections.
        summarizer (JMSummarizer): Instance for summarizing text sections or full text.
        judge (JMJudge): Instance for evaluating text quality on section and document levels.
        memory_creator (JMMemoryCreator): Instance for storing evaluation results and creating reports.

    Methods:
        process(report_mode: str,
                sec2tag: bool = True,
                scan_range: int = 2000,
                scan_overlap_ratio: float = 0.0,
                sec_eval_prompt_path: str = "",
                report_eval_prompt_path: str = "",
                eval_sys: str = "You are a human annotator that rates the quality of texts.",
                include_sec_summaries: bool = False,
                summary_sec_path: str = "./prompts/prompt_section_summary.txt",
                summary_doc_path: str = "./prompts/prompt_document_summary.txt",
                summary_sys: str = "You are a helpful assistant that summarizes text clearly and concisely.\nFocus on the most important points. Avoid repeating content.\nMaintain the original meaning without adding new information. Use plain language.",
                memory_path: str = ".memory_file.json",
                report_path: str = "./report.txt",
                sec_eval_path: str = "./sec_evaluations/",
                save_prompt_path: str = "./prompt.txt") -> dict:
            Executes the full JudgeMemo processing pipeline on the input text and returns
            the final evaluation.

    """
    def __init__(self,
                 model,
                 sampling_params,
                 model_name: str,
                 text: str = "This is a sample text.",
                 doc_id: str = "JM-TEST",
                 THINK: bool = False
                 ):
        self.doc_id = doc_id
        self.text = text
        self.THINK = THINK

        self.scanner = JMScanner(
            text=self.text,
            tokenizer=model_name,
            doc_id=doc_id
        )
        self.summarizer = JMSummarizer(
            model=model,
            sampling_params=sampling_params,
            doc_id=doc_id
        )
        self.judge = JMJudge(
            model=model,
            sampling_params=sampling_params,
            doc_id=doc_id
        )
        self.memory_creator = JMMemoryCreator(
            doc_id=doc_id
        )

    def process(self,
                report_mode: str,  # mode for final evaluation (what shall be included)
                sec2tag: bool = True,  # report format
                scan_range: int = 2000,  # section size for scanning
                scan_overlap_ratio: float = 0.0,  # scan technique used for scanning: 0.0 - hard; > 0 - stride
                sec_eval_prompt_path: str = "",  # prompt used for section evaluations
                report_eval_prompt_path: str = "",  # prompt used for final report evaluation
                eval_sys: str = """You are a human annotator that rates the quality of texts.""",  # system prompt for evaluating
                include_sec_summaries: bool = False,  # summaries of sections (while scanning)
                summary_sec_path: str = "./prompts/prompt_section_summary.txt",  # prompt used for section summaries
                summary_doc_path: str = "./prompts/prompt_document_summary.txt",  # prompt used for full summary
                summary_sys: str = """You are a helpful assistant that summarizes text clearly and concisely.
                                        Focus on the most important points. Avoid repeating content. 
                                        Maintain the original meaning without adding new information. Use plain language.""",  # system prompt for evaluation
                memory_path: str = ".memory_file.json",  # path to memory file (for storing results)
                report_path: str = "./report.txt",  # path to report file (for storing final evaluation)
                sec_eval_path: str = "./sec_evaluations/",  # path to save section evaluations
                save_prompt_path: str = "./prompt.txt"  # path to save the prompt used for processing
                ):
        """
        Processes the input text through scanning, summarization, evaluation, memory creation,
        and final report generation steps, then returns the final evaluation result.

        Args:
            report_mode (str): Mode for final evaluation. Controls what information
                is included in the final report and evaluation. Examples:
                - "report_summary": use document summary in evaluation.
                - "report_original": use original text in evaluation.
            sec2tag (bool, optional): If True, converts section IDs to tags in the final report.
                Defaults to True.
            scan_range (int, optional): Size (in tokens or characters) of sections to scan.
                Defaults to 2000.
            scan_overlap_ratio (float, optional): Overlap ratio between scanned sections.
                0.0 means hard split, >0 means stride overlap. Defaults to 0.0.
            sec_eval_prompt_path (str, optional): Path to the prompt template used for section
                evaluation. If empty, a default prompt is selected based on scanning mode.
                Defaults to "".
            report_eval_prompt_path (str, optional): Path to the prompt template used for the
                final report evaluation. If empty, a default prompt is selected based on report_mode.
                Defaults to "".
            eval_sys (str, optional): System prompt providing instructions for the evaluation model.
                Defaults to "You are a human annotator that rates the quality of texts.".
            include_sec_summaries (bool, optional): Whether to generate summaries of document sections
                before evaluation. Defaults to False.
            summary_sec_path (str, optional): Path to the prompt template for section summaries.
                Defaults to "./prompts/prompt_section_summary.txt".
            summary_doc_path (str, optional): Path to the prompt template for full document summary.
                Defaults to "./prompts/prompt_document_summary.txt".
            summary_sys (str, optional): System prompt for guiding the summarizer's behavior.
                Defaults to a clear, concise, no-addition summarization instruction.
            memory_path (str, optional): Path for saving the memory JSON file containing evaluations.
                Defaults to ".memory_file.json".
            report_path (str, optional): Path for saving the final evaluation report text file.
                Defaults to "./report.txt".
            sec_eval_path (str, optional): Directory path to save section-level evaluation results.
                Defaults to "./sec_evaluations/".
            save_prompt_path (str, optional): Path to save the prompt used during processing.
                Defaults to "./prompt.txt".

        Returns:
            str: Final evaluation results returned by the judge after processing all steps.

        Raises:
            IOError: If reading/writing files or prompt templates fails.
            Exception: For any unexpected error during processing steps.
        """
        # 0) get settings from parameters
        full_summary = ""
        scan_mode = "hard" if scan_overlap_ratio == 0.0 else "stride"

        if not sec_eval_prompt_path:  # set evaluation prompt for default
            sec_summaries = "_summary" if include_sec_summaries else ""
            sec_eval_prompt_path = f"./prompts/prompt_v6-3_JudgeMemo_{scan_mode}{sec_summaries}.txt"

        if not report_eval_prompt_path:  # set report evaluation prompt for default
            report_eval_prompt_path = f"./prompts/{report_mode}_eval_prompt.txt"

        # 1) get sections for document
        doc_sections = self.scanner.scan(
            scan_range=scan_range,
            overlap_ratio=scan_overlap_ratio,
            scan_mode=scan_mode
        )  # returns a list of JMScans

        # optional: 2a) generate full summary for reporting
        if report_mode == "report_summary":
            full_summary = self.summarizer.summarize(
                text=self.text,
                summary_template=JMUtils.load_prompt_template(summary_doc_path),
                summary_sys=summary_sys,
            )  # summarize the full document

        # optional: 2b) generate section summaries
        if include_sec_summaries:
            # summarize each section
            doc_sections = self.summarizer.summarize_sections(
                doc_sections=doc_sections,
                summary_template=JMUtils.load_prompt_template(summary_sec_path),
                summary_sys=summary_sys,
            )  # with summaries

        # 3) evaluate each section
        doc_sec_evaluations = self.judge.evaluate_sections(
            doc_sections=doc_sections,
            include_sec_summaries=include_sec_summaries,
            # sec_evaluation_template="You are a human annotator that rates the quality of texts.\n\n" + JMUtils.load_prompt_template(sec_eval_prompt_path),  # Llama-3_3-Nemotron-Super-49B-v1
            sec_evaluation_template=JMUtils.load_prompt_template(sec_eval_prompt_path) + (" \\think" if self.THINK else "\\no_think"),  # Qwen3-32B
            # sec_evaluation_template=JMUtils.load_prompt_template(sec_eval_prompt_path)  # Llama-3.3-70B-Instruct
            eval_sys=eval_sys,
            scan_technique=scan_mode,
            sec_eval_path=sec_eval_path
        )

        # 4) create memory entry for document
        self.memory_creator.create_memory(
            scanner_results=doc_sec_evaluations,
            memory_file=memory_path,
            is_raw_text=True
        )

        # 5) create final report for full document evaluation
        report = self.memory_creator.create_report(
            sec2tag=sec2tag,
            report_file=report_path
        )

        # 6) evaluate the full document based on mode (report + ...)
        final_evaluation = self.judge.get_final_evaluation(
            report_mode=report_mode,
            text=self.text if report_mode == "report_original" else "",
            report=report,
            doc_summary=full_summary if report_mode == "report_summary" else "",
            # report_eval_prompt_template="You are a human annotator that rates the quality of texts.\n\n" + JMUtils.load_prompt_template(report_eval_prompt_path),  # Llama-3_3-Nemotron-Super-49B-v1
            report_eval_prompt_template=JMUtils.load_prompt_template(report_eval_prompt_path) + " \\think" if self.THINK else "\\no_think", # Qwen3-32B
            # report_eval_prompt_template = JMUtils.load_prompt_template(report_eval_prompt_path) # Llama-3.3-70B-Instruct
            eval_sys=eval_sys,
            save_prompt_path=save_prompt_path
        )

        # 7) return final evaluation of the document
        return final_evaluation
