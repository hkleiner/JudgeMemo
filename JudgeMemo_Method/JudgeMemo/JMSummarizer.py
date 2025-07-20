from JudgeMemo import JMUtils
from JudgeMemo.JMScanner import JMScan


class JMSummarizer:
    """
    Summarizes either entire documents or individual sections using a given language model.

    Attributes:
        model: A chat-compatible language model capable of generating summaries.
        sampling_params: Sampling parameters to be used during generation.
        doc_id (str): Identifier for the document being summarized.
    """
    def __init__(self,
                 model,
                 sampling_params,
                 doc_id: str
                 ):
        """
        Initialize the JMSummarizer.

        Args:
            model: The language model used for summarization.
            sampling_params: Parameters to control the sampling behavior of the model.
            doc_id (str): Unique identifier for the document to be summarized.
        """
        self.model = model
        self.sampling_params = sampling_params
        self.doc_id = doc_id

    def summarize_sections(self,
                           doc_sections: list[JMScan],
                           summary_template: str,
                           summary_sys: str,
                           ):
        """
        Generate summaries for each section of a document.

        Each section's summary is created using a system prompt and a summary template.
        Summaries are also optionally assigned to the next section as previous context.

        Args:
            doc_sections (list[JMScan]): List of JMScan objects representing the document's sections.
            summary_template (str): Prompt template used to guide the summarization.
            summary_sys (str): System prompt string defining summarization behavior.

        Returns:
            list[JMScan]: The list of JMScan objects with their `sec_summary` and `prev_summary` attributes updated.
        """
        summary_prompts = []
        for doc_sec in doc_sections:
            summary = JMUtils.get_prompt_1(
                text=doc_sec.sec_text,
                sys_prompt=summary_sys,
                prompt_template=summary_template
            )
            summary_prompts.append(summary)
        print(f"--> Created all summary prompts for document {self.doc_id}")

        # Generate summaries
        summary_outputs = self.model.chat(
            messages=summary_prompts,
            sampling_params=self.sampling_params,
            use_tqdm=True
        )

        # Process all summaries for all sections of the document at once
        for i, response in enumerate(summary_outputs):
            summary = response.outputs[0].text
            doc_sections[i].set_sec_summary(summary)
            if i > 0:
                if i < len(summary_outputs)-1:
                    doc_sections[i+1].set_prev_sec_summary(summary)
            else:
                doc_sections[i].set_prev_sec_summary('[no preceding section summary - beginning of text]')
        print(f"--> Created all summaries for document {self.doc_id}")

        return doc_sections

    def summarize(self,
                  text,
                  summary_template: str,
                  summary_sys: str,
                  ):
        """
        Generate a summary for the full document text.

        Args:
            text (str): The complete document text to summarize.
            summary_template (str): Prompt template used to guide the summarization.
            summary_sys (str): System prompt string defining summarization behavior.

        Returns:
            str: The generated summary of the document.
        """
        prompt = [JMUtils.get_prompt_1(
                text=text,
                sys_prompt=summary_sys,
                prompt_template=summary_template
            )
        ]

        # Generate summaries
        summary_outputs = self.model.chat(
            messages=prompt,
            sampling_params=self.sampling_params,
            use_tqdm=True
        )

        for response in summary_outputs:
            summary = response.outputs[0].text
            print(f"--> Created summary for document {self.doc_id}")
            return summary
