from JudgeMemo import JMUtils
from JudgeMemo.JMScanner import JMScan
import re


class JMJudge:
    def __init__(self,
                 model,
                 sampling_params,
                 doc_id: str
                 ):
        self.model = model
        self.sampling_params = sampling_params
        self.doc_id = doc_id
        self.MAX_RETRIES = 3  # maximum number of retries for output completion

    @staticmethod
    def is_output_complete(text: str) -> bool:
        """
        Returns False if 'finish_reason: length' is found in the text,
        meaning the model output was cut off and needs to be rerun.
        Otherwise, returns True.
        """
        return not bool(re.search(r"^\s*finish_reason:\s*length\s*$", text, re.MULTILINE))

    def evaluate_sections(self,
                          doc_sections: list[JMScan],
                          sec_evaluation_template: str,
                          eval_sys: str,
                          include_sec_summaries: bool,
                          scan_technique: str,
                          sec_eval_path: str
                          ):
        eval_prompts = []

        if scan_technique == "hard":  # hard
            if include_sec_summaries:  # section with context (summary from before + info that something follows)
                for doc_sec in doc_sections:
                    sec_evaluation = JMUtils.get_prompt_2(
                        content=doc_sec.sec_text,
                        add_on=doc_sec.prev_summary,
                        sys_prompt=eval_sys,
                        prompt_template=sec_evaluation_template
                    )
                    eval_prompts.append(sec_evaluation)
            else:
                for doc_sec in doc_sections:
                    sec_evaluation = JMUtils.get_prompt_1(
                        text=doc_sec.sec_text,
                        sys_prompt=eval_sys,
                        prompt_template=sec_evaluation_template
                    )
                    eval_prompts.append(sec_evaluation)
        else:  # stride
            if include_sec_summaries:
                for doc_sec in doc_sections:
                    sec_evaluation = JMUtils.get_prompt_3(
                        content=doc_sec.sec_text,
                        text=doc_sec.prev_summary,
                        add_on=doc_sec.prev_context,
                        sys_prompt=eval_sys,
                        prompt_template=sec_evaluation_template
                    )
                    eval_prompts.append(sec_evaluation)
            else:
                for doc_sec in doc_sections:
                    sec_evaluation = JMUtils.get_prompt_2(
                        content=doc_sec.sec_text,
                        add_on=doc_sec.prev_context,
                        sys_prompt=eval_sys,
                        prompt_template=sec_evaluation_template
                    )
                    eval_prompts.append(sec_evaluation)

        completed_responses = [None] * len(eval_prompts)
        remaining_indices = list(range(len(eval_prompts)))
        attempt = 0

        # Retry logic
        while remaining_indices and attempt <= self.MAX_RETRIES:
            print(f"--> Generation attempt {attempt + 1} for {len(remaining_indices)} sections...")

            current_prompts = [eval_prompts[i] for i in remaining_indices]

            outputs = self.model.chat(
                messages=current_prompts,
                sampling_params=self.sampling_params,
                use_tqdm=True
            )

            next_remaining = []

            for i, output in enumerate(outputs):
                idx = remaining_indices[i]
                response = output.outputs[0].text
                finish_reason = output.outputs[0].finish_reason

                response_with_reason = response + f"\n\n----------------------------\nfinish_reason: {finish_reason}"

                if self.is_output_complete(response_with_reason):
                    completed_responses[idx] = response_with_reason
                else:
                    next_remaining.append(idx)

            remaining_indices = next_remaining
            attempt += 1

        # Final write to each section object
        for i, response in enumerate(completed_responses):
            if response is not None:
                doc_sections[i].set_sec_evaluation(response)
                JMUtils.save_to_text(response, f"{sec_eval_path}/sec_evaluation_{self.doc_id}/section_{i+1}.txt")
            else:
                # fallback: one last try without loop
                print(f"--> Final retry for section index {i}")
                last_output = self.model.chat(
                    messages=[eval_prompts[i]],
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )[0]

                final_text = last_output.outputs[0].text
                finish_reason = last_output.outputs[0].finish_reason
                response_with_reason = final_text + f"\n\n----------------------------\nfinish_reason: {finish_reason}"

                if self.is_output_complete(response_with_reason):
                    doc_sections[i].set_sec_evaluation(response_with_reason)
                else:
                    doc_sections[i].set_sec_evaluation(response_with_reason + "\nWARNING: INCOMPLETE OUTPUT after max retries")

        return doc_sections

    def get_final_evaluation(self,
                             report_mode: str,
                             text: str,
                             report: str,
                             doc_summary: str,
                             report_eval_prompt_template: str,
                             eval_sys: str,
                             save_prompt_path: str
                             ):
        if report_mode == "report_only":
            eval_prompt = [
                JMUtils.get_prompt_1(
                    text=report,
                    sys_prompt=eval_sys,
                    prompt_template=report_eval_prompt_template
                )
            ]
        elif report_mode == "report_summary" or report_mode == "report_original":
            eval_prompt = [
                JMUtils.get_prompt_2(
                    content=report,
                    add_on=doc_summary if report_mode == "report_summary" else text,
                    sys_prompt=eval_sys,
                    prompt_template=report_eval_prompt_template
                )
            ]
        else:
            raise ValueError(
                "Invalid mode! Cannot create an report! Choose from [report_only, report_summary, report_original, report_summary_original]")
        
        # save prompt for later use
        if save_prompt_path:
            JMUtils.save_to_text(eval_prompt[0][1]["content"], save_prompt_path)

        completed = None
        attempt = 0

        while attempt <= self.MAX_RETRIES:
            print(f"--> Generation attempt {attempt + 1} for final evaluation...")

            response = self.model.chat(messages=eval_prompt,
                                    sampling_params=self.sampling_params,
                                    use_tqdm=False)[0]

            output_text = response.outputs[0].text
            finish_reason = response.outputs[0].finish_reason
            full_response = output_text + f"\n\n----------------------------\nfinish_reason: {finish_reason}"

            if self.is_output_complete(full_response):
                completed = full_response
                break

            attempt += 1

        if not completed:
            # final fallback retry
            response = self.model.chat(messages=eval_prompt,
                                    sampling_params=self.sampling_params,
                                    use_tqdm=False)[0]

            output_text = response.outputs[0].text
            finish_reason = response.outputs[0].finish_reason
            full_response = output_text + f"\n\n----------------------------\nfinish_reason: {finish_reason}"
            
            if not self.is_output_complete(full_response):
                full_response += "\nWARNING: INCOMPLETE OUTPUT after max retries"
            completed = full_response

        return completed
