from typing import List, Any

import tqdm
from openai import OpenAI

from lcb_runner.runner.base_runner import BaseRunner
from lcb_runner.utils.multiprocess import run_tasks_in_parallel


class ApiRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.model = model


    def _run_single(self, prompt: Any) -> str:
        client = OpenAI(
            api_key=self.model.api_key,
            base_url=self.model.api_url
        )

        completion = client.chat.completions.create(
            model=self.model.model_name,
            messages=prompt,
            temperature=self.args.temperature,
            max_tokens=self.args.max_tokens,
            stop=self.args.stop,
            presence_penalty=0,
            frequency_penalty=0,
            top_p=self.args.top_p,
            n=self.args.n
        )
        return completion.choices[0].message.content

    def run_batch(self, prompts: List[str]) -> List[List[str]]:
        if self.args.multiprocess > 1:
            task_results = run_tasks_in_parallel(
                self.run_single,
                prompts,
                self.args.multiprocess,
                self.args.timeout,
                True,
            )

            result = []
            for tr in task_results:
                if tr.is_success():
                    result.append([tr.result])
                else:
                    print(tr)
                    result.append(None)
            return result
        else:
            return [[self.run_single(prompt)] for prompt in tqdm.tqdm(prompts)]
