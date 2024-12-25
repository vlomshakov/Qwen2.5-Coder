import os
import json
import sys

from openai import OpenAI

sys.path.append("../../../livecode_bench")
from lcb_runner.runner.parser import get_args
from lcb_runner.lm_styles import LanguageModelStore, LanguageModel
from lcb_runner.runner.vllm_runner import VLLMRunner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    combine_results,
    sort_and_extract_save_results,
    get_metrics,
)

import tqdm


def api_query(messages, model: LanguageModel, args):
    client = OpenAI(
        api_key=model.api_key,
        base_url=model.api_url,

    )

    completion = client.chat.completions.create(
        model=model.model_name,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=args.stop,
        presence_penalty=0,
        frequency_penalty=0,
        top_p=args.top_p,
        n=args.n
    )
    return completion.choices[0].message.content



def main():
    args = get_args()

    model = LanguageModelStore[args.model]
    if args.model_path is not None and args.output_name is not None:
        model.model_name = args.model_path
        model.model_repr = args.output_name
    benchmark, format_prompt = build_prompt_benchmark(args.scenario)

    output_path = get_output_path(model, args)

    if args.continue_existing:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                old_save_results = json.load(f)
        else:
            print(f"File {output_path} does not exist in --continue_existing, starting from scratch")
            old_save_results = []

        old_save_results_question_ids = [instance["question_id"] for instance in old_save_results]
        remaining_benchmark = [instance for instance in benchmark if instance.question_id not in old_save_results_question_ids]
        print(f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining")
    else:
        old_save_results = []
        remaining_benchmark = benchmark

    if len(remaining_benchmark) > 0:
        results = []
        if model.api_url is None:
            runner = VLLMRunner(args, model)
            results: list[str] = runner.run_main(remaining_benchmark, format_prompt)
        else:
            prompts = [
                format_prompt(problem, model.model_style) for problem in benchmark
            ]
            for prompt in tqdm.tqdm(prompts):
                response = api_query(prompt, model, args)
                results.append(response)
    else:
        results = []

    combined_results = combine_results(args.scenario, results, model)

    save_results = [instance.insert_output(outputs_list, extracted_list) for instance, (outputs_list, extracted_list) in zip(benchmark, combined_results)]

    if args.continue_existing:
        save_results += old_save_results

    save_results, combined_results = sort_and_extract_save_results(args.scenario, save_results)

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    if args.evaluate:
        # TODO: we can add --continue_existing_evaluation flag to continue from existing evaluation
        metrics = get_metrics(args.scenario, args, benchmark, combined_results)
        graded = extract_instance_results(metrics[1])

        save_eval_results = [instance.insert_output_evaluation(outputs_list, extracted_list, graded_list) for instance, (outputs_list, extracted_list), graded_list in zip(benchmark, combined_results, graded)]

        with open(output_path.replace(".json", "_eval.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        with open(output_path.replace(".json", "_eval_all.json"), "w") as f:
            json.dump(save_eval_results, f, indent=4)

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            json.dump(save_results, open(f"{args.output_dir}/generation.json", "w"), indent=4)
            json.dump(metrics, open(f"{args.output_dir}/results.json", "w"), indent=4)
            json.dump(save_eval_results, open(f"{args.output_dir}/log.json", "w"), indent=4)


if __name__ == "__main__":
    main()
