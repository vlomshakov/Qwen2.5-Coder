
OUTPUT_DIR="./output"
echo "LiveCodeBench: ${MODEL_DIR}, OUPTUT_DIR: ${OUTPUT_DIR}"


python -m lcb_runner.runner.main --model "tgi" --scenario codegeneration --evaluate --output_dir ${OUTPUT_DIR}
saved_eval_all_file="${OUTPUT_DIR}/log.json"
python -m lcb_runner.evaluation.compute_scores --eval_all_file ${saved_eval_all_file} --start_date 2024-05-01

