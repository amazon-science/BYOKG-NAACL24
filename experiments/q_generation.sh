#!/usr/bin/bash -e

model_or_path=${1}  # e.g. meta-llama/Llama-2-7b-hf
flags=${2:-""}  # e.g. "--eval_use_hf_seq_scores"
echo "SCRIPT FLAGS: ${flags}"
flags_arr=($flags)

python src/question_generator.py --model="${model_or_path}" --eval_use_answer --eval_use_schema "${flags_arr[@]}"
python src/question_generator.py --model="${model_or_path}" --eval_use_schema "${flags_arr[@]}"
python src/question_generator.py --model="${model_or_path}" "${flags_arr[@]}"
