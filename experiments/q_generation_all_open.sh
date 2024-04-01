#!/usr/bin/bash -e

flags=${1:-""}
declare -a models=("tiiuae/falcon-7b-instruct" "mosaicml/mpt-7b-instruct" \
"google/flan-t5-xxl" "mosaicml/mpt-30b-instruct" "tiiuae/falcon-40b-instruct")
# "lmsys/vicuna-7b-v1.3" "lmsys/vicuna-13b-v1.3" "lmsys/vicuna-33b-v1.3"

for model in "${models[@]}"; do
  ./experiments/q_generation.sh ${model} "${flags}"
done
