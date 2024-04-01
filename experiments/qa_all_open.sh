#!/usr/bin/bash -e

run_zeroshot=${1:-"no"}  # "yes" or "no"
flags=${2:-""}
declare -a models=("tiiuae/falcon-7b-instruct" "mosaicml/mpt-7b-instruct" \
"google/flan-t5-xxl" "mosaicml/mpt-30b-instruct" "tiiuae/falcon-40b-instruct")
declare -a hops=("0")  # ("1" "2" "3"); "0" means mixed

for model in "${models[@]}"; do
  for hop in "${hops[@]}"; do
    ./experiments/qa.sh ${model} ${hop} ${run_zeroshot} "${flags}"
  done
done
