#!/usr/bin/bash -e

flags=${1:-""}
declare -a models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-hf" \
"meta-llama/Llama-2-13b-chat-hf" "meta-llama/Llama-2-70b-hf" "meta-llama/Llama-2-70b-chat-hf" "stabilityai/StableBeluga2")

for model in "${models[@]}"; do
  ./experiments/q_generation.sh ${model} "${flags}"
done
