#!/usr/bin/bash -e

model=${1:-"mosaicml/mpt-7b-instruct"}
hop=${2:-"0"}  # "0" means mixed
run_zeroshot=${3:-"no"}  # "yes" or "no"
flags=${4:-""} # e.g. "--sparql_cache=path/to/file"
echo "SCRIPT FLAGS: ${flags}"
flags_arr=($flags)

if [ "${hop}" == "0" ]; then
  dev_fpath="data/datasets/metaqa/qa_dev.json"
  iid_train_fpath="data/datasets/metaqa/qa_train.json"
  iid_demos_label="iid"
else
  dev_fpath="data/datasets/metaqa/qa_dev_${hop}-hop.json"
  iid_train_fpath="data/datasets/metaqa/qa_train_${hop}-hop.json"
  iid_demos_label="iid-${hop}"
fi

if [ "${run_zeroshot}" == "yes" ]; then
  # Zero-shot
  python src/reasoner.py --model=${model} --load_dev_fpath=${dev_fpath} --dataset_name=metaqa --eval_n_hops=${hop} --eval_split=dev --eval_inf_fn_key=zeroshot --sparql_cache=data/graphs/metaqa/_sparql_cache.json --rev_schema_fpath=data/graphs/metaqa/schema.json --beam_size=2 "${flags_arr[@]}"  # force run with beam_size=2
fi

# Few-shot (IID)
python src/reasoner.py --model=${model} --load_dev_fpath=${dev_fpath} --dataset_name=metaqa --eval_n_hops=${hop} --eval_split=dev --eval_inf_fn_key=fewshot --load_train_fpath=${iid_train_fpath} --demos_label=${iid_demos_label} --rev_schema_fpath=data/graphs/metaqa/schema.json --sparql_cache=data/graphs/metaqa/_sparql_cache.json "${flags_arr[@]}"

## Few-shot (OOD)
python src/reasoner.py --model=${model} --load_dev_fpath=${dev_fpath} --dataset_name=metaqa --eval_n_hops=${hop} --eval_split=dev --eval_inf_fn_key=fewshot --load_train_fpath=data/datasets/grailqa/qa_train.json --demos_label=ood --rev_schema_fpath=data/graphs/metaqa/schema.json --sparql_cache=data/graphs/metaqa/_sparql_cache.json "${flags_arr[@]}"

# Few-shot (walks)
python src/reasoner.py --model=${model} --load_dev_fpath=${dev_fpath} --dataset_name=metaqa --eval_n_hops=${hop} --eval_split=dev --eval_inf_fn_key=fewshot --load_train_fpath=data/graphs/metaqa/qa_walks_1000_mpt-30b-instruct.json --demos_label=walks-1k-mpt30b --sparql_cache=data/graphs/metaqa/_sparql_cache.json --rev_schema_fpath=data/graphs/metaqa/schema.json "${flags_arr[@]}"
