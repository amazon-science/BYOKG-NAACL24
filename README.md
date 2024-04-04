## BYOKG
This is the official implementation of the NAACL'24 paper [Bring Your Own KG: Self-Supervised Program Synthesis for Zero-Shot KGQA](https://arxiv.org/abs/2311.07850).

### Environment setup
```shell
conda create -n byokg python=3.9 --yes
conda activate byokg
pip install -r requirements.txt
sh additional_packages.sh
```

### Download datasets and graphs
```shell
cd data && \
gdown --folder 13w1tfA3YL88y-HwXU3oZlr0oeyIdc_t0 && \
gdown --folder 1YfGmENowy3H1meysBi4AG7oyY2fHITDz && \
cd ..
```

### Virtuoso (SPARQL server) setup
```shell
sh scripts/setup_virtuoso.sh
```

#### Running the server
```shell
# Starting the server (-d specifies the directory containing `virtuoso.db`) at a specific port (e.g. "3001")
python3 virtuoso/virtuoso.py start 3001 -d ./virtuoso

# Stopping the server at a given port (e.g. "3001")
python3 virtuoso/virtuoso.py stop 3001
```
A machine with 100GB RAM is recommended. You may adjust the maximum amount of RAM the service can use and other configurations via the provided script.

### KG setup
#### Freebase
```shell
sh scripts/setup_freebase.sh
```
**Notes:**
- This may take ~10 minutes. It will download and unzip a `virtuoso.db` file containing the Freebase KG (~130G).
- The above command will overwrite any existing `virtuoso.db` DB file in the `./virtuoso` directory. If you plan to also use Freebase in addition to any other KG, we recommend running this first and adding the additional KGs to the same `virtuoso.db` file as described below. If you do not plan to use Freebase, you can skip this step.

#### MoviesKG/MetaQA (and any other arbitrary graph)
##### Generating an n-triples file from a text file of triples 
If you already have an n-triples (`.nt`) file, skip to [n-triples loading](#loading-n-triples-into-virtuoso).  

To generate an n-triples file from a text file (see `data/graphs/metaqa/kb.txt` for an example) of triples:
```shell
python src/explorer.py \
  --kg_name="metaqa" \
  --kg_path="data/graphs/metaqa" \
  --triples_fname="kb.txt" \
  --kg_prefix="movie" \
  --kg_write_ntriples 
```
This will result in `data/graphs/metaqa/graph.nt`, which can be loaded into Virtuoso.

##### Loading n-triples into Virtuoso 
First, stop the Virtuoso server if running, and add the following modification to `./virtuoso/virtuoso.py` to allow Virtuoso to read your n-triples file from the directory where it is stored: 
```shell
# Find "DirsAllowed" and add the absolute path to the directory containing `graph.nt` (or your .nt file)
# For e.g. if the file path is /abs/path/to/project/data/graphs/metaqa/graph.nt, then modify the line to: 
#    DirsAllowed = ., /abs/path/to/project/data/graphs/metaqa\n
```

Now, to load the n-triples file (say, `graph.nt` for MetaQA (MoviesKG)) into Virtuoso:
- Start the server
  ```shell
  python3 virtuoso/virtuoso.py start 3001 -d ./virtuoso
  ```
- Start isql
  ```shell
  virtuoso/virtuoso-opensource/bin/isql 13001
  ```
- Create the new graph
  ```
  SPARQL CREATE GRAPH <http://metaqa.com>;
  ld_dir('/abs/path/to/project/data/graphs/metaqa', 'graph.nt', 'http://metaqa.com');
  rdf_loader_run();
  select * from DB.DBA.load_list;
  exit;
  ```
  **Note:** For other KGs, replace `http://metaqa.com` with `http://<CUSTOM_KG_NAME>.com`.
---

### Graph Exploration
Example command for MoviesKG:
```shell
python src/explorer.py \
  --kg_explore \
  --kg_name="metaqa" \
  --kg_prefix="movie" \
  --kg_path="data/graphs/metaqa" \
  --sparql_cache="data/graphs/metaqa/_sparql_cache.json" \
  --kg_n_walks=10000 \
  --save_interval=500
```
**Notes:**
- This will output `results.json` containing the explored programs and `stats.json` containing some analyses for the exploration run.
- Certain flags, such as `--filter_empty_walks` and `--prune_redundant` (True by default), require a virtuoso server running at `--sparql_url`.
- `--kg_prefix` is used for MoviesKG due to the provided triples file. This flag is not needed for Freebase (`kg_name="freebase"`), and may not be needed for other KGs.

### Query Generation
First, we need to preprocess the output from `explorer.py`:
```shell
python scripts/prep_data_for_qgen.py \
  --walks_fpath=path/to/explorer/results.json \
  --rev_schema_fpath=data/graphs/metaqa/schema.json
```
**Notes:**
- This will output `qgen_walks.json` in the same directory as `path/to/explorer/results.json`.
- When using GrailQA training data instead of explorations in `--walks_fpath`, add `--sexpr_machine_key="s_expression"`.

Now, run generation:
```shell
# Example with open-source models hosted on HuggingFace
python src/question_generator_l2m.py \
  --model="mosaicml/mpt-7b-instruct" \
  --kg_name=metaqa \
  --kg_schema_fpath=data/graphs/metaqa/schema.json \
  --load_dev_fpath=path/to/qgen_walks.json \
  --eval_output_sampling_strategy=inverse-len-norm \
  --force_type_constraint \
  --save_interval=200
  
# Example with OpenAI API
python src/question_generator_l2m.py \
  --model="openai/gpt-4" \ 
  --kg_name=metaqa \
  --kg_schema_fpath=data/graphs/metaqa/schema.json \
  --load_dev_fpath=path/to/qgen_walks.json \
  --eval_output_sampling_strategy=max \  # This is the only sampling strategy available for OpenAI API (will be forced)
  --force_type_constraint \
  --save_interval=200
```
**Notes:**
- This will output `results.json` containing natural language questions for the processed programs (walks).

### Reasoning

For MetaQA, we first need to construct the dataset (this needs to be run only once):
```
python scripts/build_metaqa_dataset.py
python scripts/prep_data_for_qa.py \
  --metaqa_dir=data/datasets/metaqa \
  --rev_schema_fpath=data/graphs/metaqa/schema.json \
  --sexpr_machine_key=s_expression
```

After running query generation, first preprocess the output from `question_generator_l2m.py`:
```shell
python scripts/prep_data_for_qa.py \
  --walks_qgen_in_fpath=path/to/qgen_walks.json \
  --walks_qgen_out_fpath=path/to/qgen/results.json
```
This will output `qa_walks.json`.

Now, run reasoning. Example command for MetaQA:
```shell
python src/reasoner.py \
  --model="mosaicml/mpt-7b-instruct" \
  --kg_name=metaqa \
  --dataset_name=metaqa \
  --eval_split=test \
  --load_test_fpath=data/datasets/metaqa/qa_test.json \
  --eval_n_samples=-1 \
  --sparql_cache=data/graphs/metaqa/_sparql_cache.json \
  --rev_schema_fpath=data/graphs/metaqa/schema.json \
  --load_train_fpath=path/to/qa_walks.json \
  --demos_label=walks \
  --save_interval=100

# The above will output `results.json`. Then, run candidate re-ranking:
python src/reasoner.py \
  --model="mosaicml/mpt-7b-instruct" \
  --kg_name=metaqa \
  --sparql_cache=data/graphs/metaqa/_sparql_cache.json \
  --rev_schema_fpath=data/graphs/metaqa/schema.json \
  --rerank_fpath=path/to/reasoning/results.json
```
**Notes:**
- This will output `results.json` and `reranked_results.json`.
- You can also pass curated training data to `--load_train_fpath` instead of the explorations, but make sure it is in the same format as that expected here. See the `prep_data_for_*` files.
- As of 01/2024, OpenAI API does not provide access to the logits of the full sequence, which is required for the BYOKG reasoner. We're therefore currently only supporting models hosted on HuggingFace.

### Citation
If you use any component of this project for your work, please cite the following
```
@inproceedings{
  agarwal2024bring,
  title={Bring Your Own {KG}: Self-Supervised Program Synthesis for Zero-Shot {KGQA}},
  author={Dhruv Agarwal and Rajarshi Das and Sopan Khosla and Rashmi Gangadharaiah},
  booktitle={2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
  year={2024},
  url={https://openreview.net/forum?id=Z1IscjaN3g}
}
```
## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

