import json
import logging
import copy
import os
import random

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, \
    T5ForConditionalGeneration
from evaluate import load
from openai import OpenAI
import tiktoken

from src.utils.generation import convert_empty_strings, default_metrics, default_decoding_args, \
    default_instructions, default_instructions_answer, default_instructions_schema, default_query_prefix, \
    default_answer_prefix, default_schema_prefix, default_output_prefix, default_output_prefix_chat, llama_chat_B_INST, \
    llama_chat_E_INST, llama_chat_B_SYS, llama_chat_DEFAULT_SYSTEM_PROMPT, llama_chat_E_SYS, \
    beluga_DEFAULT_SYSTEM_PROMPT, default_output_prefix_chat_beluga, construct_args_from_example, \
    construct_query_prompt_from_args, StopAfterCharsGenerated, default_inv_instructions, default_inv_question_prefix, \
    default_inv_output_prefix, prompt_arr_2_text, default_instructions_R

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMWrapper:
    def __init__(self, model=None, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto",
                 is_chat=None, is_llama2=None, is_beluga=None, is_t5=None, load_in_8bit=False, openai_api_key=None):
        self.model_name = None
        self.model = None
        self.tokenizer = None
        self.is_llama2 = is_llama2
        self.is_beluga = is_beluga
        self.is_t5 = is_t5
        self.is_chat = is_chat
        self.is_openai = False

        if model is not None:
            self.model_name = model.lower().split('/')[-1]
            self.is_openai = 'openai' in model.lower()
            if not self.is_openai:
                if self.is_llama2 is None:  # then infer from name
                    self.is_llama2 = 'llama-2' in self.model_name
                if self.is_beluga is None:  # then infer from name
                    self.is_beluga = 'beluga' in self.model_name
                if self.is_t5 is None:  # then infer from name
                    self.is_t5 = 't5' in self.model_name
                if self.is_chat is None:  # then infer from name
                    self.is_chat = 'chat' in self.model_name

                cls_tokenizer = AutoTokenizer
                tokenizer_args = {}
                cls_model = AutoModelForCausalLM
                if self.is_llama2:
                    cls_tokenizer = LlamaTokenizer
                    tokenizer_args.update({"add_bos_token": True, "add_eos_token": False})
                    cls_model = LlamaForCausalLM
                self.tokenizer = cls_tokenizer.from_pretrained(model, **tokenizer_args)
                if self.is_t5:
                    cls_model = T5ForConditionalGeneration
                    torch_dtype = torch.float32  # because of a logsoftmax error with half precision; TODO: double-check
                self.model = cls_model.from_pretrained(model,
                                                       torch_dtype=torch_dtype,
                                                       device_map=device_map,
                                                       # low_cpu_mem_usage=low_cpu_mem_usage,
                                                       trust_remote_code=True,
                                                       load_in_8bit=load_in_8bit)
                self.model.eval()

                self.default_decoding_args = default_decoding_args
                if any([m in self.model_name for m in ['falcon', 'mpt']]):
                    self.default_decoding_args.update({
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "pad_token_id": self.tokenizer.eos_token_id
                    })
                self.strip_prompt = any([m in self.model_name for m in ['falcon']])
                # Add stopping criterion that terminates generation when a newline is generated
                self.default_decoding_args.update(
                    {"logits_processor": LogitsProcessorList([StopAfterCharsGenerated(tokenizer=self.tokenizer)])})
            else:
                self.model = {
                    'org': 'openai',
                    'name': self.model_name,
                    'key': openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY"),
                    'session_tokens': 0
                }
                assert self.model['key'] is not None
                self.model['client'] = OpenAI()
                self.default_decoding_args = {
                    "max_tokens": default_decoding_args['max_new_tokens'],
                    "top_p": default_decoding_args['top_p'],
                    "temperature": default_decoding_args['temperature'],
                    "stop": ['\n', '?']
                }
                self.strip_prompt = False
                self.tokenizer = tiktoken.encoding_for_model(self.model_name)
                logger.info(f'Loaded tokenizer for openai/{self.model_name}')

        # TODO: Change these defaults to be generic (not task-specific)
        self.default_instructions = default_instructions
        self.default_instructions_R = default_instructions_R
        self.default_instructions_answer = default_instructions_answer
        self.default_instructions_schema = default_instructions_schema
        self.default_query_prefix = default_query_prefix
        self.default_answer_prefix = default_answer_prefix
        self.default_schema_prefix = default_schema_prefix
        self.default_question_prefix = default_output_prefix  # Used in few-shot demonstrations
        self.default_output_prefix = default_output_prefix
        if self.is_chat:
            self.default_output_prefix = default_output_prefix_chat
            if self.is_beluga:
                self.default_output_prefix = default_output_prefix_chat_beluga
        self.default_inv_instructions = default_inv_instructions
        self.default_inv_question_prefix = default_inv_question_prefix
        self.default_inv_output_prefix = default_inv_output_prefix
        self.default_metrics = default_metrics
        self.datasets = {
            'train': None,
            'dev': None,
            'test': None
        }
        self.dataset_fpaths = {
            'train': None,
            'dev': None,
            'test': None
        }
        self.metrics = {}

    def load_train_set(self, fpath, filter_fns=[]):
        self.dataset_fpaths['train'] = fpath
        with open(fpath, 'r') as fh:
            self.datasets['train'] = json.load(fh)
        if len(filter_fns) > 0:
            keep = []
            for d in self.datasets['train']:
                if all(filter_fn(d) for filter_fn in filter_fns):
                    keep.append(d)
            logger.info(
                f'Filtered {len(self.datasets["train"]) - len(keep)}/{len(self.datasets["train"])} questions from the train set.')
            self.datasets['train'] = keep

    def load_dev_set(self, fpath, filter_fns=[]):
        self.dataset_fpaths['dev'] = fpath
        with open(fpath, 'r') as fh:
            self.datasets['dev'] = json.load(fh)
        if len(filter_fns) > 0:
            keep = []
            for d in self.datasets['dev']:
                if all(filter_fn(d) for filter_fn in filter_fns):
                    keep.append(d)
            logger.info(
                f'Filtered {len(self.datasets["dev"]) - len(keep)}/{len(self.datasets["dev"])} questions from the dev set.')
            self.datasets['dev'] = keep

    def load_test_set(self, fpath, filter_fns=[]):
        self.dataset_fpaths['test'] = fpath
        with open(fpath, 'r') as fh:
            self.datasets['test'] = json.load(fh)
        if len(filter_fns) > 0:
            keep = []
            for d in self.datasets['test']:
                if all(filter_fn(d) for filter_fn in filter_fns):
                    keep.append(d)
            logger.info(
                f'Filtered {len(self.datasets["test"]) - len(keep)}/{len(self.datasets["test"])} questions from the test set.')
            self.datasets['test'] = keep

    def _base_generator(self, prompt, return_output_after_prompt=True, **kwargs):
        if self.is_t5:
            return_output_after_prompt = False

        _prompt = prompt

        decoding_args = copy.deepcopy(self.default_decoding_args)
        decoding_args.update(kwargs)

        if self.strip_prompt:
            _prompt = _prompt.strip()

        assert _prompt != ""

        if not self.is_openai:
            # Tokenize
            prompt_tokenized = self.tokenizer(_prompt, return_tensors="pt", return_token_type_ids=False)
            prompt_tokenized.to("cuda")
            if decoding_args.get('logits_processor', None) is not None:
                # Set prompt length for logits processor
                decoding_args['logits_processor'][0].set_prompt_len(prompt_tokenized.input_ids.size(1))

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**prompt_tokenized, **decoding_args,
                                              return_dict_in_generate=True, output_scores=True)
            decoded = [
                self.tokenizer.decode(o, skip_special_tokens=True)[(len(_prompt) if return_output_after_prompt else 0):] for
                o in outputs.sequences]
        else:
            outputs = self.model['client'].chat.completions.create(
                model=self.model['name'],
                **decoding_args,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. You will only respond to the user's query. You will not respond with any additional conversation or information."},
                    {"role": "user", "content": _prompt}
                ]
            )
            decoded = [outputs.choices[0].message.content]
        return decoded, outputs, _prompt, decoding_args

    def zero_shot(self, query, query_prefix=None, answer=None, answer_prefix=None, schema=None,
                  schema_prefix=None, instructions=None, output_prefix=None, prompt_sep="\n\n",
                  anon_type='label', use_schema=False, use_answer=False, **kwargs):
        # Construct prompt
        prompt = ""
        prompt_arr = []
        # Task instructions
        if instructions is None:
            instructions = copy.copy(self.default_instructions)
            if not anon_type.endswith('-rev'):
                instructions += self.default_instructions_R
            if answer is not None:
                instructions += self.default_instructions_answer
            if schema is not None:
                instructions += self.default_instructions_schema
            instructions = " ".join(instructions)
        if instructions != "":
            prompt_arr.append(instructions)
        # Query
        prompt_arr += construct_query_prompt_from_args(query, query_prefix, answer, answer_prefix, schema,
                                                       schema_prefix, self.default_query_prefix,
                                                       self.default_answer_prefix, self.default_schema_prefix)

        # Get prompt text
        prompt = prompt_arr_2_text(prompt_arr, prompt_sep, self.is_llama2, self.is_beluga, self.is_chat,
                                   self.default_output_prefix if output_prefix is None else output_prefix)

        return self._base_generator(prompt, **kwargs)

    def few_shot(self, query, construct_args_fn, query_prefix=None, answer=None, answer_prefix=None, schema=None,
                 schema_prefix=None, instructions=None, output_prefix=None, prompt_sep="\n\n", n_shots=3,
                 retrieval_strategy='random', demos_split='train', anon_type='label', use_schema=False,
                 use_answer=False, **kwargs):
        demos = self.datasets[demos_split]
        if retrieval_strategy == "random":
            sampled_demos = random.sample(demos, n_shots)
        elif retrieval_strategy == "function":
            raise NotImplementedError()
        elif retrieval_strategy == "tfidf":
            raise NotImplementedError()
        elif retrieval_strategy == "dense":
            raise NotImplementedError()

        # Construct prompt
        prompt = ""
        prompt_arr = []
        # Task instructions
        if instructions is None:
            instructions = copy.copy(self.default_instructions)
            if answer is not None:
                instructions += self.default_instructions_answer
            if schema is not None:
                instructions += self.default_instructions_schema
            instructions = " ".join(instructions)
        if instructions != "":
            prompt_arr.append(instructions)
        # Demonstrations
        for d in sampled_demos:
            d_inf_args, d_ref = construct_args_fn(d, anon_type, use_schema, use_answer)
            prompt_arr += construct_query_prompt_from_args(d_inf_args['query'], query_prefix, d_inf_args['answer'],
                                                           answer_prefix, d_inf_args['schema'], schema_prefix,
                                                           self.default_query_prefix, self.default_answer_prefix,
                                                           self.default_schema_prefix, question=d_ref,
                                                           question_prefix=self.default_question_prefix)
        # Query
        prompt_arr += construct_query_prompt_from_args(query, query_prefix, answer, answer_prefix, schema,
                                                       schema_prefix, self.default_query_prefix,
                                                       self.default_answer_prefix, self.default_schema_prefix)
        # Get prompt text
        prompt = prompt_arr_2_text(prompt_arr, prompt_sep, self.is_llama2, self.is_beluga, self.is_chat,
                                   self.default_output_prefix if output_prefix is None else output_prefix)

        return self._base_generator(prompt, **kwargs)

    def load_metrics(self, metrics):
        self.metrics.update({m: load(m) for m in metrics if m not in self.metrics})

    def get_metric_scores(self, metric, predictions, references):
        self.load_metrics([metric["name"]])
        scores = self.metrics[metric["name"]].compute(
            predictions=convert_empty_strings(predictions) if metric["name"] == "bleu" else
            predictions, references=references, **metric["args"])
        return scores
