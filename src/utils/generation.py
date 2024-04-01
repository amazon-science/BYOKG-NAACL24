import copy
import logging
import os
import re
from collections import defaultdict

import numpy as np
import torch
from transformers import StoppingCriteria, LogitsProcessor
import openai

from src.utils.parser import bottom_up_to_sexpr

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

default_decoding_args = {
    "max_new_tokens": 100,
    "do_sample": False,  # enable sampling
    "top_p": 0.9,  # nucleus sampling
    "temperature": 0.6,  # lower makes the distribution sharper
    "min_length": None,
    "use_cache": True,
    "top_k": 100,  # restrict to top-k probability tokens
    "repetition_penalty": 1.,  # 1 means no penalty; up to inf
    "length_penalty": 1.,  # length_penalty > 0.0 == longer sequences; length_penalty < 0.0 == shorter sequences
    "num_beams": 10,  # beam search
    "num_return_sequences": 10,  # number of beams to return
    "no_repeat_ngram_size": 10,
    "renormalize_logits": True
}
default_instructions = [
    """### Instructions:\nTranslate the following logical form query into a natural language question in English.""",
    """The generated question must have the same meaning as the logical query.""",
    # TODO: Change "generated" to "translated"? You're not talking to the LLM!
    #       You're conditioning it to mimic tasks found on the internet, which are directed at humans.
    """The generated question must cover all and only the information present in the logical query."""]
default_instructions_R = [
    """An "R" before a relation in the logical query indicates a reverse or inverse relation."""
]
default_instructions_answer = [
    """The generated question must be a valid question for the provided sample answer.""",
    """The generated question, however, must not contain the sample answer."""]
default_instructions_schema = [
    """The generated question should use the schema which describes the entities, relations, and functions present in the logical query."""]
default_instructions_l2m = [
    """Use each previous query and solution as a hint to solve the next query."""
]
default_query_prefix = """### Logical Query:\n"""
default_answer_prefix = """### Sample Answer:\n"""
default_schema_prefix = """### Schema:\n"""
default_output_prefix = f"""### English Question:\n"""
default_output_prefix_chat = f"""Assistant: Sure, I'm happy to help you with that. Here is a possible question in English:\n\n"""
default_output_prefix_chat_beluga = f"""### Assistant:\nSure, I'm happy to help you with that. Here is a possible question in English:\n\n"""
default_inv_instructions = """### Instructions:\nTranslate the following question into its semantic parse."""
default_inv_question_prefix = """### Question:\n"""
default_inv_output_prefix = """### Semantic Parse:\n"""

default_metrics = [
    {"name": "rouge", 'score_keys': ['rouge1', 'rouge2', 'rougeL'], 'args': {}},
    {"name": "bleu", 'score_keys': ['bleu'], 'args': {'max_order': 2}},
    {"name": "bertscore", 'score_keys': ['f1'], 'args': {'model_type': 'distilbert-base-uncased'}}
]

llama_chat_B_INST, llama_chat_E_INST = "[INST]", "[/INST]"
llama_chat_B_SYS, llama_chat_E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
llama_chat_DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

beluga_DEFAULT_SYSTEM_PROMPT = "### System:\nYou are a helpful AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"


def fix_posthoc(outputs, remove_special=False):
    new_outputs = []
    for o in outputs:
        if type(o) is list:
            new_outputs.append(fix_posthoc(o, remove_special=remove_special))
        else:
            new_o = o
            # Whitespace
            new_o = new_o.strip()
            # Lowercase
            new_o = new_o.lower()
            if remove_special:
                # Special characters
                new_o = re.sub('[^A-Za-z0-9 ]+', '', new_o)
            # Quotations
            new_o = new_o.replace('"', '')
            new_outputs.append(new_o)
    # Add more fixes here as required
    return new_outputs


def convert_empty_strings(strs, to_str="n/a"):
    res = []
    for s in strs:
        if s == "" or s is None:
            res.append(to_str)
        else:
            res.append(s)
    return res


class StoppingCriteriaNewLine(StoppingCriteria):
    # NOT USING THIS; keeping code for future reference
    # Problem with this approach: the stopping flag applies to all the beams being processed.
    # This means in the code below, as soon as a new line is encountered in any branch, we stop generation.

    def __init__(self, tokenizer, stop_word="\n"):
        super().__init__()
        self.stop = tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze().reshape(-1)[-1].to("cuda")
        self.first_skipped = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.first_skipped:
            if self.stop == input_ids[0][-1]:
                self.first_skipped = False
                return True
            return False
        self.first_skipped = True
        return False


class InverseConsistencyDecoding(LogitsProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # TODO: Get top-k tokens, compute inverse-log-prob scores and set as logits for these tokens, set others to -inf
        return scores


class StopAfterCharsGenerated(LogitsProcessor):
    # Forces an EOS token in the next generation step for a branch in the beam if a newline is encountered
    def __init__(self, tokenizer, stop_chars=['\n', '?']):
        super().__init__()
        # Debug:
        # self.tokenizer = tokenizer
        self.prompt_len = 0  # init
        last_char_idx = -1 if 't5' not in tokenizer.name_or_path else -2
        self.stop_token_ids = [tokenizer(sc).input_ids[last_char_idx] for sc in stop_chars]
        self.eos_token_id = tokenizer.eos_token_id

    def set_prompt_len(self, prompt_len):
        # Make sure to call this with the length of the prompt tokens before calling model.generate
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Debug:
        # print([self.tokenizer.decode(o[self.prompt_len:], skip_special_tokens=True) for o in input_ids])
        if input_ids.size(1) == self.prompt_len:
            # Prevent chars from being the first generation
            for stop in self.stop_token_ids:
                scores[:, stop] = -float("inf")
        if input_ids.size(1) > self.prompt_len:
            forced_eos = torch.full((scores.size(1),), -float("inf"),
                                    dtype=scores.dtype).cuda()  # Forced probability of 0
            forced_eos[self.eos_token_id] = 0  # Force probability of 1
            # Force generation of EOS for whichever beam branch that just generated a newline
            for stop in self.stop_token_ids:
                scores[input_ids[:, -1] == stop] = forced_eos
        return scores


def stringify_schema(schema, anon_type, query=None):
    schema_arg = []
    if query is not None and type(query) is not list:
        query = [query]
    for k, v in schema[f'nodes_{anon_type}'].items():
        if query is None or any(k in q for q in query):
            schema_arg.append(f"{k}={v}")
    for k, v in schema['relations'].items():
        if query is None or any(k in q for q in query):
            schema_arg.append(f"{k}={v}")
    for k, v in schema['functions'].items():
        if query is None or any(f"{k} " in q for q in query):
            schema_arg.append(f"{k}={v}")
    return "; ".join(schema_arg)


def construct_args_from_example(example, anon_type, use_schema, use_answer=False, bottom_up_parse=False):
    if bottom_up_parse:
        queries = bottom_up_to_sexpr(example[f's_expression_{anon_type}'], return_all=True)
    else:
        queries = [example[f's_expression_{anon_type}']]
    inf_args_list = []
    ref = example[f'question_{anon_type}'] if f'question_{anon_type}' in example else None
    if use_schema:
        # Always use readable questions if schema is provided
        ref = example['question_label'] if 'question_label' in example else None
    for query in queries:
        inf_args = {'query': query, 'answer': None, 'schema': None}
        if use_answer:
            if 'sample_answer' not in example:
                raise ValueError('`sample_answer` not found in example while using use_answer=True mode')
            inf_args['answer'] = example['sample_answer']['value_readable']
        if use_schema:
            inf_args['schema'] = stringify_schema(schema=example['schema'], anon_type=anon_type, query=inf_args['query'])
        inf_args_list.append(inf_args)
    if len(inf_args_list) == 1:
        inf_args_list = inf_args_list[0]
    return inf_args_list, ref


def construct_query_prompt_from_args(query, query_prefix, answer, answer_prefix, schema, schema_prefix,
                                     default_query_prefix, default_answer_prefix, default_schema_prefix,
                                     question=None, question_prefix=None):
    prompt_arr = []  # this is later converted into a string using "{sep}".join(), where `sep` may be "\n\n"
    # The test or demonstration question
    if query_prefix is None:
        query_prefix = copy.copy(default_query_prefix)
    query_text = f"{query_prefix}{query}"
    prompt_arr.append(query_text)
    # A sample answer for the test question, if provided
    if answer is not None:
        if answer_prefix is None:
            answer_prefix = copy.copy(default_answer_prefix)
        answer_text = f"{answer_prefix}{answer}"
        if answer_text != "":
            prompt_arr.append(answer_text)
    # The schema elements of the test question
    if schema is not None:
        if schema_prefix is None:
            schema_prefix = copy.copy(default_schema_prefix)
        schema_text = f"{schema_prefix}{schema}"
        if schema_text != "":
            prompt_arr.append(schema_text)
    # Demonstration of expected output (used for few-shot setup; otherwise not used)
    if question is not None and question_prefix is not None:
        question_text = f"{question_prefix}{question}"
        if question_text != "":
            prompt_arr.append(question_text)
    return prompt_arr


def get_logprob_score(target, prefix, model, tokenizer, len_norm=True, bsz=10):
    if type(model) is dict and model['org'] == 'openai':
        openai.api_key = model['key']
        # Computing logprobs of a specified sequence of text is only supported by the following until 2024-01-04:
        # text-davinci-003, text-davinci-002, text-davinci-001 ($0.0200 / 1K tokens)
        # text-curie-001 ($0.0020 / 1K tokens)
        # text-babbage-001 (0.0005 / 1K tokens)
        response = openai.Completion.create(model=model['name'],
                                            prompt=list(map(lambda x: ''.join(x), zip(prefix, target))),
                                            max_tokens=0, logprobs=1, echo=True)
        scores = []
        for i, res in enumerate(response['choices']):
            prefix_len = len(tokenizer.encode(prefix[i]))
            target_len = len(tokenizer.encode(target[i]))
            tkn_logprobs = res['logprobs']['token_logprobs']
            assert len(tkn_logprobs) == prefix_len + target_len
            target_logprob_score = sum(tkn_logprobs[prefix_len:]) / (target_len if len_norm else 1.)
            scores.append(target_logprob_score)
        model['session_tokens'] += response['usage']['total_tokens']
    else:
        bos_added = eos_added = False
        try:
            bos_added = tokenizer.add_bos_token
        except:
            pass
        try:
            eos_added = tokenizer.add_eos_token
        except:
            pass
        scores = []

        if 't5' in tokenizer.name_or_path:
            prefix_tkn = tokenizer(prefix, return_tensors="pt", padding=True)
            prefix_input_ids = prefix_tkn.input_ids
            # prefix_lens = prefix_tkn.attention_mask.sum(dim=1)
            target_tkn = tokenizer(target, return_tensors="pt", padding=True)
            target_input_ids = target_tkn.input_ids
            target_lens = target_tkn.attention_mask.sum(dim=1)

            for batch_i in range(0, len(prefix_input_ids), bsz):
                batch_prefix_input_ids = prefix_input_ids[batch_i:batch_i + bsz]
                # batch_prefix_lens = prefix_lens[batch_i:batch_i + bsz]
                batch_prefix_att_mask = prefix_tkn.attention_mask[batch_i:batch_i + bsz]
                batch_target_input_ids = target_input_ids[batch_i:batch_i + bsz]
                batch_target_lens = target_lens[batch_i:batch_i + bsz]
                # batch_target_att_mask = target_tkn.attention_mask[batch_i:batch_i + bsz]
                with torch.no_grad():
                    forward = model(input_ids=batch_prefix_input_ids.cuda(), attention_mask=batch_prefix_att_mask.cuda(),
                                    labels=batch_target_input_ids.cuda())
                for i, logits in enumerate(forward.logits):
                    target_logits = logits[:batch_target_lens[i]]
                    target_logprobs_all = torch.log_softmax(target_logits, dim=-1, dtype=torch.float)
                    target_logprobs_seq = target_logprobs_all[
                        torch.arange(batch_target_lens[i]), batch_target_input_ids[i][:batch_target_lens[i]]]
                    target_logprob_score = target_logprobs_seq.mean() if len_norm else target_logprobs_seq.sum()
                    scores.append(target_logprob_score.item())
        else:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            prefix_input_ids = tokenizer(prefix).input_ids
            target_input_ids = tokenizer(target).input_ids
            if eos_added:
                # Remove EOS from prefix and suffix
                prefix_input_ids = list(map(lambda x: x[:-1], prefix_input_ids))
                target_input_ids = list(map(lambda x: x[:-1], target_input_ids))
            if bos_added:
                # Remove BOS from suffix
                target_input_ids = list(map(lambda x: x[1:], target_input_ids))
            full_input_ids = list(map(lambda x: x[0] + x[1], zip(prefix_input_ids, target_input_ids)))
            # Add padding
            max_len = max(map(len, full_input_ids))
            # attention_mask = list(map(lambda x: [1] * len(x) + [0] * (max_len - len(x)), full_input_ids))
            full_input_ids = list(map(lambda x: x + [pad_id] * (max_len - len(x)), full_input_ids))

            for batch_i in range(0, len(full_input_ids), bsz):
                batch_input_ids = full_input_ids[batch_i:batch_i + bsz]
                # batch_attention_mask = attention_mask[batch_i:batch_i + bsz]
                with torch.no_grad():
                    forward = model(
                        torch.tensor(batch_input_ids).cuda())  # , attention_mask=torch.tensor(batch_attention_mask).cuda()
                for i, logits in enumerate(forward.logits):
                    cur_i = batch_i + i
                    target_logits = logits[
                                    len(prefix_input_ids[
                                            cur_i]) - 1:-1]  # shifted since calcs. are for the next token position
                    target_logprobs_all = torch.log_softmax(target_logits, dim=-1, dtype=torch.float)
                    target_logprobs_seq = target_logprobs_all[
                        torch.arange(len(target_input_ids[cur_i])), target_input_ids[cur_i]]
                    target_logprob_score = target_logprobs_seq.mean() if len_norm else target_logprobs_seq.sum()
                    scores.append(target_logprob_score.item())
    return scores


def get_logprob_score_from_gen(outputs, prompt, model, tokenizer, length_penalty=1., use_alt=False, verbose=False):
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
    )
    prompt_len = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).input_ids.shape[1]
    generation_lens = torch.sum(transition_scores < 0, dim=1)
    # Change the normalization of the HF scores to instead use only the length of the generated tokens
    scores = (outputs.sequences_scores * ((prompt_len + generation_lens) / generation_lens) ** length_penalty).tolist()

    scores_alt = [(torch.logsumexp(ts[:generation_lens[i]], dim=0) - torch.log(generation_lens[i])).item() for i, ts in
                  enumerate(transition_scores)]
    # TODO: Implement same alt strategy for inverse-calculations?
    return scores_alt if use_alt else scores
    # eos_added = False
    # try:
    #     eos_added = tokenizer.add_eos_token
    # except:
    #     pass
    # prefix_input_ids = tokenizer(prompt).input_ids
    # if eos_added:
    #     # Remove EOS from prefix
    #     prefix_input_ids = list(map(lambda x: x[:-1], prefix_input_ids))
    #
    # input_len = len(prefix_input_ids)
    # stop_beams = set()
    # n_steps = len(outputs.scores)
    # n_beams = outputs.scores[0].shape[0]
    # beam_scores = defaultdict(list)
    #
    # stop_chars = ['\n', '<0x0A>']
    # if '_unk_token' in tokenizer.__dict__:
    #     unk = tokenizer.__dict__['_unk_token']
    #     if type(unk) is AddedToken:
    #         unk = unk.content
    #     stop_chars.append(unk)
    # if '_eos_token' in tokenizer.__dict__:
    #     eos = tokenizer.__dict__['_eos_token']
    #     if type(eos) is AddedToken:
    #         eos = eos.content
    #     stop_chars.append(eos)
    #
    # for step_idx in range(n_steps):
    #     step_scores = outputs.scores[step_idx]
    #     _seqs = [''] * n_beams
    #     for beam_idx in range(n_beams):
    #         if beam_idx in stop_beams:
    #             continue
    #         token = outputs.sequences[beam_idx][input_len + step_idx]
    #         decoded_token = tokenizer.decode(token)
    #         if decoded_token in stop_chars:
    #             if verbose:
    #                 logger.info(f"Stopped beam {beam_idx}")
    #             stop_beams.add(beam_idx)
    #             continue
    #         _seqs[beam_idx] = decoded_token
    #         _beam_scores = step_scores[beam_idx]
    #         step_beam_score = _beam_scores[outputs.sequences[beam_idx][input_len + step_idx]].item()
    #         if step_beam_score == float('-inf'):
    #             if verbose:
    #                 logger.info(f"Stopped beam {beam_idx}")
    #             stop_beams.add(beam_idx)
    #             continue
    #         beam_scores[beam_idx].append(step_beam_score)
    #         if decoded_token in ['?']:
    #             if verbose:
    #                 logger.info(f"Stopped beam {beam_idx}")
    #             stop_beams.add(beam_idx)
    #     if verbose:
    #         logger.info(f'Step {step_idx}\t', _seqs)
    # scores = [np.mean(beam_scores[b]) for b in beam_scores]
    # return scores


def prompt_arr_2_text(prompt_arr, prompt_sep, is_llama2, is_beluga, is_chat, output_prefix):
    # Output prefix (final prompt text)
    if output_prefix is None:
        output_prefix = copy.copy(default_output_prefix)
    if is_llama2 and is_chat:
        prompt = f"{llama_chat_B_SYS}{llama_chat_DEFAULT_SYSTEM_PROMPT}{llama_chat_E_SYS}"
        prompt += f"User: {(prompt_sep.join(prompt_arr)).strip()}"
        prompt = f"{llama_chat_B_INST} {prompt} {llama_chat_E_INST}"
        if output_prefix != "":
            prompt += f" {output_prefix}"
    else:
        if output_prefix != "":
            prompt_arr.append(output_prefix)
        if is_beluga and is_chat:
            prompt = f"{beluga_DEFAULT_SYSTEM_PROMPT}### User:\n{(prompt_sep.join(prompt_arr))}"
        else:
            prompt = prompt_sep.join(prompt_arr)
    return prompt
