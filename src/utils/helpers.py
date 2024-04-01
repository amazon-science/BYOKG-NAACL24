import atexit
import logging
import os
import re
import sys


def setup_logger(run_id, log_dir='./logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_fname = f"{log_dir}/{run_id}.log"
    logger = logging.getLogger()  # get root logger
    file_handler = logging.FileHandler(log_fname, mode="a", delay=False)
    file_handler.setFormatter(
        logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)  # all other loggers propagate to root; write to one log file from root
    print(f"Log path: {log_fname}")
    atexit.register(lambda: print(f"Log path: {log_fname}"))


class Dict2Obj(object):
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


def rename_key_in_dict(dictionary, oldkey, newkey):
    if type(dictionary) is dict:
        keys = list(dictionary)
        for k in keys:
            if k == oldkey:
                dictionary[newkey] = dictionary.pop(oldkey)
                rename_key_in_dict(dictionary[newkey], oldkey, newkey)
            else:
                rename_key_in_dict(dictionary[k], oldkey, newkey)
    elif type(dictionary) is list:
        for item in dictionary:
            rename_key_in_dict(item, oldkey, newkey)


def split_underscore_period(string, keep_period_groups=None):
    final = []
    period_groups = string.split('.')
    if keep_period_groups is not None:  # all
        period_groups = period_groups[-keep_period_groups:]
    for g in period_groups:
        final += g.split('_')
    return " ".join(final)


def remove_special_characters(input_string):
    pattern = re.compile('[^A-Za-z0-9 ]+')
    result_string = pattern.sub('', input_string)
    return result_string


def merge_quotes(tkns: list):
    merge_tups = []
    start = False
    for i, t in enumerate(tkns):
        if t.count('"') % 2 == 1:
            start = not start
            if start:
                merge_tups.append((i, None))
            else:
                merge_tups[-1] = (merge_tups[-1][0], i)
    new_tkns = []
    last = 0
    for tup in merge_tups:
        if tup[1] is None:
            # this means there is some string that also has a quotation inside it (e.g. to represent inches "5\"")
            # in this case, merge this token with the last ones until we find any quotation
            # raise ValueError(f"Closing quotations not found at index {tup[0]} for input: {tkns}")
            _find_idx = tup[0] - 1
            while _find_idx >= 0:
                if '"' in tkns[_find_idx]:
                    break
                _find_idx -= 1
            if _find_idx < 0:
                raise ValueError(f"Closing quotations not found at index {tup[0]} for input: {tkns}")
            new_tkns += tkns[last:_find_idx]
            new_tkns.append(" ".join(tkns[_find_idx:tup[0] + 1]))
            last = tup[0] + 1
        else:
            new_tkns += tkns[last:tup[0]]
            new_tkns.append(" ".join(tkns[tup[0]:tup[1] + 1]))
            last = tup[1] + 1
    new_tkns += tkns[last:]
    return new_tkns


def deep_get(obj, *keys, default={}):
    rtn = obj
    if type(rtn) is not dict:
        return default
    for k in keys:
        rtn = rtn.get(k, default)
        if type(rtn) is not dict:
            return rtn
    return rtn
