import ctranslate2
import pyonmttok
from typing import List
import jieba
import time
from tools.apply_bpe import BPE
from process_tag import process_tag
import re
from collections import defaultdict

CONFIG_TRANSLATOR = {
    'device':"auto",            # The device to use: "cpu", "cuda", or "auto".
    "device_index":0,           # The index of the device to place this translator on.
    "compute_type":"default",   # The computation type: "default", "int8", "int16", or "float".
    "inter_threads":1,          # Maximum number of concurrent translations (CPU only).
    "intra_threads":20           # Threads to use per translation (CPU only). 
}

CONFIG_TRANSLATE = {
    "target_prefix":None,           # An optional list of list of string.
    "max_batch_size":0,             # Maximum batch size to run the model on.
    "batch_type":"examples",        # Whether max_batch_size is the number of examples or tokens.
    "beam_size":5,                  # Beam size (set 1 to run greedy search).
    "num_hypotheses":1,             # Number of hypotheses to return (should be <= beam_size).
    "length_penalty":0,             # Length penalty constant to use during beam search.
    "coverage_penalty":0,           # Converage penalty constant to use during beam search.
    "max_decoding_length":250,      # Maximum prediction length.
    "min_decoding_length":1,        # Minimum prediction length.
    "use_vmap":False,               # Use the vocabulary mapping file saved in this model.
    "return_scores":True,           # Include the prediction scores in the output.
    "return_attention":False,       # Include the attention vectors in the output.
    "return_alternatives":False,    # Return alternatives at the first unconstrained decoding position.
    "sampling_topk":1,              # Randomly sample predictions from the top K candidates (with beam_size=1).
    "sampling_temperature":1        # Sampling temperature to generate more random samples.
}


def remove_repeat_token(sentence, max_ngram_length = 4):
    final_merge_sent = sentence.split(' ')
    if len(final_merge_sent) < 3:
        return sentence
    max_ngram_length = min(max_ngram_length, len(sentence))
    for i in range(max_ngram_length, 0, -1):
        start = 0
        end = len(final_merge_sent) - i + 1
        ngrams = []
        while start < end:
            ngrams.append(final_merge_sent[start: start + i])
            start += 1
        result = []
        for cur_word in ngrams:
            result.append(cur_word)
            if len(result) > i:
                pre_word = result[len(result) - i - 1]
                if pre_word == cur_word:
                    for k in range(i):
                        result.pop()

        cur_merge_sent = []
        for word in result:
            if not cur_merge_sent:
                cur_merge_sent.extend(word)
            else:
                cur_merge_sent.append(word[-1])
        final_merge_sent = cur_merge_sent

    return ' '.join(final_merge_sent)

def cut_sent(text:str, pattern="([。])"):
    sents = re.split(pattern, text) + [""]
    sents = ["".join(i) for i in zip(sents[0::2],sents[1::2])]
    return sents


class Server:
    def __init__(self, path_translator, path_bpe):
        self.translator = ctranslate2.Translator(path_translator, **CONFIG_TRANSLATOR)
        self.bpe = BPE(open(path_bpe, 'r'))

    def predict(self, text_list: List[str]) -> List[str]:
        t1 = time.time()
        _text_list, _id = [], []
        for i, text in enumerate(text_list):
            text = cut_sent(text)
            _id.extend([i] * len(text))
            _text_list.extend(text)
        
        text_list = _text_list

        text_list = [' '.join([e.strip() for e in jieba.lcut(line) if e.strip()]) for line in text_list]
        t2 = time.time()
        text_list = [process_tag(self.bpe.segment(text).strip()).split(' ') for text in text_list]
        totol_tokens = sum([len(e) for e in text_list])
        t3 = time.time()
        print(f'input: {text_list}')
        result = self.translator.translate_batch(text_list, **CONFIG_TRANSLATE)
        t4 = time.time()
        result = [' '.join(d[0]["tokens"]).replace('@@ ', '') for d in result]
        print(f'output: {result}')
        result = [remove_repeat_token(text) for text in result]
        print(f'remove repeat: {result}')
        
        _result = defaultdict(list)
        for i, text in zip(_id, result):
            print(i, text)
            _result[i].append(text)
        
        result = [" ".join(v) for k, v in _result.items()]

        t5 = time.time()
        
        report = f'''
        jieba:  {t2-t1}s
        bpe:    {t3-t2}s
        trans:  {t4-t3}s    {totol_tokens/(t4-t3)} tokens/s 
        post:   {t5-t4}s
        '''
        #print(text_list)
        print(report)
        return result



if __name__ == '__main__':
    from tqdm import tqdm
    def cut_list(data, batch_size):
        return [data[x:x+batch_size] for x in range(0, len(data), batch_size)]

    path_translator = "zh2en_ctranslate2"
    path_bpe="zh.bpe"
    
    server = Server(path_translator, path_bpe)

    dataset = open('../data/zh-val.txt').read().split('\n')
    outf = open('server_val.out', 'w')

    for text in tqdm(cut_list(dataset, 100)):
        result = server.predict(text)
        print('\n'.join(result), file=outf)
