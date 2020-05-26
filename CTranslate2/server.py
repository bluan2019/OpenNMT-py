import ctranslate2
import pyonmttok
from typing import List
import jieba
import time
from tools.apply_bpe import BPE

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

class Server:
    def __init__(self, path_translator, path_bpe):
        self.translator = ctranslate2.Translator(path_translator, **CONFIG_TRANSLATOR)
        self.bpe = BPE(open(path_bpe, 'r'))

    def predict(self, text_list: List[str]) -> List[str]:
        t1 = time.time()
        text_list = [' '.join([e.strip() for e in jieba.lcut(line) if e.strip()]) for line in text_list]
        t2 = time.time()
        text_list = [self.bpe.segment(text).strip().split(' ') for text in text_list]
        totol_tokens = sum([len(e) for e in text_list])
        t3 = time.time()
        result = self.translator.translate_batch(text_list, **CONFIG_TRANSLATE)
        t4 = time.time()
        result = [' '.join(d[0]["tokens"]).replace('@@ ', '') for d in result]
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
