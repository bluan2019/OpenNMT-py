import jieba
from tqdm import tqdm
import random
import logging
jieba.setLogLevel(logging.INFO)

from multiprocessing import Pool

def cut_list(data, batch_size):
    return [data[x:x+batch_size] for x in range(0, len(data), batch_size)]

def parallel(func: callable, data:list, num_worker:int=3, debug=False) -> callable:
    if num_worker == 1:
        return func(data)

    batch_size = len(data)//num_worker + 1
    with Pool(num_worker) as p:
        if debug: print('parallel debug M:', cut_list(data, batch_size))
        res = p.map(func, cut_list(data, batch_size))
        if debug: print('parallel debug R:', res)
    result = []
    for e in res: result.extend(e)
    return result


if __name__ == '__main__':
    path_en  = '/data/lbh/translate/wmt/en-zh/UNv1.0.en-zh.en'
    path_zh  = '/data/lbh/translate/wmt/en-zh/UNv1.0.en-zh.zh'
    path_out = "data/"
    
    data_pair = list(zip(open(path_en, 'r').read().split('\n'),
                         open(path_zh, 'r').read().split('\n')))
    print(f'total len: {len(data_pair)}')
    data_pair = set(data_pair)
    print(f'remove duplicate: {len(data_pair)}')

    data_en, data_zh = zip(*data_pair)
    del data_pair
   
    def cut_sent(data):
        return [[e.strip() for e in jieba.lcut(line) if e.strip()] for line in tqdm(data)]
    
    data_en = parallel(cut_sent, data_en, num_worker=40)
    data_zh = parallel(cut_sent, data_zh, num_worker=40)

    data = [(' '.join(en), ' '.join(zh)) for en, zh in zip(data_en, data_zh) \
            if len(en) * len(zh) and 1/1.3 < len(en)/ len(zh) < 1.3]
    random.shuffle(data)
    print(f'remove 1.3: {len(data)}')

    start_point = [int(e*len(data)) for e in [0, 0.05, 0.10]] + [None]
    datasets = [[], [], []]
    for i, dataset in enumerate(datasets):
        dataset.extend(data[start_point[i]:start_point[i+1]])

    print(f'dataset len: {[len(e) for e in datasets]}')
    
    outf = {}
    for k in ['dev', 'test', 'train']:
        for type in ['en', 'zh']:
            name = f'{type}-{k}.txt'
            outf[name] = open(f'{path_out}/{name}', 'w')

    for k, v in zip(['dev', 'val', 'train'], datasets):
        for en, zh in tqdm(v, f'save {k}'):
            print(en, file=outf[f'en-{k}.txt'])
            print(zh, file=outf[f'zh-{k}.txt'])
    

