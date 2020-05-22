import jieba
from tqdm import tqdm
import random

if __name__ == '__main__':
    path_en = '/data/translate/hb_trans/all.en'
    path_zh = '/data/translate/hb_trans/all.zh'

    #data_en = open(path_en, 'r').read().split('\n')[:1000]
    #data_zh = open(path_zh, 'r').read().split('\n')[:1000]
    
    data_en = open(path_en, 'r').read().split('\n')
    data_zh = open(path_zh, 'r').read().split('\n')

    data_en = [[e.strip() for e in jieba.lcut(line) if e.strip()] for line in tqdm(data_en, 'load en')]
    data_zh = [[e.strip() for e in jieba.lcut(line) if e.strip()] for line in tqdm(data_zh, 'load zh')]

    data = [(' '.join(en), ' '.join(zh)) for en, zh in zip(data_en, data_zh) \
            if len(en) * len(zh) and 1/1.3 < len(en)/ len(zh) < 1.3]
    random.shuffle(data)

    start_point = [int(e*len(data)) for e in [0, 0.05, 0.10]] + [None]
    datasets = [[], [], []]
    for i, dataset in enumerate(datasets):
        dataset.extend(data[start_point[i]:start_point[i+1]])
    
    outf = {}
    for k in ['dev', 'test', 'train']:
        for type in ['src', 'tgt']:
            name = f'{type}-{k}.txt'
            outf[name] = open(name, 'w')

    for k, v in zip(['dev', 'test', 'train'], datasets):
        for en, zh in tqdm(v, f'save {k}'):
            print(en, file=outf[f'src-{k}.txt'])
            print(zh, file=outf[f'tgt-{k}.txt'])
    

