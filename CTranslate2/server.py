import ctranslate2
import pyonmttok
from typing import List
import jieba
import time

CONFIG_TRANSLATOR = {
    'device':"auto",            # The device to use: "cpu", "cuda", or "auto".
    "device_index":0,           # The index of the device to place this translator on.
    "compute_type":"default",   # The computation type: "default", "int8", "int16", or "float".
    "inter_threads":1,          # Maximum number of concurrent translations (CPU only).
    "intra_threads":10           # Threads to use per translation (CPU only). 
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
        self.tokenizer = pyonmttok.Tokenizer("none", bpe_model_path=path_bpe)

    def predict(self, text_list: List[str]) -> List[str]:
        tic = time.time()
        
        text_list = [' '.join([e.strip() for e in jieba.lcut(line) if e.strip()]) for line in text_list]
        text_list = [[tok for tok in self.tokenizer.tokenize(text)[0] if tok.strip()] for text in text_list]
        totol_tokens = sum([len(e) for e in text_list])
        result = self.translator.translate_batch(text_list, **CONFIG_TRANSLATE)
        result = [' '.join(d[0]["tokens"]).replace('@@ ', '') for d in result]
        
        toc = time.time()
        print(f'{totol_tokens/(toc-tic)} tokens/s')
        return result



if __name__ == '__main__':
    path_translator = "zh2en_ctranslate2"
    path_bpe="zh.bpe"
    
    server = Server(path_translator, path_bpe)

    text_list = """
    加州大学洛杉矶分校 ( University of California , Los Angeles ) 的 伊斯兰教 法学 教授 阿布 · 法德勒 在 一次 录音 采访 中 表示 ， “ 中国 的 清真 女寺 传统 源于 伊斯兰教 的 历史 ， 这 并 不 新奇 ， 也 不是 什么 堕落 、 创新 或者 异端 的 做法 。 ”
    在 中科院 ， 他 当过 一段时间 的 审查员 ， 工作 是 清除 研究 论文 中 任何 不可 接受 的 观点 和 敏感 信息 。 他 儿子 说 ， 他 后来 骄傲地 谈到 自己 让 许多 论文 被 送 至 国外 发表 了 。
    《 读卖新闻 》 数月 来 一直 在 抨击 自由派 的 主要 对手 《 朝日新闻 》 ( Asahi Shimbun ) ， 指责 它 在 涉及 二战 期间 日军 性侵 行为 的 报道 中 的 错误 。 上 世纪 80 、 90 年代 ， 《 朝日新闻 》 撰写 的 一些 文章 称 ， 一名 男子 声称 在 战争 中 绑架 了 一些 朝鲜 妇女 ， 并 强迫 她们 进入 了 日本 军队 的 妓院 。 这名 男子 最终 被 证明 是 在 说谎 ， 《 朝日新闻 》 今年 8 月 撤回 了 相关 报道 ， 《 读卖新闻 》 因此 谴责 该报 犯 了 “ 极其 严重 ” 的 错误 ， 破坏 了 日本 的 国际 声誉 。
    反对 的 声音 在 中国 令人 惊讶 地 普遍 ， 即使 在 主张 改革 的 中国 经济学家 中 也 如此 。 其 背后 有 一个 很 好 的 理由 ： 如果 将 经济 理论 和 新兴 市场 国家 的 实践经验 考虑 进来 ， 中国 正在 用 完全 错误 的 方法 进行 改革 。
    他 提出 新加坡 也许 可以 “ 建造 地下 交通枢纽 、 步行街 、 自行车道 、 公共设施 、 仓储 和 研究 设施 、 工业 应用 、 购物 区 和 其他 公共 空间 。 ”
    “ 事实 是 ， 大多数 人 韩国 人 在 感情 上 都 不能 接受 这个 协议 ， ” 文在寅 的 办公室 援引 他 的话 说 。 但 他 没有 表示 他 想要 废除 这个 协议 。
    1882 年 ， 皮埃尔 · 奥古斯特 · 雷诺阿 ( Pierre - Auguste Renoir ) 来 探访 塞尚 ， 二人 携手 作画 。 那 一年 ， 雷诺阿 用 他 特色 的 朦胧 画风 在 作品 《 埃斯塔克 的 峭壁 》 ( Rocky Crags at L ’ Estaque ) 中 刻画 了 此地 的 草木 山川 。 塞尚 总是 更 清瘦 、 更 热烈 ， 绘画 时 就 像  在 孜孜不倦 地 解答 数学 难题 。 每 一笔 、 每 一幅 画 ， 都 在 揭示 自己 如何 得出 了 结论 。 每 一位 画家 都 有 独特 的 视角 。 1908 年 ， 劳尔 · 杜飞 ( Raoul Dufy ) 在 巴黎 看过 朋友 布拉克 的 杰作 之后 ， 与 后者 一起 来到 这里 作画 。 野兽派 画家 安德烈 · 德兰 ( Andr é Derain ) 再现 了 这里 五光十色 的 快乐 港口 景象 ， 相形之下 ， 塞尚 的 作品 顿显 忧伤 。
    习近平 号召 回到 办公室 、 学校 、 社群 作为 政党 崇拜 的 活跃 中心 ， 千家万户 团结 在 单一 领导者 之下 的 时代 。 他 以 让 人们 想起 毛 时代 个人崇拜 的 方式 ， 把 自己 塑造成 一个 变革 者 。
    Office Mobile 是 Office 365 付费 用户 的 一个 不错 的 福利 。 它 的 设计 简单 、 流畅 。 学习 起来 很 容易 。 很难 想象 一个 “ 苗条 可人 ” 如 “ 赫特人 贾巴 ” ( Jabba the Hutt ) 的 软件 套件 会 有 这么 个 远房亲戚 。 但是 作为 微软 Office 的 iPhone 版 ， 它 的 局限性 简直 到 了 荒唐 的 地步 。
    多年 来 ， 中国 相关 书籍 的 读者 看到 越来越 多 有关 共产主义 革命 — — 从 1949 年 夺取 政权 到 1989 年 军队 镇压 天安门 抗议者 — — 的 残暴 性 和 破坏性 。 这 两本书 通过 真实 、 哀伤 的 文字 呈现 了 比 我们 想象 中 更糟 的 情景 。
    """.replace(' ', '').split('\n')

    result1 = server.predict(text_list)
    result2 = server.predict(text_list * 10)
    print(result1)
