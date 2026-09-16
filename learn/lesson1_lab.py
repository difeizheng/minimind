"""第1课 实操实验室：先预测 -> 改代码 -> 跑起来 -> 对答案
用法：python learn/lesson1_lab.py（任意目录均可）
每个实验带 [TODO] 标记。默认可以直接跑通——先跑一遍看结构，再按 lesson1_exercises.md 逐题修改。
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import AutoTokenizer
from dataset.lm_dataset import SFTDataset

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))

print('=' * 25, '实验1: Tokenizer 探索', '=' * 25)
# [TODO 1a] 往列表里加：你的名字 / 手机号 / 一个emoji / 一个很长的英文单词
# [TODO 1b] 运行前先猜：每个字符串会被切成几个 token？跑完对答案
text_list = ['我爱秋天', '人工智能', 'Hello World', '🚀', '13800138000','zhengdifei','郑涤非','welcome beijing']
for text in text_list:
    ids = tok(text, add_special_tokens=False).input_ids
    toks = [tok.decode([i]) for i in ids]
    print(f'{text!r}: {len(text)}字符 -> {len(ids)}token（{len(text) / len(ids):.2f}字符/token）: '
          + ' | '.join(repr(t) for t in toks))

print()
print('=' * 25, '实验2: SFT 样本解剖', '=' * 25)
# [TODO 2a] 换 3 个不同的 idx（范围 0~19999），记录"学习占比"分别是多少
# [TODO 2b] 找一条回答里带 <think> 的样本：思考内容在学习（要考）的范围内吗？
idx = 890
ds = SFTDataset(os.path.join(ROOT, 'dataset/sft_smoke.jsonl'), tok, max_length=768)
x, y = ds[idx]
n_learn, n_mask = int((y != -100).sum()), int((y == -100).sum())
print(f'idx={idx}: 总长{len(x)} | 学习(要考)={n_learn}({n_learn / len(x):.0%}) | 屏蔽(-100,不考)={n_mask}')
print('前3个「看到 -> 预测」配对:')
shown = 0
for j in range(1, len(x)):
    if y[j] != -100:
        print(f'   位置{j:>3}: 看到 {tok.decode([x[j - 1]])!r} -> 预测 {tok.decode([x[j]])!r}')
        shown += 1
        if shown == 3: break

print()
print('=' * 25, '实验3: 截断与填充', '=' * 25)
# [TODO 3a] 在列表里加 32。先预测：考卷被裁到32个token，还有题可做吗？
for ml in [32, 64, 256, 768]:
    ds2 = SFTDataset(os.path.join(ROOT, 'dataset/sft_smoke.jsonl'), tok, max_length=ml)
    xx, yy = ds2[0]
    n_learn = int((yy != -100).sum())
    eos_learned = any(yy[i] != -100 and xx[i] == tok.eos_token_id for i in range(len(xx)))
    print(f'max_length={ml:4d}: 学习位置={n_learn:4d}  回答结尾<|im_end|>被考到={eos_learned}')
