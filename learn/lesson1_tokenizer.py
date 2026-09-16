"""第1课：Tokenizer 与数据 —— 文本如何变成数字，模型到底在学什么
运行方式（任意目录均可，如）：python learn/lesson1_tokenizer.py
"""
import os, sys, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
random.seed(0)

from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset, SFTDataset

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))

print('=' * 30, 'Part 1: Tokenizer 基础', '=' * 30)
print(f'词表大小 vocab_size = {tok.vocab_size}')
print(f'特殊token:  pad={tok.pad_token!r}(id={tok.pad_token_id}), '
      f'bos={tok.bos_token!r}(id={tok.bos_token_id}), eos={tok.eos_token!r}(id={tok.eos_token_id})')

for text in ['我爱秋天', '魑魅魍魉是很生僻的词', 'unbelievable ability']:
    ids = tok(text, add_special_tokens=False).input_ids
    toks = [tok.decode([i]) for i in ids]
    print(f'\n原文: {text!r}  ({len(text)}字符 -> {len(ids)}个token)')
    print('  切分结果:', ' | '.join(repr(t) for t in toks))
    print('  还原check:', repr(tok.decode(ids)) == repr(text))

print()
print('=' * 30, 'Part 2: 对话模板（SFT 数据的原始形态）', '=' * 30)
demo = [{'role': 'user', 'content': '你叫什么名字？'},
        {'role': 'assistant', 'content': '我是minimind。'}]
print(tok.apply_chat_template(demo, tokenize=False, add_generation_prompt=False))

print('=' * 30, 'Part 3: 预训练数据 PretrainDataset', '=' * 30)
ds = PretrainDataset(os.path.join(ROOT, 'dataset/pretrain_smoke.jsonl'), tok, max_length=340)
x, y = ds[0]
n_pad = int((x == tok.pad_token_id).sum())
n_real = len(x) - n_pad
print(f'样本形状: {tuple(x.shape)}  真实token数={n_real}, 填充pad数={n_pad}')
print(f'前8个位置: {[(int(i), repr(tok.decode([i]))) for i in x[:8]]}')
print(f'真实部分 labels == input_ids ? {bool((y[:n_real] == x[:n_real]).all())}')
print(f'pad部分 labels 是否全部为-100 ? {bool((y[n_real:] == -100).all())}')
print('\n"错位预测"示意（前6对）—— 模型在位置i看到X，要对Y打分:')
for i in range(6):
    print(f'  看到 {repr(tok.decode([x[i]])):>14} -> 预测 {repr(tok.decode([x[i+1]])):>14}  (label={int(y[i+1])})')

print()
print('=' * 30, 'Part 4: SFT 数据 SFTDataset（关键的 -100 掩码）', '=' * 30)
ds2 = SFTDataset(os.path.join(ROOT, 'dataset/sft_smoke.jsonl'), tok, max_length=768)
idx, best = None, None
for i in range(300):
    xx, yy = ds2[i]
    real = int((yy != -100).sum())
    if 40 <= real <= 300:
        idx = i
        break
    if best is None or real < int((ds2[best][1] != -100).sum()):
        best = i
if idx is None: idx = best
x2, y2 = ds2[idx]
n_learn = int((y2 != -100).sum())
n_mask = int((y2 == -100).sum())
print(f'选了第 {idx} 条样本, 总长768, 计loss(学习)的位置={n_learn}, 屏蔽(-100)的位置={n_mask}')

segs, start, cur = [], 0, bool(y2[0] == -100)
for i in range(1, len(x2)):
    m = bool(y2[i] == -100)
    if m != cur:
        segs.append((start, i, cur)); start, cur = i, m
segs.append((start, len(x2), cur))
print('\n按"学习/屏蔽"切开的全文（屏蔽段=模型只需阅读, 学习段=模型要学会生成）:')
for s, e, m in segs:
    t = tok.decode(x2[s:e]).replace('\n', '\\n')
    if len(t) > 70: t = t[:70] + f'...(共{e-s}个token)'
    print(f'  [{"屏蔽" if m else "学习"}] {t!r}')

print('\n学习段内的"错位预测"（前8对）—— 看到X, 要学会预测Y:')
shown = 0
for j in range(1, len(x2)):
    if y2[j] != -100:
        print(f'  看到 {repr(tok.decode([x2[j-1]])):>16} -> 预测 {repr(tok.decode([x2[j]])):>16}')
        shown += 1
        if shown >= 8: break
