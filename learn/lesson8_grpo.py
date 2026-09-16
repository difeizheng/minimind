"""第8课：GRPO —— 考卷消失，自己答题，裁判打分
用法：python learn/lesson8_grpo.py（任意目录均可）
灵魂实验：亲手执行一个完整的 GRPO 步骤——采样6个答案→打分→组内比较→算优势。
（简化说明：打分只用 train_grpo.py:54-60 的规则分，真实训练还叠加奖励模型打分）
"""
import os, sys, re, random
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch.manual_seed(42); random.seed(42)

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
model.load_state_dict(torch.load(os.path.join(ROOT, 'out/full_sft_768.pth'), map_location='cpu'))
model.eval().to(device)

print('=' * 20, 'Part 0: RL 考卷 —— 只有问题，没有答案', '=' * 20)
ds = RLAIFDataset(os.path.join(ROOT, 'dataset/rlaif_smoke.jsonl'), tok, thinking_ratio=1.0)
s = ds[0]
print(f'数据集返回的字典: {list(s.keys())}')
print(f"answer 字段的值: {s['answer']!r}   ← 空的！没有人告诉模型该说什么")
print('对比: SFT数据(input_ids+labels) / DPO数据(好答案+坏答案) / RL数据(只有题目)')
print('没有答案怎么学？—— 自己写答案，裁判打分，分数说话。')

print()
print('=' * 20, 'Part 1: 亲手做一步 GRPO', '=' * 20)
G = 6  # 每个 prompt 采样几个答案（train_grpo.py 默认 --num_generations 6）
prompt = s['prompt']
print(f'题目(prompt, 末尾已摆好<think>草稿纸): ...{prompt[-60:]!r}')
ids = tok(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
mask = torch.ones_like(ids)
with torch.no_grad():
    out = model.generate(ids, attention_mask=mask, max_new_tokens=120,
                         temperature=0.8, top_p=0.85, top_k=50,
                         num_return_sequences=G, do_sample=True)
answers = [tok.decode(out[i, ids.shape[1]:], skip_special_tokens=False) for i in range(G)]

# ---- 裁判打分（照抄 train_grpo.py:31-60 的规则部分，不含奖励模型） ----
def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

def rule_reward(resp):
    r = 0.5 if 20 <= len(resp.strip()) <= 800 else -0.5      # 长度规矩
    if '</think>' in resp:
        think, ans = resp.split('</think>', 1)
        r += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5  # 思考段长度
        r += 0.25 if resp.count('</think>') == 1 else -0.25    # 思考段完整
    r -= rep_penalty(resp)                                     # 重复扣分
    return r

rewards = torch.tensor([rule_reward(a) for a in answers])
print(f'\n模型自己写的 {G} 份答案 + 裁判打分:')
for i, (a, r) in enumerate(zip(answers, rewards.tolist())):
    show = a.replace('\n', '\\n')[:50]
    print(f'  答案{i}: {show!r:52s} 得分 {r:+.2f}')

print()
print('=' * 20, 'Part 2: GRPO 的核心动作 —— 组内比出高下', '=' * 20)
mean, std = rewards.mean(), rewards.std(unbiased=False)
advantages = (rewards - mean) / (std + 1e-4)   # train_grpo.py:121-124
print(f'组内均分 {mean:.2f}, 标准差 {std:.2f}')
print(f'优势(advantage) = (得分-均分)/标准差:')
for i, a in enumerate(advantages.tolist()):
    tag = '加强!整个答案每个字都被推高' if a > 0 else '抑制!整个答案每个字都被压低'
    print(f'  答案{i}: 优势 {a:+.2f}  → {tag}')

print()
print('更新规则(train_grpo.py:135-143):')
print('  loss = -优势 × logP(答案的每个token)   (+ 与参考模型的KL缰绳)')
print('  直觉: 优势为正的答案,把其中每个字的赌注都推高一点;')
print('        优势为负的答案,把每个字的赌注都压低一点。')
print('没有逐字纠错——只有"这整个答案值得加强/抑制"的成败论英雄。')
print('这就是RL与SFT的分水岭: SFT照字帖写字,RL按结果赏罚。')

print()
print('=' * 20, 'Part 3: 两个工程真相', '=' * 20)
print('1. 为什么组内比较(叫Group Relative)?')
print('   题目有难有易,绝对分没意义(全班平均30分的卷子,35分就是好学生)。')
print('   组内相对化后,不需要额外训练一个"难度评估器"(critic),省一半显存——')
print('   这就是GRPO比PPO好伺候的原因,DeepSeek用它训练出了R1。')
print('2. 奖励会被钻空子(reward hacking):')
print('   规则是"长度20~800字加分"→模型学会凑字数;')
print('   "有一个</think>加分"→学会摆个空思考架子。')
print('   真实训练还要叠加奖励模型打分(本机internlm2-1_8b-reward),')
print('   但规则一旦可被套路,模型就会认真钻营——设计裁判比设计学生难。')
