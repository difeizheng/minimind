"""第1课·补完：亲手跑一次"训练循环"——做题、打分、改数字
运行：python learn/lesson1b_training_loop.py （任意目录均可）
这个脚本 = 所有 train_*.py 的核心骨架的迷你版，看懂它就看懂了"训练"本身。
"""
import os, sys, random
import torch
from torch.utils.data import Subset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
random.seed(0)
torch.manual_seed(42)

from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== 1) 造一个"从没学过中文"的模型：参数全是随机数 ==========
cfg = MiniMindConfig(hidden_size=256, num_hidden_layers=4)
model = MiniMindForCausalLM(cfg).to(device)
print(f'小模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M（全是随机数，此刻它是个"文盲"）')

# 推理 = 训练时同一个猜字游戏：猜一个字，接上去，再猜
@torch.no_grad()
def greedy_generate(model, prompt, max_new_tokens=30):
    ids = [tok.bos_token_id] + tok(prompt, add_special_tokens=False).input_ids
    x = torch.tensor([ids], device=device)
    for _ in range(max_new_tokens):
        nxt = model(x).logits[0, -1].argmax().view(1, 1)
        if nxt.item() == tok.eos_token_id: break
        x = torch.cat([x, nxt], dim=1)
    return tok.decode(x[0, len(ids):])

print('训练前让它接龙「我爱秋天」:', repr(greedy_generate(model, '我爱秋天')))

# ========== 2) 出考卷：就是 lesson1 里的 PretrainDataset ==========
ds = Subset(PretrainDataset(os.path.join(ROOT, 'dataset/pretrain_smoke.jsonl'), tok, max_length=128), range(400))
loader = DataLoader(ds, batch_size=16, shuffle=True)

# ========== 3) 训练循环：做题 -> 打分 -> 改数字，重复 ==========
import math
print(f'\n参考：纯乱猜(6400选1)的理论loss = ln(6400) ≈ {math.log(6400):.2f}')
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
step = 0
for epoch in range(3):
    for x, y in loader:
        step += 1
        x, y = x.to(device), y.to(device)
        res = model(x, labels=y)
        loss = res.loss + res.aux_loss   # aux_loss 是 MoE 专用，这里恒为 0
        loss.backward()                  # 根据错题算出"每个参数该往哪边改"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()                       # 真的去改那几百万个数字
        opt.zero_grad(set_to_none=True)
        if step == 1 or step % 10 == 0:
            print(f'step {step:3d}   loss = {res.loss.item():.4f}')

# ========== 4) 看变化 ==========
print(f'\n共刷了 {step} 道题（每道题=16条×127个位置≈2000个"猜下一字"小题）')
print('训练后再接龙「我爱秋天」:', repr(greedy_generate(model, '我爱秋天')))
print('训练后再接龙「给我生成一首」:', repr(greedy_generate(model, '给我生成一首')))
