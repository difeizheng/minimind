"""第6课：DPO —— 不教模型说什么，教它偏爱哪个
用法：python learn/lesson6_dpo.py（任意目录均可）
灵魂实验：同一批(好答案, 坏答案)对，测 full_sft 和官方 dpo 两个权重的
"偏好间距" logP(好) - logP(坏)，看 DPO 训练如何把它拉开。
"""
import os, sys, math
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch.manual_seed(42)

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import DPODataset

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load(weight_name):
    m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    m.load_state_dict(torch.load(os.path.join(ROOT, f'out/{weight_name}_768.pth'), map_location='cpu'))
    return m.eval().to(device)

print('=' * 22, 'Part 0: DPO 考卷长什么样', '=' * 22)
print('每道题 = 同一个问题 + 两个完整答案:')
print('  chosen  (人类更喜欢的答案)')
print('  rejected(较差的答案)')
print('没有"正确答案"，只有"更好的答案"——这是和SFT考卷的本质区别。')

print()
print('=' * 22, 'Part 1: DPO 的损失函数在算什么', '=' * 22)
print('先定义"间距": margin = [logP(好)-logP(坏)]_当前模型 - [logP(好)-logP(坏)]_参考模型')
print('损失: loss = -log sigmoid(beta * margin)   (train_dpo.py:34-50)')
print()
print('margin    sigmoid    loss     解读')
for m in [-2, -1, 0, 1, 2, 4]:
    sig = 1 / (1 + math.exp(-0.15 * m))
    print(f'{m:6d}   {sig:7.1%}   {-math.log(sig):6.3f}   '
          f'{"模型偏爱坏答案(扣分重)" if m < 0 else "无偏好(中性)" if m == 0 else "偏爱好答案(扣分轻)"}')
print()
print('注意 margin=0 时 loss=0.693=ln(2)——训练开始时策略=参考模型,间距必为0,')
print('所以你看到的DPO日志首步loss总是≈0.693,这不是巧合而是数学必然。')

print()
print('=' * 22, 'Part 2: 灵魂实验 —— 亲眼看着偏好被拉开', '=' * 22)
sft = load('full_sft')      # DPO 之前的模型
dpo = load('dpo_6')         # 本课亲训(lr=1e-6,比官方4e-8激进,见手册)
ds = DPODataset(os.path.join(ROOT, 'dataset/dpo_smoke.jsonl'), tok, max_length=1024)

@torch.no_grad()
def answer_logp(model, x, y, mask):
    logits = model(x).logits
    lp = F.log_softmax(logits.float(), dim=2)
    tok_lp = torch.gather(lp, 2, y.unsqueeze(2)).squeeze(-1)
    return (tok_lp * mask).sum(dim=1)   # 每个样本: 答案部分的 logP 总和

N = 24
margins_sft, margins_dpo = [], []
for i in range(N):
    b = ds[i]
    x = torch.stack([b['x_chosen'], b['x_rejected']]).to(device)
    y = torch.stack([b['y_chosen'], b['y_rejected']]).to(device)
    mask = torch.stack([b['mask_chosen'], b['mask_rejected']]).to(device)
    lp = answer_logp(sft, x, y, mask)
    margins_sft.append((lp[0] - lp[1]).item())
    lp = answer_logp(dpo, x, y, mask)
    margins_dpo.append((lp[0] - lp[1]).item())

rel = torch.tensor([d - s for d, s in zip(margins_dpo, margins_sft)])  # DPO真正优化的量

def stat(v):
    t = torch.tensor(v)
    return f'均值{t.mean():+7.1f}  中位数{t.median():+7.1f}  正margin占比{(t > 0).float().mean():.0%}'

print(f'测了 {N} 对好/坏答案, 原始间距 = logP(好) - logP(坏):')
print(f'  SFT模型:  {stat(margins_sft)}')
print(f'  DPO模型:  {stat(margins_dpo)}')
print()
print(f'相对间距(当前 - 参考, DPO损失直接优化的量): 均值{rel.mean():+.2f}, '
      f'偏好变好的样本占{(rel > 0).float().mean():.0%}')
implied_loss = -torch.log(torch.sigmoid(0.15 * rel)).mean()
print(f'把这 {N} 个相对间距代回损失公式: 平均loss = {implied_loss:.3f}  (起点是ln2=0.693)')
print()
print('怎么读: 原始间距被"答案长短"主导(长答案概率连乘更小),绝对值无意义;')
print('DPO动的是相对间距——把每一对的差距往"更偏好chosen"的方向推。')

print()
print('=' * 22, 'Part 3: 三个防走火入库的设计', '=' * 22)
print('1. 参考模型(train_dpo.py:186-188): 起点模型的冻结拷贝,损失里始终减去它的间距,')
print('   相当于"原地踏步=0分"——防止模型用全面跑偏的方式刷分。')
print('2. beta=0.15(:152): 间距的放大器。太大→只能贴着参考模型动弹不得;')
print('   太小→冲太猛,语言能力退化(对齐税)。')
print('3. 学习率 4e-8(:137): 全流水线最小!对比 pretrain 5e-4 / SFT 1e-5 / LoRA 1e-4。')
print('   偏好优化是"拧最后几圈螺丝",稍微用力过猛就毁掉前面所有阶段的成绩。')
