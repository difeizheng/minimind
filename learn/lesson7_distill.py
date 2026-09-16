"""第7课：蒸馏 —— 老师把整张赌注分布传给学生
用法：python learn/lesson7_distill.py（任意目录均可）
灵魂实验：同一批考题，测 KL(老师||学生) 在蒸馏前后的下降。
本课配置：学生=pretrain_768(蒸馏前)，老师=full_sft_768，蒸馏后学生=full_dist_7_768。
"""
import os, sys
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch.manual_seed(42)

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load(w):
    m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    m.load_state_dict(torch.load(os.path.join(ROOT, f'out/{w}_768.pth'), map_location='cpu'))
    return m.eval().to(device)

teacher = load('full_sft')     # 老师：会上对话课的模型
student_before = load('pretrain')   # 学生蒸馏前：只会接龙
student_after = load('full_dist_7') # 学生蒸馏后（本课亲训1个epoch）

ds = SFTDataset(os.path.join(ROOT, 'dataset/sft_smoke.jsonl'), tok, max_length=512)

@torch.no_grad()
def next_token_logits(model, x):
    return model(x).logits[0, -1].float()

# 取一个真实考位：对话里 assistant 回答的下一个字
x, y = ds[3]
pos = next(i for i in range(1, len(x)) if y[i] != -100)  # 第一个要考的位置
ctx = x[:pos]
label_id = int(x[pos])
print('=' * 20, 'Part 0: 硬标签 vs 软标签', '=' * 20)
print(f'考位上下文(结尾): ...{tok.decode(ctx[-12:])!r}')
print(f'硬标签(SFT/CE用的): {tok.decode([label_id])!r} —— 一个one-hot: 对=100%, 其余6399个=0')
logits = next_token_logits(teacher, ctx.unsqueeze(0).to(device))
probs = F.softmax(logits, dim=-1)
top = torch.topk(probs, 8)
print('软标签(老师给的赌注分布) Top8:')
for v, i in zip(top.values, top.indices):
    print(f'   {tok.decode([i])!r:8s} {v.item():7.2%}'
          + ('   ← 正确答案也只拿这点概率' if i == label_id else ''))
print('软标签多教的东西: 不仅"谁对", 还有"谁接近对、谁八竿子打不着"——')
print('这些相对关系叫"暗知识"(dark knowledge),one-hot标签里不存在。')

print()
print('=' * 20, 'Part 1: 温度 —— 把分布"烫平"看清暗知识', '=' * 20)
print('蒸馏loss用的分布不是裸softmax,而是 softmax(logits/T):')
print('温度T   Top1概率   Top8合计   分布熵    效果')
for T in [0.5, 1.0, 2.0, 4.0]:
    p = F.softmax(logits / T, dim=-1)
    t8 = torch.topk(p, 8).values.sum().item()
    ent = -(p * (p + 1e-12).log()).sum().item()
    tag = {0.5: '更尖:只看最有把握的', 1.0: '原样', 2.0: '烫平:长尾知识浮现', 4.0: '很平:几乎均匀'}[T]
    print(f'{T:4.1f}   {p.max().item():8.2%}   {t8:8.2%}   {ent:6.2f}   {tag}')
print('T越大,老师"错选项们"的相对高低越清晰可学——这就是train_distillation.py:27的/T。')
print('公式里再乘T²(:36)是补偿:烫平后的梯度会变小,T²把它放大回来。')

print()
print('=' * 20, 'Part 2: 灵魂实验 —— KL(老师||学生) 蒸馏前后', '=' * 20)
@torch.no_grad()
def avg_kl(teacher, student, n_samples=8, T=1.5):
    kls, ces = [], []
    for idx in range(n_samples):
        x, y = ds[idx]
        x = x.unsqueeze(0).to(device)
        t_logits = teacher(x).logits[0, :-1].float()
        s_logits = student(x).logits[0, :-1].float()
        mask = (y[1:] != -100)
        tp = F.softmax(t_logits[mask] / T, dim=-1)
        slp = F.log_softmax(s_logits[mask] / T, dim=-1)
        kl = F.kl_div(slp, tp, reduction='batchmean')
        kls.append(kl.item())
        ce = F.cross_entropy(s_logits[mask], y[1:][mask].to(device))
        ces.append(ce.item())
    return sum(kls) / len(kls), sum(ces) / len(ces)

for name, s in [('蒸馏前(=pretrain)', student_before), ('蒸馏后(full_dist_7)', student_after)]:
    kl, ce = avg_kl(teacher, s)
    print(f'  {name}:  平均KL(老师||学生)={kl:.4f}   学生自己的CE(对硬标签)={ce:.3f}')
print()
print('怎么读: KL骤降=学生的赌注分布快速向老师靠拢;CE也在降=老师还兼顾了标准答案。')
print('对照训练日志: distill 0.59→0.22, ce 5.82→5.06 —— 和这里的测量互相印证。')

print()
print('=' * 20, 'Part 3: 代码要点', '=' * 20)
print('总损失(:93) = alpha×CE + (1-alpha)×KL   (alpha默认0.5,两路信号各半)')
print('老师全程冻结(:44-45, :208-209): eval模式+no_grad,只出卷不学习。')
print('(:66) 老师词表比学生大时,logits截到学生词表——大小模型也能蒸馏。')
print('蒸馏 vs SFT: 数据可以完全相同(sft_t2t_mini),差别只在"跟谁学"——')
print('跟one-hot学(CE)还是跟老师的分布学(KL)。')
