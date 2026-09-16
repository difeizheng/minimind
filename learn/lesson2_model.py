"""第2课：模型结构 —— 打开"学生"的大脑
用法：python learn/lesson2_model.py（任意目录均可）
模型 = 查字典(嵌入) → 8层理解(注意力+MLP) → 输出赌注(6400选1)。
本脚本加载项目已训练好的 out/pretrain_768.pth 权重做真实演示。
"""
import os, sys
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch.manual_seed(42)

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
model.load_state_dict(torch.load(os.path.join(ROOT, 'out/pretrain_3_768.pth'), map_location='cpu'))
model.eval().to(device)

print('=' * 22, 'Part 1: 模型解剖 —— 参数都住在哪', '=' * 22)
parts = {'查字典层(词嵌入表)': 'embed_tokens', '8层注意力(圆桌会议)': 'self_attn',
         '8层MLP(独立思考间)': '.mlp', '归一化(数值稳定器)': 'norm'}
total = sum(p.numel() for p in model.parameters())
for name, key in parts.items():
    n = sum(p.numel() for k, p in model.named_parameters() if key in k)
    print(f'  {name:20s} {n / 1e6:7.2f}M  ({n / total:.1%})')
print(f'  {"合计":18s} {total / 1e6:7.2f}M')
print('  数据流: 词→向量 → [Block×8: 开会→回房消化] → 输出6400个赌注')
print('  [TODO C] 用下面这行看每一层的矩阵形状（把注释去掉运行）:')
print('  print([(k, tuple(v.shape)) for k, v in model.named_parameters() if "layers.0" in k])')
print([(k, tuple(v.shape)) for k, v in model.named_parameters() if "layers.0" in k])
print([(k, tuple(v.shape)) for k, v in model.named_parameters() if "layers.1" in k])
print([(k, tuple(v.shape)) for k, v in model.named_parameters() if "layers.2" in k])
print([(k, tuple(v.shape)) for k, v in model.named_parameters() if "layers.3" in k])
print([(k, tuple(v.shape)) for k, v in model.named_parameters() if "layers.4" in k])
print([(k, tuple(v.shape)) for k, v in model.named_parameters() if "layers.5" in k])
print([(k, tuple(v.shape)) for k, v in model.named_parameters() if "layers.6" in k])
print([(k, tuple(v.shape)) for k, v in model.named_parameters() if "layers.7" in k])

print()
print('=' * 22, 'Part 2: 查字典 —— 模型眼里的"秋"', '=' * 22)
E = model.model.embed_tokens.weight.detach()
for ch in ['秋', '天']:
    i = tok(ch, add_special_tokens=False).input_ids[0]
    v = E[i]
    print(f'  {ch} = token {i} → 查表第{i}行 = 一个{v.shape[0]}维向量, 前5个分量: '
          f'{[round(x, 3) for x in v[:5].tolist()]}')
print('  此后模型眼里没有"字"，只有这些数字向量。')

print()
print('=' * 22, 'Part 3: 输出赌注 —— 下一字是啥', '=' * 22)
RANDOM_P = 1 / 6400
def bets(prompt, k=5):
    ids = [tok.bos_token_id] + tok(prompt, add_special_tokens=False).input_ids
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device)).logits[0, -1]
    p = F.softmax(logits.float(), dim=-1)
    top = torch.topk(p, k)
    return p, [(tok.decode([i]), v.item()) for v, i in zip(top.values, top.indices)]

# [TODO A] 换成你自己的 prompt：成语上半句、歌词、你的口头禅……先猜 Top1 再跑
for prompt in ['昨天下雨，今天是晴天，明', '今天晚上吃']:
    p, top = bets(prompt)
    print(f'  问: {prompt!r} + ?   模型的Top5赌注:')
    for w, v in top:
        print(f'     {w!r:8s} {v:7.2%}   （≈瞎蒙水平的 {v / RANDOM_P:.0f} 倍）')
    print(f'     其余6395个字分摊剩余 {1 - sum(v for _, v in top):.1%}')

print()
print('  同一个"天"字，上下文不同，赌注不同（注意力在起作用）:')
tian = tok('天', add_special_tokens=False).input_ids[0]
for prompt in ['我爱秋', '秋天来了，金色的秋天终']:
    p, _ = bets(prompt)
    print(f'     p(天 | {prompt}) = {p[tian].item():.2%}')

# [TODO B] 把上面 model.load_state_dict(...) 那行注释掉（模型变回"文盲"）再跑本脚本，
#          对比 Top5：赌注会退化成 0.016% 附近的均匀噪声——这就是"训练前"和"训练后"的差别。
