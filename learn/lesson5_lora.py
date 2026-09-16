"""第5课：LoRA —— 冻结大脑，外挂一小片"补丁"
用法：python learn/lesson5_lora.py（任意目录均可）
演示：什么是低秩外挂、它有多小、以及它如何改变模型人格（身份认同）。
"""
import os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch.manual_seed(42)

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_base():
    m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    m.load_state_dict(torch.load(os.path.join(ROOT, 'out/full_sft_768.pth'), map_location='cpu'))
    return m.eval().to(device)

@torch.no_grad()
def chat(model, question, max_new_tokens=60, temperature=0.85, top_k=50):
    prompt = tok.apply_chat_template([{'role': 'user', 'content': question}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_p=0.85, top_k=top_k, do_sample=True)[0]
    return tok.decode(out[ids.shape[1]:], skip_special_tokens=True).strip()

print('=' * 22, 'Part 1: 外挂长什么样', '=' * 22)
model = load_base()
apply_lora(model, rank=16)   # 给模型挂上 LoRA
total = sum(p.numel() for p in model.parameters())
lora_p = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n)
print(f'冻结的原模型参数: {(total - lora_p) / 1e6:.2f}M（requires_grad=False，训练时纹丝不动）')
print(f'可训练的LoRA参数: {lora_p / 1e6:.3f}M（只占 {lora_p / total:.2%}）')
hooked = [n for n, m in model.named_modules() if hasattr(m, 'lora')]
print(f'外挂位置: {len(hooked)} 个方阵Linear，如 {hooked[0]}')
w = dict(model.named_modules())[hooked[0]].lora
print(f'每个外挂 = 两个小矩阵串联: A{tuple(w.A.weight.shape)} × B{tuple(w.B.weight.shape)}')
print(f'原矩阵 768×768={768 * 768:,} 个参数，外挂只要 {w.A.weight.numel() + w.B.weight.numel():,} 个')
print('关键设计: B 初始化为全 0 → 外挂刚挂上时输出恒为 0，模型行为与原来分毫不差；')
print('         训练只更新 A、B，原矩阵从头到尾不动。')

print()
print('=' * 22, 'Part 2: 数学魔法 —— 合并', '=' * 22)
print('推理时可以"撕掉外挂"，把效果永久烙进原矩阵:')
print('   W_new = W + B @ A   （768×16 乘 16×768 = 768×768，形状完美对上）')
print('这就是 merge_lora(model_lora.py:56) 干的事——合并后不再有额外计算开销。')
print('为什么叫"低秩"？改动量 B@A 的秩最多为 16，远小于 768。')
print('LoRA 论文的洞察: 微调一个模型所需的改动，本质上是低秩的——')
print('不需要动 768×768 那么多自由度，16 个方向就够表达"这次微调想改什么"。')

print()
print('=' * 22, 'Part 3: 灵魂实验 —— 91条数据改写人格', '=' * 22)
# 原模型 vs 原模型+身份LoRA（本课在真机上用 train_lora.py 训练，见手册实验三）
base = load_base()
with_lora = load_base()
apply_lora(with_lora, rank=16)
load_lora(with_lora, os.path.join(ROOT, 'out/lora_id5b_768.pth'))
print('身份LoRA = 用 dataset/lora_identity.jsonl（91条"你叫什么/你是谁"问答）训出的补丁')
print('训练耗时: 在这张卡上约1分钟（91条数据×50轮，只更新那 0.39M 个外挂参数）')
for q in ['你是谁？', '你叫什么名字？']:
    print(f'\n问: {q}')
    print(f'  原模型:     {chat(base, q, temperature=0.85)[:80]}')
    print(f'  挂身份LoRA: {chat(with_lora, q, temperature=0.1, top_k=5)[:80]}')
print('\n注意: 原模型 63.9M 参数一个都没变——变的只是那 0.39M 的外挂。')
print('（回答仍重复结巴: 64M小模型的极限；但"我是由Jingyaogong"这个身份信息是外挂带去的）')

print()
print('=' * 22, 'Part 4: LoRA 的现实意义', '=' * 22)
print(f'训练显存: 优化器只需为 {lora_p / 1e6:.2f}M 参数保存动量，而不是 63.9M')
print('一个基座 + N 个外挂: 医疗版/法律版/客服版共享同一个底座，切换只换小补丁')
print('这就是把"微调大模型"从大厂专利变成人人可玩的技术。')
