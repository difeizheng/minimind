"""第4课：SFT —— 同一个大脑，换一份考卷，换一种人格
用法：python learn/lesson4_sft.py（任意目录均可）
灵魂实验：同一个聊天问题，分别喂给 out/pretrain_768.pth（只会接龙）和
out/full_sft_768.pth（学会对话）。模型结构完全相同，唯一的区别是训练数据。
"""
import os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch.manual_seed(42)

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load(weight_name):
    m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    m.load_state_dict(torch.load(os.path.join(ROOT, f'out/{weight_name}_768.pth'), map_location='cpu'))
    return m.eval().to(device)

@torch.no_grad()
def chat(model, question, max_new_tokens=80):
    # 关键：推理时用 chat template 包装问题，并加"该你说了"的生成提示
    prompt = tok.apply_chat_template([{'role': 'user', 'content': question}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.85,
                         top_p=0.85, top_k=50, do_sample=True)[0]
    return tok.decode(out[ids.shape[1]:], skip_special_tokens=False)

pretrain_model = load('pretrain')   # 只读过纯文本的"接龙机器"
sft_model = load('full_sft')        # 上过对话课的"聊天助手"

print('=' * 24, 'Part 0: 两份考卷的差别', '=' * 24)
print('预训练考卷: "白日依山尽，黄河入海流..."            → 每个字都要接龙（labels=input_ids）')
print('SFT 考卷:   "<|im_start|>user\\n你好<|im_end|>...assistant\\n你好！<|im_end|>"')
print('             → 只有 assistant 的话要考，其余全是 -100（第1课学的掩码）')
print('两份考卷喂的是同一个模型、同一个交叉熵、同一个训练循环（第3课学的骨架）。')

print()
print('=' * 24, 'Part 1: 灵魂实验 —— 同一问题，两种人格', '=' * 24)
# [TODO A] 换成你想问的问题，先预测两个模型各会怎么答
questions = ['你好，你是谁？', '你是谁开发的？', '写一首诗', '1+1等于几？']
for q in questions:
    print(f'\n问: {q}')
    print('-' * 66)
    a = chat(pretrain_model, q)
    print(f'预训练模型（只会接龙）答:\n    {a[:300]}')
    b = chat(sft_model, q)
    print(f'SFT模型（上过对话课）答:\n    {b[:300]}')

print()
print('=' * 24, 'Part 2: 亲眼看 SFT 在学什么', '=' * 24)
# 对"SFT考卷考核的第一个token"，两份权重的赌注对比
import torch.nn.functional as F
prompt = tok.apply_chat_template([{'role': 'user', 'content': '推荐一本好书'}],
                                 tokenize=False, add_generation_prompt=True)
# 模板默认会补上空的<think></think>块，这里截掉，让提示停在"assistant\n"——
# 此时"下一个token"正是SFT考卷每道题的第一个必答token
prompt = prompt[:prompt.index('<think>')]
ids = tok(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
think_id = tok('<think>', add_special_tokens=False).input_ids[0]
print(f'问题包装后以 ...{prompt[-30:]!r} 结尾，此时模型该接的第一个 token 是 <think>')
for name, m in [('预训练', pretrain_model), ('SFT', sft_model)]:
    with torch.no_grad():
        logits = m(ids).logits[0, -1]
    p = F.softmax(logits.float(), dim=-1)
    rank = (p > p[think_id]).sum().item() + 1
    print(f'  {name}模型给 <think> 的赌注: {p[think_id].item():.2%}（全词表第{rank}名）')

print()
print('解读: SFT 之后，"轮到我说话先摊开草稿纸<think>"成了肌肉记忆——')
print('不是模型变聪明了，是考卷只考这些，它把赌注重新分配到了"对话该有的样子"上。')
