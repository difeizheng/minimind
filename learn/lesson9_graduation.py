"""第9课(毕业课)：整条流水线的成果检阅
用法：python learn/lesson9_graduation.py（任意目录均可）
同一张考卷交给流水线每一站的模型：看每一站给模型叠了什么。
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

@torch.no_grad()
def chat(model, q, n=70):
    p = tok.apply_chat_template([{'role': 'user', 'content': q}],
                                tokenize=False, add_generation_prompt=True)
    ids = tok(p, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    out = model.generate(ids, max_new_tokens=n, temperature=0.8, top_p=0.9,
                         top_k=50, do_sample=True)[0]
    return tok.decode(out[ids.shape[1]:], skip_special_tokens=True).strip()

def build(spec):
    """spec: (名字, 权重, 说明, 是否挂lora)"""
    name, w, note, lora = spec
    m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    m.load_state_dict(torch.load(os.path.join(ROOT, f'out/{w}_768.pth'), map_location='cpu'))
    m.eval().to(device)
    if lora:
        apply_lora(m, rank=16)
        load_lora(m, os.path.join(ROOT, 'out/lora_id5_768.pth'))
    return name, m, note

STAGES = [
    ('① 预训练(pretrain)', 'pretrain',    '读过127万条文本的接龙机器', False),
    ('② SFT(full_sft)',   'full_sft',    '上过对话课: 学会了对话的形式', False),
    ('②+身份LoRA(lora_id5)', 'full_sft',  'SFT + 0.39M外挂改写身份(第5课亲训)', True),
    ('③ 蒸馏学生(full_dist_7)', 'full_dist_7', 'pretrain学生跟full_sft老师学1轮(第7课亲训)', False),
    ('④ RL(grpo)',        'grpo',        '官方GRPO权重(裁判打分精修)', False),
]

QUESTIONS = ['你是谁？', '用一句话解释什么是机器学习']

models = []
for spec in STAGES:
    name, m, note = build(spec)
    models.append((name, m, note))

for q in QUESTIONS:
    print('=' * 72)
    print(f'💬 问题: {q}')
    print('=' * 72)
    for name, m, note in models:
        ans = chat(m, q).replace('\n', ' ')[:88]
        print(f'{name}\n   [{note}]\n   答: {ans}')
    print()

print('=' * 72)
print('怎么读这张对比表:')
print('  ①→② 形式之变: 从词语沙拉到"人话腔调"(对话格式/收口/口吻)——但知识仍贫乏')
print('  ②→②+ 一个1MB外挂就能改写身份——基座纹丝未动')
print('  ③ 蒸馏学生: 只跟老师学了一轮,已有对话雏形——知识的搬运效率')
print('  ④ RL精修: 表面难分高下(弱模型+短训练),它的改动在"赌注的细微处"')
print()
print('记住这个诚实结论: 64M小模型每站的变化是"结构性"的(形式/身份/偏好),')
print('"知识含量"的提升主要靠预训练规模——这正是规模定律的意义。')
