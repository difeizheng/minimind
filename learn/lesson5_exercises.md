# 第5课 实操手册：LoRA 低秩微调（命令行版）

> 运行演示：`python learn/lesson5_lora.py`
> 核心认知：冻结全部 64M 参数，只训 0.39M 的"外挂"（0.61%），也能改写模型行为。

---

## 原理速览：外挂是什么

对模型里每个方阵 Linear（如 768×768 的 `q_proj`/`o_proj`），不改原矩阵 W，而是：

```
输出 = W·x + B·(A·x)
        ↑原路冻结    ↑外挂旁路（可训练）
A: 768→16 高斯初始化   B: 16→768 全0初始化
```

三个精妙设计（对应 `model/model_lora.py:6-18`）：
1. **B 全 0 初始化** → 外挂刚挂上时输出恒为 0，模型行为与挂之前分毫不差（训练从"无伤"起步）；
2. **低秩** → 改动量 B@A 的秩最多 16。LoRA 论文的洞察：微调所需的改动本质是低秩的，16 个方向足以表达"这次微调想改什么"；
3. **可合并** → 部署时 `W_new = W + B@A`（`model_lora.py:56` 的 merge_lora），烙进原矩阵后零额外开销。

## 实验一：看懂外挂结构（REPL 10 分钟）

```bash
python
>>> import sys; sys.path.insert(0, '/app/zdf/minimind')
>>> import torch
>>> from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
>>> from model.model_lora import apply_lora
>>> m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
>>> apply_lora(m, rank=16)
>>> lora_p = sum(p.numel() for n, p in m.named_parameters() if 'lora' in n)
>>> all_p = sum(p.numel() for p in m.parameters())
>>> lora_p, all_p, f'{lora_p/all_p:.2%}'
>>> [n for n, mod in m.named_modules() if hasattr(mod, 'lora')][:3]
>>> exit()
```

**思考题**：为什么 k_proj/v_proj 没有外挂？（提示：`apply_lora` 只处理 `in_features==out_features` 的方阵，k/v 是 768→384；q/o 是 768→768。方阵才能直接 `W + B@A` 合并）

## 实验二：灵魂演示（2 分钟）

```bash
cd /app/zdf/minimind && python learn/lesson5_lora.py
```

已验证看点：Part 3 里原模型答"你是谁"是词语沙拉；挂上身份 LoRA（0.39M 参数）后回答"**我是由 Jingyaogong**"——身份信息只存在于外挂里，原矩阵一个数字没动。

## 实验三：亲手训一个身份 LoRA（约 1 分钟，强烈推荐）

```bash
cd /app/zdf/minimind/trainer
python train_lora.py --lora_name lora_id5b \
  --data_path ../dataset/lora_identity.jsonl \
  --from_weight full_sft --epochs 50 --batch_size 32 \
  --log_interval 30 --num_workers 4
```

观察日志开头三行：`LLM 总参数量` / `LoRA 参数量` / `LoRA 参数占比 0.61%`——**优化器只收 LoRA 参数**（train_lora.py:141-152 冻结其余全部）。
已验证：50 轮（每轮仅 3 步！）约 1 分钟。产物是 `out/lora_id5b_768.pth`——**只有 ~1MB**，对比全量 SFT 存档 128MB。

**进阶玩法（选做）**：把 `dataset/lora_identity.jsonl` 复制一份，把回答里的 "MiniMind"/"Jingyao Gong" 改成你自己的名字（`sed` 或 vim 都行），重新训练——**做一个"以为是你开发"的模型**：

```bash
sed 's/Jingyao Gong/你的名字/g; s/MiniMind/你的模型名/g' ../dataset/lora_identity.jsonl > ../dataset/lora_me.jsonl
# 然后把 --data_path 换成 ../dataset/lora_me.jsonl 再跑
```

（lora_me.jsonl 已加入想xdg忽略的话可自行删除；改完数据记得换 --lora_name）

## 实验四：合并外挂（选做，10 分钟）

```bash
cd /app/zdf/minimind
python -c "
import sys; sys.path.insert(0, '.')
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import merge_lora
m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
m.load_state_dict(torch.load('out/full_sft_768.pth', map_location='cpu'))
merge_lora(m, 'out/lora_id5_768.pth', 'out/merged_id5_768.pth')
print('合并完成: out/merged_id5_768.pth（普通权重，无外挂结构）')"
```

合并后的权重 = 普通模型存档，可用 `lesson4_sft.py` 的方式加载聊天（把 weight 名换掉），行为与挂外挂版相同。这就是"部署时撕掉外挂"。

## 验收自测

1. LoRA 的外挂由哪两个矩阵组成？各自怎么初始化？为什么这样初始化？
2. rank=16 时每个外挂多少参数？整个模型挂了多少？占比？
3. `W_new = W + B@A` 为什么形状刚好合法？
4. LoRA 为什么省显存？（提示：优化器状态只需要为哪些参数准备？）
5. 一个基座模型如何变成医疗版+法律版两个产品？LoRA 的存储优势在哪？

<details>
<summary>自测答案（先自己答，再展开）</summary>

1. A(768→16，高斯初始化)和 B(16→768，全0)。B 全0 → 初始时 B·A·x=0，外挂不改变行为；A 高斯 → 保证梯度能流（若 A 也全0，两个零矩阵相乘梯度全为0，永远学不动）。
2. 每个外挂 2×768×16=24576 个；全模型 16 个方阵 Linear（8层×q_proj+o_proj）共 0.393M，占 63.9M 的 0.61%。
3. B 是 768×16，A 是 16×768，B@A 得 768×768，与 W 同形状可相加。
4. 反向传播不为冻结参数算梯度/动量；AdamW 只需为 0.39M 参数保存一阶+二阶动量（对比全量的 63.9M×2 份状态）。
5. 基座存一份，医疗/法律 LoRA 各存 ~1MB；用时挂载或合并。对比各存一份 128MB 全量模型，存储和切换成本低百倍。

</details>

---

## 过关之后

第 6 步：DPO。考卷升级成"一好一坏两个答案"——不再教模型说什么，而是教它**在两种说法里挑更好的那种**。这是"对齐"的开始。
