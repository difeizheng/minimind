# 第4课 实操手册：SFT 监督微调（命令行版）

> 运行演示：`python learn/lesson4_sft.py`
> 核心认知：SFT 与预训练**训练循环完全相同**，差异只有"考卷"（数据+掩码）和几行默认参数。

---

## 实验一：diff 揭晓真相（1 分钟）

```bash
cd /app/zdf/minimind/trainer
diff train_pretrain.py train_full_sft.py
```

你会看到**训练循环一行没改**，实质差异只有 5 处（我验证过的完整清单）：

| 差异点 | 预训练默认 | SFT 默认 | 为什么 |
|---|---|---|---|
| 数据集类 | `PretrainDataset` | `SFTDataset` | 纯文本 vs 对话+掩码 |
| `--from_weight` | `none`（从零） | `pretrain`（接上一步） | 流水线传球 |
| `--learning_rate` | 5e-4 | **1e-5（小50倍）** | 微调=轻拿轻放，别毁掉已有知识 |
| `--max_seq_len` | 340 | 768 | 对话比短文本长 |
| `--batch_size`/累积 | 32×8 | 16×1 | 小数据不需要大等效batch |

**这个 50 倍小的学习率是"SFT"里"F(fine)"的全部含义**——好学生已经有了，只是精修。

## 实验二：灵魂演示（3 分钟）

```bash
cd /app/zdf/minimind && python learn/lesson4_sft.py
```

三部分输出（均已实测验证）：
- **Part 1**：同一问题"你好，你是谁？"，预训练模型输出**词语沙拉**（它在接龙，见开头 `1.`/`2.` 之类被切断的列表残骸），SFT 模型输出**人话**（第一人称、自我介绍、结尾 `<|im_end|>` 收口）。
- **Part 2**：提示停在 `assistant\n` 时，`<think>` 的赌注：预训练模型 **0.00%（第6046名，全词表倒数）** → SFT 模型 **33.37%（第1名）**。
- 注意：SFT 模型的回答**形式对了但内容仍贫乏**（64M 小模型的极限）——SFT 教会的是"行为"，不是"知识"。

## 实验三：亲手 SFT（约 2 分钟）

```bash
cd /app/zdf/minimind/trainer
python train_full_sft.py --epochs 1 --batch_size 16 \
  --data_path ../dataset/sft_smoke.jsonl \
  --from_weight pretrain --save_weight full_sft_4 \
  --log_interval 50 --num_workers 4
```

**已验证结果**：1250 步，loss 降到 ~4.1。
⚠️ 必须带 `--save_weight full_sft_4`，否则默认名会覆盖官方 `out/full_sft_768.pth`。

观察两个细节：
1. 首步 loss ≈ 6 出头（不是 8.78）——因为它从 pretrain 权重出发，"阅读理解"能力已预付；
2. SFT 的 loss 降到 4.1，比预训练的 6.08 低——**不是 SFT 更厉害，是考卷不同**（只考 assistant 段，且对话句式重复度高）。跨阶段的 loss 不可比。

## 实验四：SFT 数据长什么样（REPL 5 分钟）

```bash
python
>>> import sys; sys.path.insert(0, '/app/zdf/minimind')
>>> from transformers import AutoTokenizer
>>> tok = AutoTokenizer.from_pretrained('/app/zdf/minimind/model')
>>> import json
>>> line = open('/app/zdf/minimind/dataset/sft_smoke.jsonl').readline()
>>> convs = json.loads(line)['conversations']
>>> print(tok.apply_chat_template(convs, tokenize=False))
>>> exit()
```

对照第 1 课：这段渲染出的文本 + `generate_labels` 的 -100 掩码 = SFT 考卷。

## TODO（动手改）

- **TODO A**：改 `lesson4_sft.py` 的 questions（换成"写一首诗"/"1+1等于几"），先预测两边表现再跑。
- **TODO B**：实验三之后，把 `lesson2_model.py` 的权重改成 `full_sft_4`，看"你SFT一轮的模型"给 `<think>` 的赌注是多少名。
- **TODO C（思考）**：SFT 回答里那个 `<think>\n...\n</think>` 是什么？为什么考卷要考它？（提示：第8课 RL 的主角就是它）

## 验收自测

1. 用一句话说清 SFT 和预训练的全部区别。
2. 为什么 SFT 的学习率要小 50 倍？大学习率微调会怎样？
3. 为什么 SFT 的 loss（4.1）比预训练（6.08）低？这说明 SFT 模型更强吗？
4. `--from_weight pretrain` 这行参数如果忘了带（变成 none），会发生什么？
5. SFT 模型回答"形式对了内容贫乏"，说明 SFT 教会了什么、没教会什么？

<details>
<summary>自测答案（先自己答，再展开）</summary>

1. 同一个模型、同一个训练循环、同一个损失函数；只是考卷从"每字必考的纯文本"换成"只考 assistant 回答的对话"（-100 掩码），外加更小的学习率。
2. 微调是在已成型的知识上精修，大步子会把预训练学到的语言能力冲毁（灾难性遗忘）；1e-5 是"轻拿轻放"。
3. 不说明更强。两份考卷不同：SFT 只考 assistant 段且对话句式重复度高，天然分低。跨阶段比较 loss 没有意义。
4. 从随机初始化的"文盲"直接学对话——loss 从 8.78 起步，学到的质量差得多。这就是流水线"预训练打底"的意义。
5. 教会了：对话的**形式**（角色轮换、先 `<think>` 再作答、`<|im_end|>` 收口、第一人称口吻）。没教会：知识和推理能力——那些要靠预训练规模和后面的 RL。

</details>

---

## 过关之后

第 5 步：LoRA。SFT 动了全部 64M 参数，LoRA 只动 0.2%——为什么冻结原模型、只训一小片"外挂"也能改行为？那是通向"人人都能微调大模型"的门。
