# 第9课 实操手册（毕业课）：流水线成果检阅 + 毕业考

> 运行：`python learn/lesson9_graduation.py`

---

## 实验一：五站对比（3 分钟）

看输出时对照这张"应该看到什么"：

| 站点 | 预期表现 | 说明什么 |
|---|---|---|
| ① pretrain | 词语沙拉（列表残骸、乱接龙） | 会语言不懂对话 |
| ② full_sft | "作为AI我应该保持简洁…"腔调 | 学会了形式，知识仍贫乏 |
| ②+身份LoRA | "我是由MiniMind/Jingyao…" | 1MB 外挂改写人格 |
| ③ 蒸馏学生 | "好的，用户问的是…"（对话雏形） | **只跟老师学 1 轮**就有此形态 |
| ④ grpo | 腔调 + 提到"由jyaogong创建" | RL 精修在赌注细微处 |

## 实验二：真正和你的模型聊天（eval_llm.py，5 分钟）

```bash
cd /app/zdf/minimind
python eval_llm.py --weight full_sft --max_new_tokens 256 --historys 4 --open_thinking 1
# 选 [1] 手动输入，输入任何话；Ctrl+C 退出
# 换权重再聊: --weight grpo / --weight pretrain / --weight full_dist_7
# 挂身份外挂聊天: --weight full_sft --lora_weight lora_id5
```

`--historys 4` 带上最近 2 轮对话（多轮上下文的实现：把历史一起塞进 prompt 重算）。
`--open_thinking 1` 让模型先写 `<think>` 再作答——第 8 课 RL 训练的就是这个思考段。

## 实验三（选做）：把你自己训练的所有权重检阅一遍

你已经亲手训过：`pretrain_3`（若保留）、`full_sft_4`、`lora_id5b`、`dpo_6b`、`full_dist_7`……
改 `lesson9_graduation.py` 的 STAGES 列表把它们加进去，办一场自己的成果展。

---

# 🎓 毕业考（10 题，覆盖 9 课；先答完再展开答案）

1. 为什么预训练不需要人工标注？这叫什么学习范式？
2. SFT 数据里 -100 出现在哪些位置？为什么这样设计？
3. loss 从 8.76 降到 6.0，模型对正确答案的把握提升了几倍？
4. 学习率 10 会发生什么？1e-6 会发生什么？余弦调度怎么两头兼顾？
5. SFT 和预训练的代码差异总共就 5 处，最能体现"F(微调)"的是哪一处？
6. LoRA 为什么 B 矩阵零初始化、A 矩阵高斯初始化？
7. one-hot 硬标签和老师软标签各教什么？温度 T 起什么作用？
8. DPO 为什么需要参考模型？为什么第一步 loss 必是 0.693？
9. GRPO 的一步是哪五拍？为什么减"组均分"就省掉了 critic？
10. 用一句话分别说出：SFT/蒸馏/RL 各自的能力天花板。

<details>
<summary>毕业考答案</summary>

1. 标签=文本右移一位自动生成（下一词预测），自监督学习。
2. user 提问、`assistant\n` 前缀、结尾 padding。只考核"轮到模型说话时说什么"。
3. e^2.76≈16 倍。
4. 10→loss 飙到 811（参数改飞）；1e-6→75 步原地踏步；先大后小：前期大步下山后期小步精修。
5. 学习率 5e-4→1e-5（小 50 倍）：在已成型的知识上轻拿轻放。
6. B 零 → 外挂初始无伤（输出恒 0）；A 高斯 → 梯度能流（双零则梯度死锁）。
7. 硬标签教"对什么"（one-hot 单选卡）；软标签还教"错的相对远近"（暗知识）；T 烫平分布让长尾可学。
8. 参考模型提供"零点"防跑偏刷分；起点时策略=参考，间距=0，-logsigmoid(0)=ln2。
9. 采样→打分→组内标准化优势→按优势加强/抑制每个token→KL缰绳。组内相对化免费获得 baseline，无需 critic。
10. SFT≤示范数据；蒸馏≤老师；RL≤裁判（唯一可超越一切示范）。
</details>

---

# 学完之后去哪

**项目里还没探索的角落**（都已在你能力圈内）：
- `trainer/train_tokenizer.py`——第 1 课的 tokenizer 本身是怎么训出来的（BPE）
- `--use_moe 1` 训一个 MoE 模型——第 2 课说过的"多专家会诊"+ aux_loss
- `trainer/train_ppo.py` / `train_agent.py`——PPO 与工具调用的 agent RL
- `README.md`——作者写的完整教程，现在你应该能无障碍读懂每一节
- 多卡 DDP：给训练命令加上 `torchrun --nprocfs节点数` 前缀的官方用法（单卡可跳过）

**外部世界**（按性价比排序）：
- 论文：LoRA (2021)、DPO (2023)、DeepSeek-R1 (2025)——你现在读得懂方法节了
- HuggingFace `transformers` + `trl` 库：工业界的训练流水线（你已懂原理，只剩 API）
- 换个真模型玩：Qwen 系列开源权重 + LoRA 微调，流程和第 5 课一模一样
