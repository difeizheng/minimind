# 第7课 实操手册：知识蒸馏（命令行版）

> 运行演示：`python learn/lesson7_distill.py`
> 核心认知：蒸馏和 SFT 可以用**同一份数据**，差别只在"跟谁学"——one-hot 硬标签 vs 老师的整张赌注分布。

---

## 原理：软标签与暗知识

SFT 的交叉熵标签是 one-hot：`<think>` 对 = 100%，其余 6399 个 token = 0%。
蒸馏的老师标签是一张分布（本课实测）：`<think>` 45%、`我` 10.7%、`今天` 3.2%……

**多出来的信息 = "错误选项之间的相对高低"**（"我"比"今天"更像正解，"香蕉"八竿子打不着）。
这叫**暗知识**。一句话总结：硬标签教你"对什么"，软标签还教你"错的方式哪种更体面"。

## 损失函数（train_distillation.py）

```
总损失 = alpha × CE(学生 vs 硬标签) + (1-alpha) × KL(老师 || 学生)    (:93)
KL 计算前双方都除以温度 T，再乘 T² 补偿梯度                          (:25-36)
```

- **alpha=0.5**：硬软两路信号各半（0=纯蒸馏，1=退化回普通 SFT）
- **温度 T=1.5**：把老师分布"烫平"，让长尾的相对关系显现（见演示 Part 1 的熵变化）
- **老师冻结**（:44-45, :208-209）：`eval() + requires_grad_(False)`，只出卷不学习
- **:66 词表对齐**：老师词表大于学生时 logits 截齐——大老师蒸小学生是常规操作

## 本课实测配置与结果（全部可复现）

配置：学生从 `pretrain` 出发（只会接龙），老师 = `full_sft`，数据 sft_smoke，1 epoch：

```bash
cd /app/zdf/minimind/trainer
python train_distillation.py --epochs 1 --batch_size 32 \
  --data_path ../dataset/sft_smoke.jsonl \
  --from_student_weight pretrain --from_teacher_weight full_sft \
  --teacher_use_moe 0 --save_weight full_dist_7 \
  --log_interval 100 --num_workers 4
```

**已验证**：约 3-4 分钟，625 步；训练日志 `distill: 0.59→0.22, ce: 5.82→5.06`；
独立测量 `KL(老师||学生): 0.5016→0.1069`（降为 1/5），`CE: 6.47→4.88`。
（默认配置 teacher_use_moe=1 需要 MoE 权重 `full_sft_moe_768.pth`，本机没有，故显式传 0）

## 实验一：温度直觉（改 lesson7_distill.py 的 T 列表）

把 Part 1 的 `[0.5, 1.0, 2.0, 4.0]` 加入 `0.1` 和 `10.0`，先预测现象再跑：
- T=0.1 应该接近 one-hot（只留第一名）
- T=10 应该接近均匀（全部拉平，暗知识被"过度稀释"到看不见）

体会：温度是"看老师赌注的显微镜倍数"——倍数太高什么都看不清。

## 实验二：alpha 两极（各约 3 分钟）

```bash
# 纯蒸馏: 老师是唯一老师
python train_distillation.py --epochs 1 --batch_size 32 --data_path ../dataset/sft_smoke.jsonl \
  --from_student_weight pretrain --from_teacher_weight full_sft --teacher_use_moe 0 \
  --save_weight full_dist_pure --alpha 0.0 --log_interval 200 --num_workers 4
# 纯CE: 退化回普通SFT,老师白加载
python train_distillation.py --epochs 1 --batch_size 32 --data_path ../dataset/sft_smoke.jsonl \
  --from_student_weight pretrain --from_teacher_weight full_sft --teacher_use_moe 0 \
  --save_weight full_dist_noT --alpha 1.0 --log_interval 200 --num_workers 4
```

对比日志里 `ce` 列的走势：纯蒸馏的 ce 也在降（老师的软标签里含着硬标签的信息），
但通常不如 alpha=0.5 的混合配方降得稳——这就是"两路信号互补"的实证。

## 实验三（思考）：为什么大厂痴迷蒸馏

1. GPT-4 级老师 + 开源小学生的组合为什么比"直接训练小学生"强？
2. 为什么 R1 出来后全网都在蒸馏它的推理链？（提示：第 8 课 RL 的成果怎么"批发"给小模型）
3. 蒸馏和 LoRA 能同时用吗？（能——冻结基座+外挂，跟老师学外挂）

## 验收自测

1. 软标签比硬标签多教了什么？这叫什么知识？
2. 温度 T 调大，分布怎么变？为什么蒸馏需要烫平？T² 项补偿什么？
3. alpha=0 / 0.5 / 1 分别对应什么训练方式？
4. 老师模型为什么必须 no_grad + eval？
5. 蒸馏和 SFT 用同一份数据时，到底差在哪？

<details>
<summary>自测答案（先自己答，再展开）</summary>

1. 错误选项之间的相对远近（类间相似结构）；暗知识（dark knowledge）。
2. 分布变平、熵变大、Top1 概率骤降；烫平后老师对"错选项"的排序变得可学（尖锐分布里长尾梯度几乎为0）；T² 补偿烫平导致的梯度量级衰减。
3. 0=纯蒸馏（只跟老师）；0.5=蒸馏+SFT 混合（默认）；1=普通 SFT（老师白请）。
4. 老师不学习：eval 关 dropout/no_grad 不建计算图省显存——老师是教材不是学生；若老师也更新就成了"两个学生互学"，失去稳定的教学信号。
5. 跟谁学：one-hot（从数据学）vs 老师分布（从模型学）。数据 pipeline、模型、优化器都可以一模一样。

</details>

---

## 过关之后

第 8 步：RL（GRPO）。考卷将彻底消失——模型自己答题，裁判打分，好的加强坏的抑制。DeepSeek-R1 推理能力的真正来源。
