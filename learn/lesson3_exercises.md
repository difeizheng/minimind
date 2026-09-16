# 第3课 实操手册：预训练 train_pretrain.py（命令行版）

> 核心认知：`train_pretrain.py` = 第1课的考卷 + 第2课的学生 + 第1b课的五行循环 + **六件工程盔甲**。
> 所有 trainer（SFT/LoRA/DPO/蒸馏/RL）共享这套骨架，学会这一个等于学会全部。

---

## ⚠️ 开跑前必读（保护 lesson2 用过的官方权重）

trainer 的默认 `--save_weight pretrain` 会**直接覆盖** `out/pretrain_768.pth`（第2课演示用的官方权重）！
本课所有命令统一带 `--save_weight pretrain_3`，这是安全 newName，别省略。

## 实验一：亲手跑一次真正的预训练（约 2 分钟）

```bash
cd /app/zdf/minimind/trainer
python train_pretrain.py \
  --epochs 1 --batch_size 32 \
  --data_path ../dataset/pretrain_smoke.jsonl \
  --save_weight pretrain_3 \
  --log_interval 25 --num_workers 4
```

**已验证的预期结果**：共 625 步，loss 从 **8.78 一路降到 ~6.08**（这张卡上约 1~2 分钟跑完）。

跑的时候对照日志逐项认领：

| 日志字段 | 含义 | 本课对应 |
|---|---|---|
| `loss` | 总扣分（这里=logits_loss，aux=0） | 第1课的 -ln(赌注) |
| `logits_loss` | 纯"猜下一字"的扣分 | 同上 |
| `aux_loss` | MoE 路由损耗（dense 模型恒为 0） | 第2课提过的 MoE |
| `lr` | 当前学习率，**注意它在变小** | 实验三 |
| `epoch_time` | 预计剩余分钟 | 工程便利 |

## 实验二：六件工程盔甲在代码的哪里（读代码 20 分钟）

打开 `trainer/train_pretrain.py`，用 `grep -n 关键词` 逐个找到（括号是我验证过的行为）：

1. **学习率调度** :31-33，公式在 `trainer_utils.py:40` —— 余弦退火：5e-4 起步，收尾降到 5e-5（10% 地板）。跑 `python learn/lesson3_lr_curve.py` 看曲线。
2. **混合精度** :122-123 —— 用 bfloat16 计算，速度约 2 倍、显存减半；:138 的 GradScaler 只在 fp16 时启用（bf16 数值范围大，不需要放缩）。
3. **梯度累积** :38 和 :42 —— `loss ÷ 8` 攒 8 个 batch 的梯度才更新一次。**等效 batch_size = 32×8 = 256**，但显存只需装得下 32。
4. **梯度裁剪** :44 —— 梯度总长度超过 1.0 就等比例缩小。还记得第1课 lr=10 把 loss 顶到 811 吗？裁剪就是防这种"一步登天变一步升天"的保险丝。
5. **检查点与续训** :61-70 —— 两份存档：`out/pretrain_3_768.pth`（半精度纯权重，推理用，128MB）+ `checkpoints/pretrain_3_768_resume.pth`（全精度权重+优化器状态+进度，650MB，续训用）。
6. **日志/监控** :51-59 —— 屏幕日志 + 可选 wandb 画曲线。

思考题（手册底部有答案）：为什么推理存档存半精度、续训存档必须全量？

## 实验三：看学习率的余弦曲线（1 分钟）

```bash
cd /app/zdf/minimind && python learn/lesson3_lr_curve.py
```

已验证输出：0% 时 5.00e-04（100%）→ 50% 时 2.75e-04（55%）→ 100% 时 5.00e-05（10%）。
联系第1课实验三：lr 太大学飞、太小学不动 → 所以"先大后小"两全其美。

## 实验四：断点续训（约 3 分钟，模拟"机器突然断电"）

```bash
cd /app/zdf/minimind/trainer
# 第一步：只让它跑 20 秒就被强制"断电"
timeout 20 python train_pretrain.py --epochs 1 --batch_size 32 \
  --data_path ../dataset/pretrain_smoke.jsonl --save_weight pretrain_3 \
  --save_interval 100 --log_interval 50 --num_workers 4
# 第二步：从断点接着跑
python train_pretrain.py --epochs 1 --batch_size 32 \
  --data_path ../dataset/pretrain_smoke.jsonl --save_weight pretrain_3 \
  --from_resume 1 --log_interval 50 --num_workers 4
```

**已验证**：第二步会打印 `Epoch [1/1]: 跳过前N个step，从step N+1开始`，然后从断点继续（不会重刷已学过的题——`trainer_utils.py:134` 的 SkipBatchSampler 干的活）。
注：20 秒内可能已跑完不少步；若第一步已跑完全部 625 步，第二步会打印"跳过前625个step"然后直接结束——这也是正确行为。

## 实验五（TODO，动手改）

- **TODO A**：跑完实验一后，再来一次"接力"：同命令加 `--from_weight pretrain_3`（注意不是 from_resume）。观察首步 loss：从 ~6.0 附近起步而不是 8.78——**权重就是学习的全部积累**。
- **TODO B**：把 `learn/lesson2_model.py` 里加载权重那行改成 `pretrain_3_768.pth`，对比"你训练 1 轮的模型"和"官方完整预训练模型"在 `'我爱秋'` 上的 Top5 赌注——你那版的天赋点还差多少？
- **TODO C**：`--accumulation_steps 8` 再跑实验一。观察：总步数不变（还是 625），但优化器每 8 步才动一次手（日志 lr 的阶梯不变，loss 曲线几乎一样）——想想为什么等效于大 batch。

## 验收自测

1. 余弦学习率解决了第1课实验三暴露的哪对矛盾？
2. batch_size=32 + accumulation_steps=8 等效于什么？为什么不直接用 batch_size=256？
3. `out/` 和 `checkpoints/` 里的两份存档各存了什么、给谁用？
4. `--from_weight` 和 `--from_resume` 的区别是什么？
5. 整条流水线（SFT→LoRA→DPO→蒸馏→RL）靠哪个参数把上一阶段的成果传给下一阶段？

<details>
<summary>自测答案（先自己答，再展开）</summary>

1. 大 lr 学得快但后期震荡（lr=10 爆炸），小 lr 稳但太慢（1e-6 装死）。先大后小：前期大步下山，后期小步精修。
2. 等效 batch_size=256。直接开 256 显存装不下（激活值随 batch 线性涨）；累积用"分 8 次攒梯度"骗过显存限制。
3. `out/`：半精度纯模型权重，给推理/下一阶段加载（小，128MB）；`checkpoints/`：全精度权重+优化器动量+epoch/step 进度，给断点续训（大，650MB）。优化器状态无法半精度（会破坏动量的精确性），且续训要求"分毫不差地回到现场"。
4. from_weight=只继承模型权重，从 step 0 重新开始（换数据/换阶段时用）；from_resume=权重+优化器+进度全套恢复，从断点继续（同任务断电续跑用）。
5. `--from_weight`：SFT 默认 `pretrain`，DPO 默认 `full_sft`……上一阶段的 out/*.pth 就是下一阶段的起跑线。

</details>

---

## 过关之后

第 4 步：SFT。剧透——`train_full_sft.py` 与 `train_pretrain.py` 的差异小到令人失望（数据类换名、默认 lr 1e-5、from_weight=pretrain），而正是这个"小差异 + 第1课的 -100 掩码"把"接龙机器"变成"对话助手"。
