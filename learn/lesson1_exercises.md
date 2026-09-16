# 第1课 实操手册：Tokenizer 与数据（命令行版）

> 原则：**每个实验都先预测、再运行、再对答案**。预测错了才是学到东西的时刻。
> 参考结果都是我在这台机器上实际验证过的，你的数值可能有个位数 token 的浮动（数据预处理带一点随机性）。

---

## 准备（1 分钟）

```bash
cd /app/zdf/minimind
ls learn/        # 你应该看到 4 个文件：手册、2个演示脚本、1个实验室脚本
```

三个脚本各司其职：

| 文件 | 用途 |
|---|---|
| `learn/lesson1_tokenizer.py` | 观察版：把数据"解剖"给你看（已跑过） |
| `learn/lesson1b_training_loop.py` | 观察版：迷你训练循环，看 loss 下降（已跑过） |
| `learn/lesson1_lab.py` | **实验室版：留了 [TODO]，等你动手改** |

---

## 实验一：在 Python 交互式命令行里亲手把字变编号（10 分钟）

这不是跑脚本，是你在 REPL 里一行一行敲。先启动：

```bash
python
```

看到 `>>>` 提示符后，逐行粘贴（一次一行，回车）：

```python
import sys; sys.path.insert(0, '/app/zdf/minimind')
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/app/zdf/minimind/model')
tok.vocab_size
ids = tok('我爱秋天', add_special_tokens=False).input_ids
ids
[tok.decode([i]) for i in ids]
tok.decode(ids)
```

**动手任务（REPL 里继续）：**

1. 把 `我爱秋天` 换成你自己的名字、一个 emoji（如 `🚀`）、你的手机号，各跑一遍，先猜 token 数再看结果。
2. 试试 `tok('人工智能', add_special_tokens=False).input_ids` —— 4 个字只切了几个 token？为什么？
3. 试试 `tok('hi', add_special_tokens=False).input_ids` —— 为什么常见英文词 'hi' 反而被切成两个？
4. 玩够了输入 `exit()` 退出。

**参考结果**（我验证过的）：

| 输入 | 字符数 | token数 | 切分 |
|---|---|---|---|
| `🚀` | 1 | 3 | 3个字节片段（emoji 4字节，词表里没有 → 字节兜底） |
| `13800138000` | 11 | 6 | `'13'\|'8'\|'00'\|'13'\|'8'\|'000'`（数字按 BPE 规则切） |
| `Hello World` | 11 | 2 | `'Hello'\|' World'` |
| `人工智能` | 4 | 1 | 整个词单独占一个 token（训练语料里高频词） |
| `hi` | 2 | 2 | `'h'\|'i'`（词表只有 6400，装不下所有英文词） |

---

## 实验二：实验室脚本（30 分钟，核心）

```bash
python learn/lesson1_lab.py
```

先原样跑一遍，然后打开 `learn/lesson1_lab.py`，按 [TODO] 逐题修改，每改一题跑一次：

- **TODO 1a/1b**：往 `text_list` 加你自己的字符串。**先写下预测的 token 数，再跑**。
- **TODO 2a**：换 3 个不同的 `idx`（0~19999），记录"学习占比"。问题：为什么占比波动很大？
- **TODO 2b**：找到一条回答带 `<think>` 的样本（多换几个 idx）。思考：模型的"内心独白"要不要学？（答案：要——思考链也是 assistant 输出的一部分）
- **TODO 3a**：在 `ml` 列表里加 `32`。**先预测**"考卷被裁到 32 个 token 还有没有题可做"，再运行。

**参考结果**（idx=0，学习位置数）：

| max_length | 学习位置 | 结尾 `<|im_end|>` 被考到 |
|---|---|---|
| 64 | 19 | 是 |
| 256 | 193 | 是 |
| 768 | 414 | 是 |

思考：max_length 越小，学习位置越少——如果截断发生在回答中间，模型会学到"半句话"。这就是为什么训练脚本要选合适的 `max_seq_len`。

---

## 实验三：学习率三重奏（20 分钟，最震撼）

打开 `learn/lesson1b_training_loop.py`，找到学习率那一行（先 `grep -n AdamW learn/lesson1b_training_loop.py` 看行号），分别改成下面三个值，每次跑完记录最后的 loss。**跑之前先预测谁会学得最快：**

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)   # 改这个 lr
```

**参考结果**（同样 75 步，同一起点 8.78）：

| lr | 末步 loss | 解读 |
|---|---|---|
| `1e-6` | **8.758** | 几乎原地踏步：每步挪得太小，75 步什么都没学会 |
| `1e-3` | **7.098** | 健康下降：本次演示的最优区间 |
| `10.0` | **811.322** | **爆炸**！步子太大，参数被"改飞"，loss 冲天 |

体验完这三行数字，"学习率"就不再是抽象名词：**它是"每次改参数改多大"**——太小学不动，太大学飞了。

## 实验四：加大模型（10 分钟）

还是 `lesson1b_training_loop.py`，改这一行（grep -n 找 `MiniMindConfig`）：

```python
cfg = MiniMindConfig(hidden_size=256, num_hidden_layers=4)   # 原来 64 / 2
```

**参考结果**：末步 loss 7.098 → **6.891**（同数据同步数）。参数从 0.53M 涨到约 7M，同样的 75 步学得更多——这就是"模型容量"的直观感受。

---

## 验收自测（能口头回答这 5 题就算第 1 步过关）

1. 为什么任何一段普通文字都能直接变成训练题，不需要人工标注？
2. loss=8.76 意味着什么？从 8.76 降到 7.0，模型对正确答案的把握提升了几倍？
3. SFT 的 labels 里，-100 出现在哪些位置？为什么 assistant 的回答不标 -100？
4. 用你实验三的数据说明：学习率太大/太小分别会发生什么？
5. max_length 设得太小会有什么风险？

<details>
<summary>自测答案（先自己答，再展开）</summary>

1. "下一词预测"的标签就是原文自己右移一位，天然存在，所以海量文本免标注 → 这叫**自监督学习**。
2. loss≈-ln(猜对概率)。8.76≈ln(6400)，即 6400 选 1 的瞎蒙水平。降 1.76，把握提升 e^1.76≈5.8 倍。
3. 出现在：user 提问、`assistant\n` 前缀、结尾 padding。回答不标，因为 SFT 只考核"轮到模型说话时说什么"。
4. 太小（1e-6）：75 步后 loss 几乎不动（8.758），学不动；太大（10）：loss 飙到 811，参数被改飞，彻底学废。
5. 回答被从中间截断，模型学"半句话"甚至学不到任何回答；prompt 也可能被裁掉导致样本作废。

</details>

---

## 过关之后

下一步（第 2 步：模型结构）将解剖"学生本人"：`model/minimind` 的 8 层 Transformer 里，注意力如何让"看到秋"联想到"赌天"。准备好就说一声。
