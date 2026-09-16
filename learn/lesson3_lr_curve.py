"""第3课：学习率调度曲线 —— train_pretrain.py 实际使用的"余弦退火"
用法：python learn/lesson3_lr_curve.py
公式来自 trainer/trainer_utils.py:40：lr * (0.1 + 0.45*(1 + cos(pi * 进度)))
"""
import math

BASE = 5e-4    # train_pretrain.py 默认 --learning_rate
TOTAL = 625    # 本项目 smoke 数据、batch_size=32、1 个 epoch 的总步数

def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))

print('训练进度   当前学习率    占峰值比例   图示（每格≈2%峰值）')
print('-' * 72)
for frac in [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1]:
    v = get_lr(frac * TOTAL, TOTAL, BASE)
    bar = '█' * round(v / BASE * 50)
    print(f'  {frac:>4.0%}    {v:.2e}    {v / BASE:6.1%}     {bar}')

print()
print('解读：开局用大步子快速下山，越接近终点步子越小——')
print('前期大学习率快速接近"好区域"，后期小学习率在谷底精细打磨，不来回震荡。')
print('对照你第1课实验三的教训：恒定大学习率(lr=10)会把 loss 顶到 811，')
print('恒定小学习率(1e-6)又学不动——所以生产训练都让学习率"先大后小"地变化。')
