# 损失函数修复说明

## 问题诊断

### 症状
训练损失一直卡在0.313左右不下降，所有epoch的损失都完全相同：
```
Train Loss: 0.3133, Val Loss: 0.3133
```

### 根本原因

1. **L2归一化导致相似度范围受限**
   - 模型输出L2归一化的嵌入向量（模长为1）
   - 点积相似度范围：`[-1, 1]`
   - BCE Loss期望的logits应该在`[-∞, +∞]`范围

2. **BCE Loss无法有效工作**
   - 由于logits在[-1, 1]范围，BCE始终输出固定损失
   - 当所有预测相似度都在某个值附近时，损失停滞不前

3. **梯度无法有效传播**
   - L2归一化后，梯度无法充分传播到嵌入层
   - 模型参数更新缓慢或停滞

## 解决方案

### 添加可学习的Logit Scale参数

引入一个可学习的温度参数（logit scale），将相似度分数缩放到更大范围：

```python
class TwoTowerModel(nn.Module):
    def __init__(self, ..., logit_scale=10.0):
        # 添加可学习参数
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))
        
    def compute_similarity_dot(self, user_emb, item_emb):
        # 裁剪到合理范围
        self.logit_scale.data = torch.clamp(self.logit_scale.data, min=0.01, max=100.0)
        
        # 计算相似度并缩放
        similarity = torch.sum(user_emb * item_emb, dim=1)
        return similarity * self.logit_scale
```

### 修改损失计算

```python
# 原来：直接使用点积
similarity = torch.sum(user_emb * item_emb, dim=1)  # 范围 [-1, 1]
loss = F.binary_cross_entropy_with_logits(similarity, labels)

# 现在：使用缩放后的相似度
similarity_scores = model.compute_similarity_dot(user_emb, item_emb)  # 范围 [0.01, 100.0]
loss = F.binary_cross_entropy_with_logits(similarity_scores, labels)
```

## 为什么这样修复有效？

### 1. Logit Scale的作用

- **扩展相似度范围**: 将 [-1, 1] 扩展到 [-10, 10] 或更大
- **增强梯度信号**: 更大的数值范围使得梯度能够有效传播
- **可学习性**: 模型可以学习最优的缩放因子

### 2. BCE Loss的工作原理

BCE Loss的公式：
```
loss = -[y * log(sigmoid(logit)) + (1-y) * log(1-sigmoid(logit))]
```

其中 sigmoid 把 logits 映射到 [0, 1]。

当 logits 范围太小时（如[-1, 1]），sigmoid 输出都很接近，损失对梯度不敏感。
当 logits 范围较大时（如[-10, 10]），sigmoid 可以区分不同置信度，梯度更有效。

### 3. 实验验证

修复后，你应该看到：
```
Train Loss: 0.3133 -> 0.25 -> 0.20 -> ...  # 持续下降
Val Loss: 0.3133 -> 0.26 -> 0.21 -> ...    # 也下降
```

## 修改的文件

1. `src/models/two_tower_model.py`
   - 添加 `logit_scale` 参数
   - 添加 `compute_similarity_dot()` 方法
   - 修改 `compute_similarity()` 使用缩放

2. `src/training/trainer.py`
   - 修改 `_compute_positive_negative_loss()` 使用新的相似度计算
   - 修改 `validate_epoch()` 使用新的相似度计算

3. `scripts/training/train_two_tower.py`
   - 添加 `logit_scale` 参数传递

4. `config/training_config.yaml`
   - 添加 `logit_scale: 10.0` 配置

5. `test_train.py`
   - 更新模型初始化调用

## 推荐配置

```yaml
model:
  logit_scale: 10.0  # 初始值，模型会自动学习最优值
```

建议范围：5.0 - 20.0

## 预期结果

修复后重新训练，你应该看到：

1. **损失持续下降**
   ```
   Epoch 1: Train Loss 0.31, Val Loss 0.31
   Epoch 2: Train Loss 0.28, Val Loss 0.29
   Epoch 3: Train Loss 0.25, Val Loss 0.27
   ...
   ```

2. **Logit scale自动调整**
   - 初始值：10.0
   - 训练后：可能会增长到 15-20 或更大

3. **更好的泛化**
   - 训练和验证损失都在下降
   - 模型真正在学习

## 技术细节

### Logit Scale的约束

```python
self.logit_scale.data = torch.clamp(self.logit_scale.data, min=0.01, max=100.0)
```

- 防止值过大导致数值不稳定
- 防止值过小导致梯度消失
- 保持在合理的学习范围内

### 为什么初始值设为10？

- 对于归一化嵌入的点积（范围[-1,1]），10倍的缩放可以产生[-10, 10]的logits
- 这个范围足以让BCE Loss有效工作
- 模型可以进一步学习和调整

## 测试建议

重新运行训练前，建议先运行测试：

```bash
python test_train.py
```

确认：
1. 模型能够初始化
2. Logit scale参数存在
3. Forward pass 正常工作
4. Loss 计算正常

然后重新训练，你应该会看到损失正常下降了！

