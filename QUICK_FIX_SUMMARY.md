# 验证损失修复 - 快速总结

## 问题

验证损失显示为 0.0000，但训练损失正常（0.32左右）

## 原因

### 错误的代码逻辑

```python
# ❌ 错误：计算 (batch_size, batch_size) 相似度矩阵
similarity_scores = model.compute_similarity(user_emb, item_emb)

# ❌ 错误：扩展 targets 为矩阵
targets = batch['targets'].unsqueeze(1).expand(-1, batch_size)

# ❌ 错误：用 MSE Loss 计算矩阵差异
loss = criterion(similarity_scores, targets)  # 结果是0
```

### 为什么损失是0？

1. `similarity_scores` 是 L2归一化后嵌入的点积矩阵
2. 值域在 [-1, 1] 之间
3. MSE Loss 计算矩阵差值的平方，结果接近0

## 修复

### ✅ 正确的代码逻辑

```python
# ✅ 正确：计算每个样本的点积相似度
similarity_scores = torch.sum(user_embeddings * item_embeddings, dim=1)
# shape: (batch_size,) - 每个 (user, item) pair 的相似度

# ✅ 正确：使用原始 labels
labels = batch['labels'].float()

# ✅ 正确：用 BCE Loss
loss = F.binary_cross_entropy_with_logits(similarity_scores, labels)
```

### 修复位置

`src/training/trainer.py` 第 183-192 行

## 验证

重新运行训练后，应该看到：

```
Train Loss: 0.3134, Val Loss: 0.1234  # ✅ 不再是为0
```

而不是：

```
Train Loss: 0.3134, Val Loss: 0.0000  # ❌ 错误的
```

## 原理

- **训练集**: 包含正负样本（label=1或0），用BCE Loss ✅
- **验证集**: 只有正样本（label=1），用BCE Loss ✅
- **目标**: 让正样本的相似度分数接近1

## 快速检查

运行以下命令查看验证损失是否正常：

```bash
python scripts/training/train_two_tower.py
```

查看输出中的 `Val Loss`，应该 > 0

