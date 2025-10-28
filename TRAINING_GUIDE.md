# Two Tower Model 训练指南

## 📁 项目结构

```
Final-Project/
├── src/
│   ├── models/
│   │   └── two_tower_model.py      # Two Tower 模型
│   ├── data/
│   │   └── data_loader.py           # 数据加载器
│   ├── training/
│   │   └── trainer.py               # 训练器
│   └── utils/
│       └── metrics.py               # 评估指标
├── scripts/
│   └── training/
│       └── train_two_tower.py       # 训练脚本
├── config/
│   └── training_config.yaml         # 配置文件
└── data/
    └── processed/                   # 处理后的数据
```

## 🚀 快速开始

### 1. 准备数据

确保 `data/processed/` 目录下有以下文件：
- `interactions_mapped.parquet` - 交互数据
- `uid2idx.json` - 用户ID映射
- `iid2idx.json` - 物品ID映射
- `splits.json` - 数据集分割
- `items_text_emb_*.json` - 物品文本嵌入
- `negatives.json` - 负样本（可选）

### 2. 配置训练参数

编辑 `config/training_config.yaml`:

```yaml
data:
  data_path: data/processed
  text_embedding_path: data/processed/items_text_emb_*.json
  neg_ratio: 1
  seed: 42
  emb_dim: 768

model:
  embedding_dim: 128
  text_feature_dim: 768
  user_mlp_hidden: [256, 128]
  item_mlp_hidden: [256, 128]
  dropout: 0.1

training:
  batch_size: 256
  num_epochs: 10
  learning_rate: 0.001
  weight_decay: 0.00001
```

### 3. 运行训练

```bash
# 方式1: 使用默认配置
python scripts/training/train_two_tower.py

# 方式2: 指定配置文件
python scripts/training/train_two_tower.py --config config/training_config.yaml

# 方式3: 测试训练流程（推荐先运行）
python test_train.py
```

### 4. 查看结果

训练完成后，结果保存在：
- `outputs/results/best_model.pth` - 最佳模型
- `outputs/results/training_results.yaml` - 训练结果
- `outputs/results/plots/training_history.png` - 训练曲线图

## 📊 数据流

```
DataLoader (data_loader.py)
    ↓
batch = {
    'user_features': {'user_id': tensor},
    'item_features': {'item_id': tensor, 'category': tensor, 'brand': tensor},
    'text_features': tensor,  # (B, 768)
    'labels': tensor,
    'ratings': tensor
}
    ↓
Model Forward (two_tower_model.py)
    ↓
user_emb, item_emb = model(
    user_features=batch['user_features'],
    item_features=batch['item_features'],
    text_features=batch['text_features']
)
    ↓
Trainer (trainer.py)
    ↓
loss = compute_loss(user_emb, item_emb, batch['labels'])
```

## 🔧 组件说明

### DataLoader (`src/data/data_loader.py`)

**输入**: 配置和数据文件
```python
processor = DataProcessor(config['data'])
train_ds, val_ds, test_ds = processor.process_data('data/processed')
```

**输出**: PyTorch Dataset，包含：
- `user_features` - 用户特征字典
- `item_features` - 物品特征字典
- `text_features` - 文本嵌入 (B, 768)
- `labels` - 标签
- `ratings` - 评分

### Model (`src/models/two_tower_model.py`)

**Two Tower 架构**:
- **User Tower**: 编码用户特征 → 用户嵌入
- **Item Tower**: 编码物品特征 + 文本嵌入 → 物品嵌入
- **相似度**: 计算用户和物品嵌入的余弦相似度

**初始化**:
```python
model = TwoTowerModel(
    user_feature_dims=processor.user_feature_dims,
    item_feature_dims=processor.item_feature_dims,
    text_feature_dim=processor.emb_dim,
    embedding_dim=128,
    ...
)
```

### Trainer (`src/training/trainer.py`)

**功能**:
- 训练/验证循环
- 优化器和学习率调度
- 模型保存/加载
- 训练历史可视化

**使用**:
```python
trainer = TwoTowerTrainer(model, train_loader, val_loader, config)
results = trainer.train()
```

## 🐛 常见问题

### 1. 缺少数据文件

**错误**: `FileNotFoundError: Expected interactions_mapped.parquet`

**解决**: 确保 `data/processed/` 目录下有所有必需的数据文件

### 2. CUDA 内存不足

**错误**: `CUDA out of memory`

**解决**: 减小 `batch_size`（在配置文件中）

### 3. 导入错误

**错误**: `ModuleNotFoundError: No module named 'src'`

**解决**: 从项目根目录运行脚本，或设置 PYTHONPATH

### 4. 文本嵌入维度不匹配

**错误**: 模型期望 768 维但得到其他维度

**解决**: 检查配置文件中的 `text_feature_dim` 设置

## 📝 训练流程详解

### Step 1: 数据处理
```python
processor = DataProcessor(data_config)
train_ds, val_ds, test_ds = processor.process_data(data_path)
```

### Step 2: 创建 DataLoader
```python
train_loader = create_data_loader(train_ds, batch_size=256, shuffle=True)
val_loader = create_data_loader(val_ds, batch_size=256, shuffle=False)
```

### Step 3: 初始化模型
```python
model = TwoTowerModel(
    user_feature_dims=processor.user_feature_dims,
    item_feature_dims=processor.item_feature_dims,
    text_feature_dim=768,
    embedding_dim=128,
    ...
)
```

### Step 4: 初始化训练器
```python
trainer = TwoTowerTrainer(model, train_loader, val_loader, config)
```

### Step 5: 开始训练
```python
results = trainer.train()
```

## 📈 监控训练

训练过程中会输出：
- 每个 epoch 的训练损失
- 验证损失和指标
- 最佳模型会自动保存

查看训练曲线：
```bash
# 训练曲线图保存在
outputs/results/plots/training_history.png
```

## 🔍 调试

### 测试数据处理
```python
python test_train.py
```

这会测试：
1. ✅ 数据加载
2. ✅ DataLoader 创建
3. ✅ 模型初始化
4. ✅ Forward 通过
5. ✅ 训练步骤

### 检查数据格式
```python
from src.data.data_loader import DataProcessor

processor = DataProcessor({'data_path': 'data/processed'})
train_ds, _, _ = processor.process_data()

print(f"User features: {processor.user_feature_dims}")
print(f"Item features: {processor.item_feature_dims}")
print(f"Text embedding dim: {processor.emb_dim}")
```

## 📦 输出文件

训练完成后会生成：

1. **best_model.pth** - PyTorch 模型检查点
   - `model_state_dict` - 模型参数
   - `config` - 训练配置
   - `train_losses` - 训练损失历史
   - `val_losses` - 验证损失历史
   - `val_metrics` - 验证指标历史

2. **training_results.yaml** - 训练结果摘要

3. **training_history.png** - 训练曲线可视化

## 🎯 下一步

训练完成后，你可以：

1. **加载模型进行推理**
```python
from src.models.two_tower_model import TwoTowerModel
import torch

checkpoint = torch.load('outputs/results/best_model.pth')
model = TwoTowerModel(...)
model.load_state_dict(checkpoint['model_state_dict'])
```

2. **生成推荐**
```python
user_emb = model.get_user_embeddings(user_features)
item_emb = model.get_item_embeddings(item_features, text_features)
similarities = model.compute_similarity(user_emb, item_emb)
top_items = similarities.argsort(descending=True)[:10]
```

3. **评估模型**
```python
# 运行测试集评估
python test_train.py
```

