# AI4S 多层介质薄膜光谱预测与辅助设计

## 个人信息
- 学号：270159
- 目标波长 λ_target：730 nm
- 随机种子 seed：270159
- 设计种子 design_seed：270160

## 项目简介
本项目使用传输矩阵法（TMM）生成 Air/H/L/H/L/Glass 四层介质膜的“膜厚–反射光谱”数据，训练 MLP 代理模型，并用 MLP 快速筛选目标波长 730 nm 下的候选膜系，最终由 TMM 重新验证。

## 环境配置
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
