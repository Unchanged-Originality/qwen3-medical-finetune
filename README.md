# Qwen3-Medical-SFT：医疗 R1 推理风格大模型微调



- **基础模型**：[Qwen3-1.7B](https://modelscope.cn/models/Qwen/Qwen3-1.7B/summary)
- **数据集**：[delicate_medical_r1_data](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data)
- **SwanLab**：[qwen3-sft-medical](https://swanlab.cn/@vic/qwen3-sft-medical/runs/fuodc2mawots2jbhzp610/chart)
- **微调方式**：LoRA微调
- **推理风格**：R1推理风格

## 安装环境

```bash
pip install -r requirements.txt
```

## 数据准备

自动完成数据集下载、预处理、验证集划分，生成`train.jsonl`和`val.jsonl`文件。

```bash
python data.py
```

## 训练

**LoRA微调**
```bash
python train_lora.py
```

## 推理

**LoRA微调**
```bash
python inference_lora.py
```
