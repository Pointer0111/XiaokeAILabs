import torch
from transformers import AutoModel

# 定义模型名称，可以是本地路径或Hugging Face模型ID
MODEL_NAME = "BAAI/BGE-VL-large"

# 加载预训练模型，必须设置trust_remote_code=True以运行远程代码
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
# 设置模型处理器，用于处理输入的图像和文本
model.set_processor(MODEL_NAME)
# 将模型设置为评估模式，不计算梯度
model.eval()

from FlagEmbedding.inference.embedder.model_mapping import AUTO_EMBEDDER_MAPPING

with open("model_mapping.txt", "w") as f:
    for key in AUTO_EMBEDDER_MAPPING.keys():
        f.write(key + "\n")


# 使用torch.no_grad()上下文管理器，在推理阶段禁用梯度计算以提高速度和减少内存使用
with torch.no_grad():
    # 编码查询，包含图像和文本描述
    query = model.encode(
        images = "datas/test_bge_vl/cat.png", 
    )

    # 编码候选图像，不包含文本描述
    candidates = model.encode(
        images = ["datas/test_bge_vl/cat.png", "datas/test_bge_vl/dog.png"]
    )
    
    # 计算查询向量与候选向量之间的相似度分数（点积）
    scores = query @ candidates.T
# 打印相似度分数
print(scores)


# 使用torch.no_grad()上下文管理器，在推理阶段禁用梯度计算以提高速度和减少内存使用
with torch.no_grad():
    # 编码查询，包含图像和文本描述
    query = model.encode(
        images = "datas/test_bge_vl/cat.png", 
        text = "将背景变暗，就像相机在夜间拍摄的照片一样"  # 中文文本描述
    )

    # 编码候选图像，不包含文本描述
    candidates = model.encode(
        images = ["datas/test_bge_vl/cat.png", "datas/test_bge_vl/dog.png"]
    )
    
    # 计算查询向量与候选向量之间的相似度分数（点积）
    scores = query @ candidates.T
# 打印相似度分数
print(scores)


# 使用torch.no_grad()上下文管理器，在推理阶段禁用梯度计算以提高速度和减少内存使用
with torch.no_grad():
    # 编码查询，包含图像和文本描述
    query = model.encode(
        images = "datas/test_bge_vl/cat.png", 
        text = "图片里面是一只狗"  # 中文文本描述
    )

    # 编码候选图像，不包含文本描述
    candidates = model.encode(
        images = ["datas/test_bge_vl/cat.png", "datas/test_bge_vl/dog.png"]
    )
    
    # 计算查询向量与候选向量之间的相似度分数（点积）
    scores = query @ candidates.T
# 打印相似度分数
print(scores)


# 使用torch.no_grad()上下文管理器，在推理阶段禁用梯度计算以提高速度和减少内存使用
with torch.no_grad():
    # 编码查询，包含图像和文本描述
    query = model.encode(
        images = "datas/test_bge_vl/cat.png", 
        text = "The image contains a dog"  # English text description
    )

    # 编码候选图像，不包含文本描述
    candidates = model.encode(
        images = ["datas/test_bge_vl/cat.png", "datas/test_bge_vl/dog.png"]
    )
    
    # 计算查询向量与候选向量之间的相似度分数（点积）
    scores = query @ candidates.T
# 打印相似度分数
print(scores)
