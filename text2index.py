import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. 从文件中读取文本
file_path = "data/wallStreetWeek/WallStreetWeek20250314.txt"  # 替换为你的文件路径
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# 2. 预处理文本：去除多余换行符并分割句子
processed_text = re.sub(r'\n+', ' ', text)  # 替换换行符为空格
sentences = re.split(r'(?<=[.!?])\s+', processed_text)  # 按句子分割

# 3. 加载 Sentence Transformer 模型（用于生成句子向量）
model = SentenceTransformer('all-MiniLM-L6-v2')  # 轻量级模型，适合本地运行

# 4. 将句子编码为向量
sentence_vectors = model.encode(sentences)

# 5. 构建 FAISS 索引
dimension = sentence_vectors.shape[1]  # 向量维度（取决于模型）
index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离（欧式距离）
index.add(sentence_vectors)  # 添加向量到索引

# 6. 保存 FAISS 索引到磁盘
faiss.write_index(index, "sentences_faiss_index.index")

print("句子已成功存储到 FAISS 向量数据库！")
print(f"共存储 {len(sentences)} 个句子。")