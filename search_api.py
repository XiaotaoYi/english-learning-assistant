from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os
import sqlite3
import base64


# 创建数据库和表
def setup_database():
    conn = sqlite3.connect('english_learning.db')
    cursor = conn.cursor()
    
    # 创建translate表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS translate (
        base64_text TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        translate_text TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()
    print("数据库和表创建成功")

app = Flask(__name__)
CORS(app)  # 启用跨域请求支持

# 加载模型和索引
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("sentences_faiss_index.index")

# 从文件中读取原始句子
file_path = "data/wallStreetWeek/WallStreetWeek20250314.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()
import re
processed_text = re.sub(r'\n+', ' ', text)
sentences = re.split(r'(?<=[.!?])\s+', processed_text)

# DeepSeek API配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your_deepseek_api_key")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/translate"

@app.route('/')
def home():
    """渲染主页"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    
    # 将查询编码为向量
    query_vector = model.encode([query])
    
    # 在FAISS索引中搜索最相似的5个句子
    k = 5  # 返回前5个结果
    distances, indices = index.search(query_vector, k)
    
    # 获取对应的原始句子
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(sentences):
            results.append({
                'id': int(idx),
                'sentence': sentences[idx],
                'score': float(distances[0][i])
            })
    
    return jsonify(results)

@app.route('/api/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text', '')
    
    api_url = "https://api.deepseek.com/v1/chat/completions"
    api_key = DEEPSEEK_API_KEY
    
    prompt = f"Translate the following text into 中文:\n\n{text}"

     # 将文本转换为base64编码作为缓存键
    base64_text = base64.b64encode(text.encode('utf-8')).decode('utf-8')
    
    # 连接数据库
    conn = sqlite3.connect('english_learning.db')
    cursor = conn.cursor()
    
    # 查询缓存
    cursor.execute("SELECT translate_text FROM translate WHERE base64_text = ?", (base64_text,))
    cached_result = cursor.fetchone()
    
    if cached_result:
        # 如果找到缓存的翻译，直接返回
        conn.close()
        return jsonify({"translation": cached_result[0], "source": "cache"})
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3  # 控制随机性，越低翻译越稳定
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code == 200:
            translated_text = response.json()["choices"][0]["message"]["content"]
            # 将结果保存到数据库
            cursor.execute(
                "INSERT INTO translate (base64_text, text, translate_text) VALUES (?, ?, ?)",
                (base64_text, text, translated_text.strip())
            )
            conn.commit()
            conn.close()

            return jsonify({"translation": translated_text.strip()})
    except Exception as e:
        return jsonify({"error": str(e), "translation": "翻译服务暂时不可用"}), 500

if __name__ == '__main__':
    setup_database()
    app.run(debug=True, host='0.0.0.0', port=8082) 