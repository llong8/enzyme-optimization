FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 安装基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件到容器中
COPY requirements.txt .
COPY README.md .
COPY src ./src

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建结果目录
RUN mkdir -p results/data results/models results/plots

# 设置环境变量
ENV PYTHONPATH=/app

# 对外暴露Jupyter端口
EXPOSE 8888

# 默认命令
CMD ["python", "src/main.py"] 