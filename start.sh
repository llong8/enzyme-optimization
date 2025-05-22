#!/bin/bash

# 启动脚本，用于快速部署项目

# 检查是否安装了Docker和Docker Compose
if ! command -v docker &> /dev/null; then
    echo "错误: 未安装Docker。请先安装Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "错误: 未安装Docker Compose。请先安装: https://docs.docker.com/compose/install/"
    exit 1
fi

# 确保结果目录存在
mkdir -p results/data results/models results/plots

# 构建并启动容器
echo "正在启动酶工程容器..."
docker-compose up -d

# 显示服务信息
echo "容器已启动!"
echo "服务访问信息:"
echo "- Jupyter Lab: http://localhost:8888"
echo "- API服务: http://localhost:5000"
echo ""
echo "API测试命令:"
echo "curl -X GET http://localhost:5000/api/health"
echo ""
echo "停止服务命令:"
echo "docker-compose down" 