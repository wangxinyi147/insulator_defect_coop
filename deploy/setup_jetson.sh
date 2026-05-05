#!/bin/bash
# ==============================================
# Jetson Nano / Orin 边缘端环境一键配置脚本
# 适用: JetPack 4.6+ (Nano) / JetPack 5.0+ (Orin)
# 用法: chmod +x setup_jetson.sh && ./setup_jetson.sh
# ==============================================

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
echo -e "${GREEN}===== 绝缘子缺陷检测 - Jetson 边缘端环境配置 =====${NC}"

# ---- 1. 系统基础依赖 ----
echo -e "${YELLOW}[1/7] 安装系统依赖...${NC}"
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y libopenblas-dev libatlas-base-dev
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgtk2.0-dev libcanberra-gtk-module

# ---- 2. pip 升级 ----
echo -e "${YELLOW}[2/7] 升级 pip...${NC}"
python3 -m pip install --upgrade pip setuptools wheel

# ---- 3. Python 核心依赖 ----
echo -e "${YELLOW}[3/7] 安装 Python 依赖...${NC}"
pip3 install numpy==1.23.5
pip3 install opencv-python-headless==4.5.5.62
pip3 install Pillow>=8.0.0
pip3 install requests pyyaml

# ---- 4. ONNX Runtime GPU (Jetson ARM64) ----
echo -e "${YELLOW}[4/7] 安装 ONNX Runtime GPU...${NC}"
# 尝试从 pip 安装（部分版本支持 ARM64）
pip3 install onnxruntime-gpu 2>/dev/null || {
    echo -e "${YELLOW}pip 安装失败，尝试预编译 wheel...${NC}"
    # JetPack 4.6 兼容版本
    ONNX_URL="https://nvidia.box.com/shared/static/49fzcqa1g4obbwx6nytb2k7m0vz6wdip.whl"
    wget -O /tmp/onnxruntime_gpu.whl "$ONNX_URL" 2>/dev/null && \
        pip3 install /tmp/onnxruntime_gpu.whl || \
        echo -e "${RED}ONNX Runtime GPU 安装失败，回退到 CPU 版本${NC}"
    pip3 install onnxruntime 2>/dev/null
}

# ---- 5. 验证 TensorRT（JetPack 预装） ----
echo -e "${YELLOW}[5/7] 验证 TensorRT...${NC}"
python3 -c "import tensorrt; print(f'TensorRT {tensorrt.__version__}')" 2>/dev/null && \
    echo -e "${GREEN}TensorRT 可用${NC}" || \
    echo -e "${YELLOW}TensorRT 不可用（JetPack 预装应包含）${NC}"

# ---- 6. 性能优化 ----
echo -e "${YELLOW}[6/7] 开启 Jetson 最高性能模式...${NC}"
sudo nvpmodel -m 0                    # MAXN 模式（Nano: 4核 + GPU最大频率）
sudo jetson_clocks                    # 锁定最高时钟频率
echo -e "${GREEN}性能模式: $(nvpmodel -q | grep 'NV Power Mode')${NC}"

# ---- 7. 创建工作目录 ----
echo -e "${YELLOW}[7/7] 创建项目目录...${NC}"
mkdir -p ~/insulator_defect/edge/models
mkdir -p ~/insulator_defect/edge/cache
mkdir -p ~/insulator_defect/edge/feedback_cache
echo -e "${GREEN}项目目录已创建: ~/insulator_defect/${NC}"

# ---- 验证 ----
echo -e "\n${GREEN}===== 环境配置完成 =====${NC}"
echo -e "${GREEN}Python:${NC} $(python3 --version)"
echo -e "${GREEN}OpenCV:${NC} $(python3 -c 'import cv2; print(cv2.__version__)')"
echo -e "${GREEN}ONNX Runtime:${NC} $(python3 -c 'import onnxruntime; print(onnxruntime.__version__)' 2>/dev/null || echo '检查中...')"
echo -e "${GREEN}TensorRT:${NC} $(python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || echo '检查中...')"
echo -e "${GREEN}Jetson 温度:${NC} $(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 'N/A')"

echo -e "\n${YELLOW}提示: 请将模型文件拷贝至 ~/insulator_defect/edge/models/current.onnx${NC}"
echo -e "${YELLOW}后续运行: python3 edge/edge_inference.py --source 0${NC}"
