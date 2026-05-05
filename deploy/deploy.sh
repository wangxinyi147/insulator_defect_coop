#!/bin/bash
# ==============================================
# 绝缘子缺陷检测 - 一键部署到 Jetson 边缘端
# 用法: ./deploy.sh <JETSON_IP> [JETSON_USER]
# 示例: ./deploy.sh 192.168.1.100 jetson
# ==============================================

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

JETSON_IP="${1}"
JETSON_USER="${2:-jetson}"

if [ -z "$JETSON_IP" ]; then
    echo -e "${RED}用法: ./deploy.sh <JETSON_IP> [JETSON_USER]${NC}"
    echo -e "${RED}示例: ./deploy.sh 192.168.1.100 jetson${NC}"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo -e "${GREEN}===== 部署绝缘子缺陷检测系统到 Jetson =====${NC}"
echo -e "${GREEN}目标设备: ${JETSON_USER}@${JETSON_IP}${NC}"
echo -e "${GREEN}项目路径: ${PROJECT_ROOT}${NC}"

# ---- 1. 上传边缘端代码 ----
echo -e "${YELLOW}[1/5] 上传 edge 目录到 Jetson...${NC}"
ssh ${JETSON_USER}@${JETSON_IP} "mkdir -p ~/insulator_defect/edge"
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "${PROJECT_ROOT}/edge/" \
    "${JETSON_USER}@${JETSON_IP}:~/insulator_defect/edge/"

# ---- 2. 上传部署脚本 ----
echo -e "${YELLOW}[2/5] 上传部署配置...${NC}"
scp "${PROJECT_ROOT}/deploy/insulator-detect.service" \
    "${JETSON_USER}@${JETSON_IP}:/tmp/insulator-detect.service"

scp "${PROJECT_ROOT}/requirements-jetson.txt" \
    "${JETSON_USER}@${JETSON_IP}:~/insulator_defect/requirements-jetson.txt"

# ---- 3. 在 Jetson 上执行环境配置 ----
echo -e "${YELLOW}[3/5] 在 Jetson 上安装依赖...${NC}"
ssh ${JETSON_USER}@${JETSON_IP} "cd ~/insulator_defect && pip3 install -r requirements-jetson.txt" || \
    echo -e "${YELLOW}部分依赖可能需要手动安装${NC}"

# ---- 4. 安装 systemd 服务 ----
echo -e "${YELLOW}[4/5] 安装 systemd 服务...${NC}"
ssh ${JETSON_USER}@${JETSON_IP} "
    sudo cp /tmp/insulator-detect.service /etc/systemd/system/
    sudo sed -i 's|/home/jetson/insulator_defect|/home/${JETSON_USER}/insulator_defect|g' /etc/systemd/system/insulator-detect.service
    sudo systemctl daemon-reload
    sudo systemctl enable insulator-detect.service
"

# ---- 5. 验证部署 ----
echo -e "${YELLOW}[5/5] 验证部署...${NC}"
ssh ${JETSON_USER}@${JETSON_IP} "
    echo '=== 目录结构 ==='
    find ~/insulator_defect -type f | head -20
    echo ''
    echo '=== Python 环境 ==='
    python3 --version
    python3 -c 'import onnxruntime; print(f\"ONNX Runtime: {onnxruntime.__version__}\")' 2>/dev/null || echo 'ONNX Runtime: 未安装'
    python3 -c 'import tensorrt; print(f\"TensorRT: {tensorrt.__version__}\")' 2>/dev/null || echo 'TensorRT: 未安装'
    echo ''
    echo '=== service 状态 ==='
    systemctl status insulator-detect.service --no-pager 2>/dev/null || echo 'service 未启动（无模型文件）'
"

echo -e "\n${GREEN}===== 部署完成 =====${NC}"
echo -e "${GREEN}边缘端项目路径: ~/insulator_defect/${NC}"
echo -e "${YELLOW}提示:"
echo -e "  - 首次部署后运行: ssh ${JETSON_USER}@${JETSON_IP} 'cd ~/insulator_defect/edge && python3 edge_inference.py --source 0'"
echo -e "  - 查看服务状态: ssh ${JETSON_USER}@${JETSON_IP} 'sudo systemctl status insulator-detect'"
echo -e "  - 启动服务: ssh ${JETSON_USER}@${JETSON_IP} 'sudo systemctl start insulator-detect'"
echo -e "  - 查看日志: ssh ${JETSON_USER}@${JETSON_IP} 'journalctl -u insulator-detect -f'${NC}"
