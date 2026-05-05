# cloud/auto_train.py
import os
import time
import schedule
import subprocess
from pathlib import Path

# 配置
BASE_DIR = Path(__file__).parent
INCREMENTAL_IMAGES_DIR = BASE_DIR / "data" / "images" / "train"
LAST_TRAIN_FILE = BASE_DIR / ".last_train_timestamp"
TRAIN_SCRIPT = BASE_DIR / "train_better.py"

def check_new_data():
    """检查自上次训练后是否有新数据"""
    if not LAST_TRAIN_FILE.exists():
        return True
    last_time = os.path.getmtime(LAST_TRAIN_FILE)
    # 检查增量目录中是否有比 last_time 更新的文件
    for f in INCREMENTAL_IMAGES_DIR.glob("*.jpg"):
        if os.path.getmtime(f) > last_time:
            return True
    return False

def run_training():
    if not check_new_data():
        print("无新数据，跳过训练")
        return
    print("检测到新数据，开始增量训练...")
    # 调用 train_better.py，启用增量训练模式
    env = os.environ.copy()
    env["INCREMENTAL_TRAIN"] = "1"  # 通过环境变量通知训练脚本
    result = subprocess.run(["python", str(TRAIN_SCRIPT)], env=env)
    if result.returncode == 0:
        # 更新最后训练时间戳
        with open(LAST_TRAIN_FILE, 'w') as f:
            f.write(str(time.time()))
        print("增量训练完成，模型已发布到边缘端")
    else:
        print("训练失败，请检查日志")

if __name__ == "__main__":
    # 每天凌晨2点执行训练
    schedule.every().day.at("02:00").do(run_training)
    print("自动训练调度器启动，等待执行...")
    while True:
        schedule.run_pending()
        time.sleep(60)