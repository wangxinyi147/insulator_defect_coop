# cloud/data_collector/capture.py
"""多源图像采集适配器 - 无人机 (RTSP/视频流) + 固定摄像机 (USB/RTSP/IP)"""
import os
import time
import cv2
import threading
from pathlib import Path
from datetime import datetime
from .normalize import normalize_image, normalize_naming


class BaseCapture:
    def __init__(self, source_id="default", save_dir="cloud/data/images/collected"):
        self.source_id = source_id
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cap = None
        self._running = False

    def open(self):
        raise NotImplementedError

    def capture_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def save_frame(self, frame, source_type):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = normalize_naming(source_type, self.source_id, timestamp)
        img_path = self.save_dir / filename
        frame = normalize_image(frame)
        cv2.imwrite(str(img_path), frame)
        return str(img_path)

    def close(self):
        self._running = False
        if self.cap:
            self.cap.release()

    def continuous_capture(self, interval_seconds=2, max_frames=None):
        """连续采集模式"""
        self._running = True
        frame_count = 0
        source_type = getattr(self, 'source_type', 'unknown')
        print(f"[采集] 开始连续采集: {source_type}, 间隔={interval_seconds}s")
        while self._running:
            if max_frames and frame_count >= max_frames:
                break
            frame = self.capture_frame()
            if frame is not None:
                path = self.save_frame(frame, source_type)
                frame_count += 1
                print(f"[采集] {frame_count}: {path}")
            time.sleep(interval_seconds)
        print(f"[采集] 连续采集结束，共 {frame_count} 帧")

    def start_async(self, interval_seconds=2):
        t = threading.Thread(target=self.continuous_capture,
                            args=(interval_seconds, None), daemon=True)
        t.start()
        return t


class DroneCapture(BaseCapture):
    """无人机视角采集 - 通过 RTSP 流或视频文件获取航拍图像"""
    source_type = "drone"

    def __init__(self, source_id="drone_01", save_dir="cloud/data/images/collected",
                 rtsp_url=None, video_path=None):
        super().__init__(source_id, save_dir)
        self.rtsp_url = rtsp_url
        self.video_path = video_path

    def open(self):
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
        elif self.rtsp_url:
            self.cap = cv2.VideoCapture(self.rtsp_url)
        else:
            # 默认尝试摄像头
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError(f"无人机视频源打开失败: {self.rtsp_url or self.video_path or 'camera 0'}")
        print(f"[无人机采集] 视频源已打开: {self.source_id}")
        return self

    def capture_interval(self, interval_seconds=2):
        """按时间间隔采集（避免重复帧）"""
        return self.continuous_capture(interval_seconds)


class FixedCameraCapture(BaseCapture):
    """固定监控点采集 - USB摄像头 / RTSP 网络摄像头"""
    source_type = "fixed"

    def __init__(self, source_id="fixed_01", save_dir="cloud/data/images/collected",
                 camera_index=0, rtsp_url=None):
        super().__init__(source_id, save_dir)
        self.camera_index = camera_index
        self.rtsp_url = rtsp_url
        self._bg_subtractor = None

    def open(self):
        if self.rtsp_url:
            self.cap = cv2.VideoCapture(self.rtsp_url)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"固定摄像机打开失败: {self.rtsp_url or f'camera {self.camera_index}'}")
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=36)
        print(f"[固定摄像机] 已打开: {self.source_id}")
        return self

    def capture_on_motion(self, motion_threshold=500, cooldown_seconds=1.0):
        """运动检测触发采集"""
        self._running = True
        frame_count = 0
        last_capture_time = 0
        print(f"[运动检测] 开始监控: {self.source_id}, 运动阈值={motion_threshold}")
        while self._running:
            frame = self.capture_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            fg_mask = self._bg_subtractor.apply(frame)
            motion_level = cv2.countNonZero(fg_mask)
            now = time.time()
            if motion_level > motion_threshold and (now - last_capture_time) > cooldown_seconds:
                path = self.save_frame(frame, self.source_type)
                frame_count += 1
                last_capture_time = now
                cv2.putText(frame, f"Motion: {motion_level}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"[运动触发] #{frame_count}: motion={motion_level}, {path}")
            cv2.imshow(f"Monitor: {self.source_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.close()
        cv2.destroyAllWindows()
        print(f"[运动检测] 结束，共采集 {frame_count} 帧")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="多源图像采集工具")
    parser.add_argument("--source", choices=["drone", "fixed"], required=True)
    parser.add_argument("--rtsp", type=str, help="RTSP 流地址")
    parser.add_argument("--video", type=str, help="视频文件路径")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--interval", type=float, default=2, help="采集间隔(秒)")
    parser.add_argument("--motion", action="store_true", help="启用运动检测(固定摄像机)")
    parser.add_argument("--max-frames", type=int, default=100)
    args = parser.parse_args()

    if args.source == "drone":
        cap = DroneCapture(rtsp_url=args.rtsp, video_path=args.video).open()
        cap.continuous_capture(interval_seconds=args.interval, max_frames=args.max_frames)
    elif args.source == "fixed":
        cap = FixedCameraCapture(camera_index=args.camera, rtsp_url=args.rtsp).open()
        if args.motion:
            cap.capture_on_motion()
        else:
            cap.continuous_capture(interval_seconds=args.interval, max_frames=args.max_frames)
