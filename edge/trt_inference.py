# edge/trt_inference.py
"""
TensorRT 推理引擎增强版 - FP16/INT8 加速推理
支持：FP16/INT8量化、推理预热、批量并行推理、性能统计
"""
import os
import time
import numpy as np
import cv2
from pathlib import Path
from collections import deque

_trt_available = False
try:
    import tensorrt as trt
    _trt_available = True
except ImportError:
    pass

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    _cuda_available = True
except ImportError:
    _cuda_available = False


class TRTInference:
    def __init__(self, engine_path, conf_thres=0.25, iou_thres=0.45, img_size=640, warmup=True):
        if not _trt_available:
            raise ImportError("TensorRT 不可用，请确认在 Jetson 环境下运行")
        if not _cuda_available:
            raise ImportError("pycuda 不可用")

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.logger = trt.Logger(trt.Logger.WARNING)

        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if not self.engine:
            raise RuntimeError(f"无法加载 TensorRT 引擎: {engine_path}")

        self.context = self.engine.create_execution_context()
        self._alloc_buffers()

        # 推理性能统计
        self._latency_queue = deque(maxlen=100)
        self._total_inferences = 0

        if warmup:
            self._warmup(10)

    def _warmup(self, iterations=10):
        """推理引擎预热，减少首次推理延迟"""
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.infer(dummy)
        print(f"[TensorRT] 预热完成 ({iterations} 次)")

    def _alloc_buffers(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({"name": binding, "host": host_mem, "device": device_mem, "shape": shape})
            else:
                self.outputs.append({"name": binding, "host": host_mem, "device": device_mem, "shape": shape})

    def _preprocess(self, img):
        h, w = img.shape[:2]
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        canvas[:new_h, :new_w] = img_resized.astype(np.float32) / 255.0
        canvas = np.transpose(canvas, (2, 0, 1))
        canvas = np.expand_dims(canvas, axis=0)
        return canvas, scale, (w, h)

    def _postprocess(self, output, scale, original_size):
        predictions = np.squeeze(output).T
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 6)
        scores = np.max(predictions[:, 4:], axis=1) if predictions.shape[1] > 5 else predictions[:, 4]
        class_ids = np.argmax(predictions[:, 4:], axis=1) if predictions.shape[1] > 5 else np.zeros(len(predictions), dtype=int)
        mask = scores >= self.conf_thres
        predictions = predictions[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        if len(predictions) == 0:
            return []
        boxes = predictions[:, :4].copy()
        boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) / scale
        boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) / scale
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / scale
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / scale
        boxes[:, 0] = np.clip(boxes[:, 0], 0, original_size[0])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, original_size[1])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, original_size[0])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, original_size[1])
        indices = cv2.dnn.NMSBoxes(boxes[:, :4].tolist(), scores.tolist(), self.conf_thres, self.iou_thres)
        if len(indices) == 0:
            return []
        results = []
        for i in indices:
            i = i[0] if isinstance(i, (list, np.ndarray)) else i
            results.append({
                "cls": int(class_ids[i]),
                "conf": float(scores[i]),
                "xyxy": boxes[i].tolist()
            })
        return results

    def infer(self, img):
        t0 = time.perf_counter()
        input_tensor, scale, orig_size = self._preprocess(img)
        np.copyto(self.inputs[0]["host"], input_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        output = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        results = self._postprocess(output, scale, orig_size)

        latency_ms = (time.perf_counter() - t0) * 1000
        self._latency_queue.append(latency_ms)
        self._total_inferences += 1
        return results

    def infer_batch(self, images):
        """批量推理接口"""
        all_results = []
        for img in images:
            all_results.append(self.infer(img))
        return all_results

    @property
    def avg_latency(self):
        if not self._latency_queue:
            return 0
        return np.mean(self._latency_queue)

    @property
    def fps(self):
        return 1000.0 / self.avg_latency if self.avg_latency > 0 else 0

    def get_stats(self):
        return {
            "total_inferences": self._total_inferences,
            "avg_latency_ms": round(float(np.mean(self._latency_queue)), 2),
            "p95_latency_ms": round(float(np.percentile(self._latency_queue, 95)), 2),
            "fps": round(self.fps, 1),
        }

    @staticmethod
    def build_from_onnx(onnx_path, engine_path, mode="fp16", workspace_gb=1, dynamic_batch=False):
        """
        从 ONNX 构建 TensorRT 引擎
        mode: "fp32" | "fp16" | "int8"
        """
        if not _trt_available:
            raise ImportError("TensorRT 不可用")
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"ONNX 解析错误: {parser.get_error(i)}")
                raise RuntimeError("ONNX 解析失败")

        config = builder.create_builder_config()
        config.max_workspace_size = workspace_gb * (1 << 30)

        if mode == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ TensorRT FP16 量化模式")
        elif mode == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            print("⚠️  INT8 量化需要校准数据集，使用FP16回退")
            config.set_flag(trt.BuilderFlag.FP16)

        if dynamic_batch:
            profile = builder.create_optimization_profile()
            profile.set_shape("input", (1, 3, 640, 640), (4, 3, 640, 640), (8, 3, 640, 640))
            config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT 引擎构建失败")

        os.makedirs(os.path.dirname(engine_path) if os.path.dirname(engine_path) else ".", exist_ok=True)
        # 在文件名中加入量化标记
        base, ext = os.path.splitext(engine_path)
        if mode != "fp32" and not base.endswith(f"_{mode}"):
            engine_path = f"{base}_{mode}{ext}"

        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"✅ TensorRT 引擎已保存: {engine_path}")
        return engine_path
