# edge/benchmark.py
"""
边缘推理基准测试工具
对比 ONNX Runtime vs TensorRT 在目标设备上的推理性能
生成毕设所需的性能对比表格
"""
import os
import sys
import time
import cv2
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class InferenceBenchmark:
    def __init__(self, model_path, img_size=640, warmup=10, iterations=100):
        self.model_path = model_path
        self.img_size = img_size
        self.warmup = warmup
        self.iterations = iterations
        self.backend = self._detect_backend(model_path)

    def _detect_backend(self, model_path):
        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".engine":
            return "tensorrt"
        elif ext == ".onnx":
            return "onnx"
        elif ext == ".pt":
            return "pytorch"
        else:
            raise ValueError(f"不支持的模型格式: {ext}")

    def _create_dummy_image(self):
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    def _load_model(self):
        if self.backend == "tensorrt":
            from edge.trt_inference import TRTInference
            return TRTInference(self.model_path)
        elif self.backend == "onnx":
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            return ort.InferenceSession(self.model_path, providers=providers)
        elif self.backend == "pytorch":
            from ultralytics import YOLO
            return YOLO(self.model_path)

    def run(self):
        print(f"\n{'='*60}")
        print(f"边缘推理基准测试")
        print(f"模型: {self.model_path}")
        print(f"后端: {self.backend}")
        print(f"预热: {self.warmup} 次 | 测试: {self.iterations} 次")
        print(f"{'='*60}\n")

        model = self._load_model()
        dummy_img = self._create_dummy_image()

        # 预热
        print(f"预热中 ({self.warmup} 次)...")
        for _ in range(self.warmup):
            self._infer(model, dummy_img)

        # 正式测试
        print(f"测试中 ({self.iterations} 次)...")
        latencies = []
        for i in range(self.iterations):
            t0 = time.perf_counter()
            results = self._infer(model, dummy_img)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{self.iterations}")

        latencies = np.array(latencies)
        throughput = 1000.0 / np.mean(latencies)

        report = {
            "model_path": self.model_path,
            "backend": self.backend,
            "iterations": self.iterations,
            "warmup": self.warmup,
            "avg_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "fps": float(throughput),
            "std_ms": float(np.std(latencies)),
        }

        print(f"\n{'='*60}")
        print(f"基准测试结果")
        print(f"{'='*60}")
        print(f"  平均延迟:   {report['avg_latency_ms']:.2f} ms")
        print(f"  中位延迟:   {report['median_latency_ms']:.2f} ms")
        print(f"  P95 延迟:   {report['p95_latency_ms']:.2f} ms")
        print(f"  P99 延迟:   {report['p99_latency_ms']:.2f} ms")
        print(f"  吞吐量:     {report['fps']:.2f} FPS")
        print(f"  标准差:     {report['std_ms']:.2f} ms")
        print(f"{'='*60}\n")

        return report

    def _infer(self, model, img):
        if self.backend == "tensorrt":
            return model.infer(img)
        elif self.backend == "onnx":
            from edge.edge_inference import preprocess_image
            input_tensor, _, _ = preprocess_image(img, (self.img_size, self.img_size))
            input_name = model.get_inputs()[0].name
            return model.run(None, {input_name: input_tensor})
        elif self.backend == "pytorch":
            return model.predict(img, imgsz=self.img_size, verbose=False)


def compare_models(onnx_path, trt_path=None, pt_path=None, output_json=None):
    """对比多个后端/模型的性能"""
    results = []
    models = [onnx_path]
    if trt_path and os.path.exists(trt_path):
        models.append(trt_path)
    if pt_path and os.path.exists(pt_path):
        models.append(pt_path)

    for model_path in models:
        try:
            bench = InferenceBenchmark(model_path, iterations=50)
            result = bench.run()
            results.append(result)
        except Exception as e:
            print(f"跳过 {model_path}: {e}")

    if len(results) >= 2:
        speedup = results[0]["avg_latency_ms"] / results[1]["avg_latency_ms"]
        print(f"\n加速比 (ONNX/TensorRT): {speedup:.2f}x")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存至: {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绝缘子缺陷检测 - 边缘推理基准测试")
    parser.add_argument("--model", type=str, default="edge/models/current.onnx",
                       help="模型路径 (.pt / .onnx / .engine)")
    parser.add_argument("--trt-model", type=str, default=None,
                       help="对比 TensorRT 模型路径")
    parser.add_argument("--output", type=str, default="benchmark_result.json",
                       help="结果输出 JSON 路径")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if args.trt_model:
        compare_models(args.model, args.trt_model, output_json=args.output)
    else:
        bench = InferenceBenchmark(args.model, warmup=args.warmup, iterations=args.iterations)
        result = bench.run()
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
