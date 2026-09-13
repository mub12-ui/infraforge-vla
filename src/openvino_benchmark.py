"""
InfraForge VLA — OpenVINO Inference Benchmark

Satisfies the brief's "Intel Inference Benchmark Script" deliverable:
a bench-test that runs on Intel hardware and reports latency,
throughput, device selection, and model precision.

This benchmarks whatever policy model ends up driving the VLA
pipeline. For now it uses a small placeholder CNN (same-shaped
input/output as a lightweight vision policy head) so the benchmark
harness itself is proven correct before the real trained policy
exists -- swap `build_dummy_model()` for the real exported policy
once training/fine-tuning is done; everything else in this script
stays the same.

Run this on the actual target machine (Intel Core Ultra Series 2/3
for full brief compliance, or in the meantime on your Iris Xe
i7-1185G7 for real CPU/iGPU numbers).
"""

import time
import json
import numpy as np
import openvino as ov

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def build_dummy_model_onnx(path: str, input_shape=(1, 3, 64, 64)):
    """
    Builds a small CNN with the same rough shape as a lightweight
    vision-policy head (image in, small action vector out) and
    exports it to ONNX, so we have something real to benchmark
    before the trained policy exists.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required to build the placeholder model for this benchmark")

    class TinyPolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 16)  # 16 = 14-dim action space + headroom

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = TinyPolicyNet()
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model, dummy_input, path,
        input_names=["observation"], output_names=["action"],
        opset_version=13,
        dynamo=False,  # use the legacy TorchScript-based exporter (no onnxscript dependency)
    )
    return path


def benchmark_on_device(compiled_model, input_shape, device_name, n_warmup=10, n_runs=100):
    infer_request = compiled_model.create_infer_request()
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # Warmup (first-run JIT/compile overhead shouldn't pollute the measurement)
    for _ in range(n_warmup):
        infer_request.infer({0: dummy_input})

    latencies_ms = []
    for _ in range(n_runs):
        start = time.perf_counter()
        infer_request.infer({0: dummy_input})
        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms = np.array(latencies_ms)
    return {
        "device": device_name,
        "mean_latency_ms": float(latencies_ms.mean()),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "throughput_fps": float(1000 / latencies_ms.mean()),
        "n_runs": n_runs,
    }


def main():
    core = ov.Core()
    available_devices = core.available_devices
    print("Available OpenVINO devices:", available_devices)

    onnx_path = "tiny_policy.onnx"
    input_shape = (1, 3, 64, 64)
    build_dummy_model_onnx(onnx_path, input_shape)
    print(f"Exported placeholder model to {onnx_path}")

    model = core.read_model(onnx_path)

    results = []
    for device in available_devices:
        print(f"\nCompiling and benchmarking on {device}...")
        compiled = core.compile_model(model, device)
        result = benchmark_on_device(compiled, input_shape, device)
        results.append(result)
        print(f"  {device}: mean={result['mean_latency_ms']:.3f} ms  "
              f"p95={result['p95_latency_ms']:.3f} ms  "
              f"throughput={result['throughput_fps']:.1f} fps")

    print("\n" + "=" * 60)
    print("SUMMARY (precision: FP32, unquantized placeholder model)")
    print("=" * 60)
    for r in results:
        print(f"{r['device']:8s}  mean={r['mean_latency_ms']:8.3f} ms  "
              f"throughput={r['throughput_fps']:8.1f} fps  (n={r['n_runs']})")
    print("\nNOTE: this benchmarks a placeholder model to validate the harness.")
    print("Swap build_dummy_model_onnx() for the real trained/exported policy")
    print("once fine-tuning is complete, and consider INT8 quantization via")
    print("OpenVINO's NNCF for a real precision comparison row.")

    with open("openvino_benchmark_results.json", "w") as f:
        json.dump({
            "available_devices": available_devices,
            "precision": "FP32 (placeholder model)",
            "results": results,
        }, f, indent=2)
    print("\nWrote openvino_benchmark_results.json")


if __name__ == "__main__":
    main()
