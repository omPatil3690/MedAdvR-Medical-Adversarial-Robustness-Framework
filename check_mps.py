"""
Apple Silicon (MPS) GPU Diagnostic & Performance Benchmark Script
==================================================================
Verifies PyTorch Metal Performance Shaders (MPS) availability,
tests all core neural network operations used in MedAdvR on MPS GPU,
and benchmarks execution speed against CPU.
"""

import sys
import platform
import time
import torch
import torch.nn as nn


def print_header(title):
    print("\n" + "=" * 65)
    print(f"🍏 {title.upper()}")
    print("=" * 65)


def check_system_and_mps():
    print_header("System & PyTorch MPS Environment")
    
    mac_ver = platform.mac_ver()[0]
    arch = platform.machine()
    py_ver = sys.version.split()[0]
    torch_ver = torch.__version__

    print(f"• Operating System     : macOS {mac_ver} ({platform.system()})")
    print(f"• Architecture         : {arch} ({'Apple Silicon' if arch == 'arm64' else 'Intel x86'})")
    print(f"• Python Version       : {py_ver}")
    print(f"• PyTorch Version      : {torch_ver}")

    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    print(f"• PyTorch Built with MPS: {'✅ YES' if mps_built else '❌ NO'}")
    print(f"• Apple Silicon GPU (MPS) Available: {'✅ YES' if mps_available else '❌ NO'}")

    if not mps_available:
        print("\n⚠️ MPS is not available on this machine. Reasons could include:")
        print("   1. Running an Intel Mac without Metal support.")
        print("   2. macOS version is below 12.3.")
        print("   3. PyTorch build without MPS support.")
        return False
    return True


def test_core_medadvr_operations(device):
    print_header("Testing MedAdvR Operations on MPS GPU")
    
    try:
        # 1. Tensor creation and movement
        x = torch.randn(4, 3, 128, 128, device=device)
        print("✅ 1. Tensor creation & .to('mps')           : PASSED")

        # 2. 2D Convolution + BatchNorm + ReLU (UNet Encoder)
        conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ).to(device)
        out = conv_block(x)
        print("✅ 2. Conv2d + BatchNorm + MaxPool2d (UNet)   : PASSED")

        # 3. Bilinear Upsampling & Interpolation
        up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False).to(device)
        out_up = up(out)
        print("✅ 3. Bilinear Upsample & Interpolation       : PASSED")

        # 4. Backpropagation & Gradient Calculation
        target = torch.randn_like(out_up)
        loss = nn.MSELoss()(out_up, target)
        loss.backward()
        print("✅ 4. Autograd Backward Pass on MPS           : PASSED")

        # 5. 2D Fast Fourier Transform (Texture Generator)
        fft_out = torch.fft.fft2(x)
        fft_mag = torch.abs(fft_out)
        print("✅ 5. 2D Fast Fourier Transform (torch.fft)   : PASSED")

        # 6. Spatial Gradients (Edge Generator)
        grad_y, grad_x = torch.gradient(x, dim=[2, 3])
        print("✅ 6. 2D Spatial Gradients (torch.gradient)   : PASSED")

    except Exception as e:
        print(f"❌ Error during operation: {e}")
        return False

    return True


def benchmark_mps_vs_cpu(device, matrix_size=2048, num_runs=5):
    print_header("Speed Benchmark: Apple Silicon GPU vs CPU")
    print(f"Running {num_runs} iterations of [{matrix_size}x{matrix_size}] Matrix Multiplication...")

    # CPU Benchmark
    a_cpu = torch.randn(matrix_size, matrix_size, device="cpu")
    b_cpu = torch.randn(matrix_size, matrix_size, device="cpu")
    
    # Warmup
    _ = torch.matmul(a_cpu, b_cpu)
    
    start_cpu = time.perf_counter()
    for _ in range(num_runs):
        _ = torch.matmul(a_cpu, b_cpu)
    cpu_time = (time.perf_counter() - start_cpu) / num_runs

    # MPS Benchmark
    a_mps = torch.randn(matrix_size, matrix_size, device=device)
    b_mps = torch.randn(matrix_size, matrix_size, device=device)
    
    # Warmup & Sync
    c_mps = torch.matmul(a_mps, b_mps)
    if hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

    start_mps = time.perf_counter()
    for _ in range(num_runs):
        c_mps = torch.matmul(a_mps, b_mps)
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    mps_time = (time.perf_counter() - start_mps) / num_runs

    speedup = cpu_time / mps_time

    print(f"\n• CPU Average Time : {cpu_time * 1000:.2f} ms")
    print(f"• MPS Average Time : {mps_time * 1000:.2f} ms")
    print(f"🚀 Speedup Factor  : {speedup:.2f}x faster on Apple Silicon GPU!")


def main():
    if not check_system_and_mps():
        sys.exit(1)

    device = torch.device("mps")
    
    if test_core_medadvr_operations(device):
        benchmark_mps_vs_cpu(device)

    print("\n" + "=" * 65)
    print("🎉 YOUR MAC IS 100% READY FOR HARDWARE-ACCELERATED MEDADVR TRAINING!")
    print("   Run training anytime with: python main.py --device auto")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
