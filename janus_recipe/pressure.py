"""
使用方法：
python math_recipe/press.py
python math_recipe/press.py --size 32768

#### 推荐参数参考
- T4 (16GB): `--size 8192` 或 `--size 16384`
- V100 (32GB): `--size 16384`
- A100 / A800 (80GB): `--size 32768` (这会占用大量显存并产生极高的计算密度)
"""

import argparse
import sys

import torch
import torch.multiprocessing as mp


def stress_single_gpu(gpu_id, matrix_size, dtype=torch.float16):
    """
    针对单张 GPU 的压力测试函数
    """
    # 设置当前进程可见的 GPU
    device = torch.device(f"cuda:{gpu_id}")

    try:
        # 获取显卡信息
        props = torch.cuda.get_device_properties(device)
        print(f"[GPU {gpu_id}] ({props.name}) 准备就绪。")

        # 启用 Tensor Core 精度设置 (关键步骤)
        torch.set_float32_matmul_precision("medium")

        # 初始化数据
        # 计算显存占用估算 (3个矩阵: A, B, C)
        mem_usage_mb = (3 * (matrix_size**2) * 2) / (1024 * 1024)
        print(f"[GPU {gpu_id}] 初始化矩阵: {matrix_size}x{matrix_size} (预计占用显存: {mem_usage_mb:.2f} MB)")

        a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)

        # 预热 (Warm-up)
        for _ in range(10):
            torch.matmul(a, b)
        torch.cuda.synchronize(device)

        print(f"[GPU {gpu_id}] >>> 🚀 开始全速压测 (Tensor Core 模式)...")

        # 主循环
        while True:
            # 持续进行矩阵乘法
            torch.matmul(a, b)

            # 这里的逻辑是：不进行显式同步，让 CUDA 队列尽可能堆满指令
            # 只有当队列满了，CPU 才会阻塞，从而保证 GPU 始终有活干

    except RuntimeError as e:
        print(f"[GPU {gpu_id}] ❌ 发生错误 (可能是显存不足): {e}")
    except KeyboardInterrupt:
        pass  # 子进程静默退出


def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="多卡 GPU 压力测试脚本 (Tensor Core)")
    parser.add_argument(
        "--size", type=int, default=8192, help="矩阵大小 (N x N)。默认 8192。增大此值会增加显存占用和单次计算负载。"
    )
    args = parser.parse_args()

    matrix_size = args.size

    # 2. 检测 GPU 数量
    if not torch.cuda.is_available():
        print("❌ 错误: 未检测到 GPU！")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    print("========================================")
    print(f"检测到 {num_gpus} 张 GPU")
    print(f"矩阵尺寸: {matrix_size} x {matrix_size}")
    print("数据类型: float16 (开启 Tensor Core)")
    print("========================================")

    # 3. 启动多进程
    # 使用 'spawn' 启动方式在 CUDA 环境下更安全且兼容性更好
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes = []
    for i in range(num_gpus):
        p = mp.Process(target=stress_single_gpu, args=(i, matrix_size))
        p.start()
        processes.append(p)

    print("\n所有进程已启动。正在运行中...")
    print("请在一个新终端运行 'nvidia-smi dmon' 查看实时负载。")
    print("按 Ctrl+C 停止所有测试。\n")

    # 4. 优雅退出的处理
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n\n[主进程] 接收到停止信号，正在终止所有子进程...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        print("[主进程] 已退出。")


if __name__ == "__main__":
    main()
