
import sys
import time
import signal

from rknnpool.rknnpool_ld import rknnPoolExecutor
from func.func_yolov8_optimize import myFunc
from func.display_utils import DisplayManager
from func.camera_utils import OV13855Camera
from func.servo_ctrl import get_servo

# ====================================================================
# 配置参数
# ====================================================================
MODEL_PATH = "./rknnModel/best.rknn"
TPEs = 8                       # NPU 推理线程数
CAM_W, CAM_H = 800, 600        # 摄像头采集分辨率
CAM_FPS = 30                   # 摄像头帧率

# ====================================================================
# 全局退出标志
# ====================================================================
_quit = False


def _on_signal(signum, frame):
    global _quit
    _quit = True
    print("\n[Main] 收到退出信号，正在释放资源...")


signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)

# ====================================================================
# 启动诊断
# ====================================================================
print("=" * 50)
print("  飞凌 ELFBoard RV1126B — YOLOv8 实时检测")
print("=" * 50)
print(f"  摄像头: OV13855 MIPI CSI ({CAM_W}x{CAM_H}@{CAM_FPS}fps)")
print(f"  显示屏: 7寸 MIPI DSI (1024x600)")
print(f"  模型:   {MODEL_PATH}")
print(f"  推理线程: {TPEs}")
print("=" * 50)

DisplayManager.print_info()

# ====================================================================
# 初始化
# ====================================================================
print("\n[Main] 初始化摄像头...")
camera = OV13855Camera(width=CAM_W, height=CAM_H, fps=CAM_FPS)
if not camera.open():
    print("[Main] 摄像头初始化失败，退出", file=sys.stderr)
    sys.exit(-1)

print("[Main] 初始化 RKNN 推理池...")
pool = rknnPoolExecutor(rknnModel=MODEL_PATH, TPEs=TPEs, func=myFunc)

print("[Main] 初始化显示屏...")
display = DisplayManager("ELFBoard-YOLOv8", fullscreen=True)
print(f"[Main] 显示后端: {'Framebuffer' if display.is_framebuffer else 'OpenCV imshow'}")
print(f"[Main] 屏幕分辨率: {display.screen_size[0]}x{display.screen_size[1]}")

print("[Main] 初始化舵机...")
servo = get_servo()
servo.start()

# ====================================================================
# 预填充推理队列
# ====================================================================
print("[Main] 预填充推理队列...")
for i in range(TPEs + 1):
    ok, frame = camera.read()
    if not ok:
        print(f"[Main] 预填充第 {i} 帧失败", file=sys.stderr)
        camera.release()
        pool.release()
        display.release()
        sys.exit(-1)
    pool.put(frame)
print("[Main] 预填充完成，开始主循环...")

# ====================================================================
# 主循环
# ====================================================================
frames, loop_time, init_time = 0, time.time(), time.time()

while camera.is_opened and not _quit:
    frames += 1

    ok, frame = camera.read()
    if not ok:
        print("[Main] 摄像头读取失败，退出循环")
        break

    pool.put(frame)
    result, flag = pool.get()
    if not flag:
        break

    key = display.show(result)
    if key == ord('q'):
        break

    if frames % 30 == 0:
        elapsed = time.time() - loop_time
        fps = 30.0 / elapsed if elapsed > 0 else 0
        print(f"[Main] 30帧平均帧率: {fps:.1f} fps")
        loop_time = time.time()

# ====================================================================
# 统计 & 清理
# ====================================================================
total_time = time.time() - init_time
avg_fps = frames / total_time if total_time > 0 else 0
print(f"\n[Main] 总计 {frames} 帧, 耗时 {total_time:.1f}s, 平均 {avg_fps:.1f} fps")

print("[Main] 释放资源...")
camera.release()
display.release()
servo.release()
pool.release()
print("[Main] 退出完成")
