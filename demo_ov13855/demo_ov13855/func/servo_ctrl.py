"""
ELFBoard RV1126B — 舵机控制模块
GPIO 129 = GPIO4_A1 (40Pin P13)
PWM 周期 20ms, 脉宽 0.5ms(0°) ~ 2.5ms(270°)
"""
import os
import time
import threading
import sys

GPIO_NUM = 129
GPIO_PATH = "/sys/class/gpio/gpio129/value"
GPIO_EXPORT = "/sys/class/gpio/export"
GPIO_UNEXPORT = "/sys/class/gpio/unexport"
GPIO_DIRECTION = "/sys/class/gpio/gpio129/direction"

PERIOD_S = 0.02                # 20ms PWM 周期
PULSE_MIN = 0.0001             # 0.1ms
PULSE_MAX = 0.0030             # 3.0ms


def angle_to_pulse(angle):
    """角度→脉宽(秒)"""
    if angle < 0:
        angle = 0
    if angle > 270:
        angle = 270
    return PULSE_MIN + (PULSE_MAX - PULSE_MIN) * angle / 270.0


def _init_gpio():
    """初始化 GPIO 并导出"""
    if not os.path.exists(GPIO_PATH):
        try:
            with open(GPIO_EXPORT, "w") as f:
                f.write(str(GPIO_NUM))
        except Exception:
            pass
        time.sleep(0.1)

    for _ in range(10):
        if os.path.exists(GPIO_DIRECTION):
            try:
                with open(GPIO_DIRECTION, "w") as f:
                    f.write("out")
                break
            except Exception:
                time.sleep(0.05)
    print("[Servo] GPIO129 就绪 (40Pin P13)")


def _cleanup_gpio():
    """释放 GPIO"""
    try:
        with open(GPIO_UNEXPORT, "w") as f:
            f.write(str(GPIO_NUM))
    except Exception:
        pass
    print("[Servo] GPIO129 已释放")


def _send_pulse(pulse_s):
    """发送一个高电平脉冲（尽量精准）"""
    fd = os.open(GPIO_PATH, os.O_WRONLY)
    os.write(fd, b"1")
    # 忙等待高精度延时
    end = time.perf_counter() + pulse_s
    while time.perf_counter() < end:
        pass
    os.write(fd, b"0")
    os.close(fd)


def _sweep_0_270_blocking():
    """阻塞式扫一次 0→270 度，尽量快"""
    STEPS = 100
    print("[Servo] 开始扫掠 0→270 度 (脉宽 {:.1f}~{:.1f}ms)".format(PULSE_MIN*1000, PULSE_MAX*1000))

    t_start = time.time()
    for i in range(STEPS + 1):
        angle = int(270 * i / STEPS)
        pulse = PULSE_MIN + (PULSE_MAX - PULSE_MIN) * angle / 270.0
        _send_pulse(pulse)
        time.sleep(PERIOD_S)

    # 在 270° 末端多发送几拍，确保舵机完全推到底
    for _ in range(50):
        _send_pulse(PULSE_MAX)
        time.sleep(PERIOD_S)

    t_elapsed = time.time() - t_start
    print(f"[Servo] 扫掠完成, 耗时 {t_elapsed:.1f}s")


# ====================================================================
class ServoController:
    """舵机控制器（非阻塞）"""

    def __init__(self):
        self._running = False
        self._thread = None
        self._last_trigger = 0
        self._cooldown = 1.0       # 两次触发最小间隔（秒）

    def start(self):
        """初始化 GPIO"""
        _init_gpio()

    def trigger(self):
        """非阻塞触发一次 0→270 扫掠（冷却时间内忽略）"""
        now = time.time()
        if now - self._last_trigger < self._cooldown:
            return False

        self._last_trigger = now

        if self._thread and self._thread.is_alive():
            return False        # 上一次还没结束，跳过

        self._thread = threading.Thread(target=_sweep_0_270_blocking, daemon=True)
        self._thread.start()
        return True

    def release(self):
        """释放资源"""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        # 回正到 0°
        try:
            _send_pulse(PULSE_MIN)
        except Exception:
            pass
        _cleanup_gpio()


# 全局单例
_servo = None


def get_servo():
    global _servo
    if _servo is None:
        _servo = ServoController()
    return _servo
