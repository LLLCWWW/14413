"""
飞凌 ELFBoard RV1126B — 7寸 MIPI DSI 电容触摸屏 显示管理模块

后端自动检测优先级：
  1. Framebuffer /dev/fb0（串口终端场景，直接写显存）
  2. OpenCV imshow（Wayland / X11 图形桌面场景）

7寸 MIPI DSI 典型参数: 1024x600 @ 32bpp (XRGB8888)
"""
import os
import sys
import cv2
import numpy as np
import struct
import fcntl
import mmap

_FBIOGET_VSCREENINFO = 0x4600
_FBIOGET_FSCREENINFO = 0x4602

# 飞凌 7寸 MIPI DSI 屏幕默认参数
_ELF_SCREEN_W = 1024
_ELF_SCREEN_H = 600

# ====================================================================
class _FramebufferDevice:
    """Linux framebuffer 设备封装 (ARM 32-bit ioctl)"""

    def __init__(self, fb_dev="/dev/fb0"):
        self.fb_fd = None
        self.fb_mmap = None
        self.width = 0
        self.height = 0
        self.bpp = 0
        self._fb_size = 0

        self.fb_fd = os.open(fb_dev, os.O_RDWR)

        # 获取可变参数 (FBIOGET_VSCREENINFO, ARM32 struct fb_var_screeninfo)
        var_info = bytearray(160)
        fcntl.ioctl(self.fb_fd, _FBIOGET_VSCREENINFO, var_info)
        self.width  = struct.unpack_from("I", var_info, 0)[0]
        self.height = struct.unpack_from("I", var_info, 4)[0]
        self.bpp    = struct.unpack_from("I", var_info, 24)[0]

        # RGB 位域 (仅用于诊断)
        r_off = struct.unpack_from("I", var_info, 32)[0]
        r_len = struct.unpack_from("I", var_info, 36)[0]
        g_off = struct.unpack_from("I", var_info, 44)[0]
        g_len = struct.unpack_from("I", var_info, 48)[0]
        b_off = struct.unpack_from("I", var_info, 56)[0]
        b_len = struct.unpack_from("I", var_info, 60)[0]

        # 获取固定参数 (line_length 等)
        fix_info = bytearray(68)
        fcntl.ioctl(self.fb_fd, _FBIOGET_FSCREENINFO, fix_info)
        self._fb_size = struct.unpack_from("I", fix_info, 0)[0]

        # mmap 整个 framebuffer
        self.fb_mmap = mmap.mmap(
            self.fb_fd, self._fb_size,
            mmap.MAP_SHARED, mmap.PROT_WRITE,
        )

        print(f"[Display] 飞凌 7寸 MIPI DSI → {self.width}x{self.height} "
              f"{self.bpp}bpp "
              f"R({r_off},{r_len}) G({g_off},{g_len}) B({b_off},{b_len})")

    # ----------------------------------------------------------------
    def render(self, frame_bgr):
        """将 BGR 帧渲染到 framebuffer"""
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

        if self.bpp == 32:
            buf = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA).tobytes()
        elif self.bpp == 24:
            buf = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).tobytes()
        elif self.bpp == 16:
            b = frame_bgr[:, :, 0].astype(np.uint16)
            g = frame_bgr[:, :, 1].astype(np.uint16)
            r = frame_bgr[:, :, 2].astype(np.uint16)
            rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
            buf = rgb565.tobytes()
        else:
            sys.stderr.write(f"[Display] 不支持的 bpp: {self.bpp}\n")
            return

        self.fb_mmap.seek(0)
        self.fb_mmap.write(buf)

    def release(self):
        if self.fb_mmap:
            self.fb_mmap.close()
            self.fb_mmap = None
        if self.fb_fd is not None:
            os.close(self.fb_fd)
            self.fb_fd = None

    def __del__(self):
        self.release()


# ====================================================================
class DisplayManager:
    """显示管理器：自动选择最佳后端"""

    def __init__(self, window_name="ELFBoard-Display", fullscreen=True):
        self.window_name = window_name
        self.fullscreen = fullscreen
        self.fb = None
        self._fb_mode = False
        self._window_ready = False

        self._detect()

    # ---- 后端检测 -------------------------------------------------
    def _detect(self):
        # 1) 优先 framebuffer — 串口终端场景唯一可靠方案
        for dev in ("/dev/fb0", "/dev/fb1", "/dev/graphics/fb0"):
            if not os.path.exists(dev):
                continue
            try:
                self.fb = _FramebufferDevice(dev)
                self._fb_mode = True
                return
            except Exception as exc:
                print(f"[Display] {dev} 打开失败: {exc}")

        # 2) Wayland
        if os.environ.get("WAYLAND_DISPLAY"):
            print("[Display] Wayland → OpenCV imshow")
            return

        # 3) X11
        for dpy in (":0", ":0.0", ":1", ":1.0"):
            os.environ["DISPLAY"] = dpy
            try:
                cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
                cv2.destroyWindow("__probe__")
                cv2.waitKey(1)
                print(f"[Display] X11 (DISPLAY={dpy}) → OpenCV imshow")
                return
            except Exception:
                pass

        # 4) 兜底
        os.environ["DISPLAY"] = ":0"
        print("[Display] 兜底模式 → OpenCV imshow (DISPLAY=:0)")

    # ---- 显示接口 -------------------------------------------------
    def show(self, frame_bgr):
        """显示一帧，返回 key-code（fb 模式永远返回 -1）"""
        if self._fb_mode:
            self.fb.render(frame_bgr)
            return -1

        if not self._window_ready:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
            if self.fullscreen:
                cv2.resizeWindow(self.window_name, _ELF_SCREEN_W, _ELF_SCREEN_H)
                cv2.moveWindow(self.window_name, 0, 0)
            self._window_ready = True

        cv2.imshow(self.window_name, frame_bgr)
        return cv2.waitKey(1) & 0xFF

    # ----------------------------------------------------------------
    @property
    def is_framebuffer(self):
        return self._fb_mode

    @property
    def screen_size(self):
        """返回 (width, height)"""
        if self._fb_mode and self.fb:
            return (self.fb.width, self.fb.height)
        return (_ELF_SCREEN_W, _ELF_SCREEN_H)

    def release(self):
        if self._fb_mode and self.fb:
            self.fb.release()
        else:
            cv2.destroyAllWindows()

    # ---- 静态诊断 ------------------------------------------------
    @staticmethod
    def print_info():
        """打印显示屏诊断信息"""
        print("=" * 50)
        print("  飞凌 ELFBoard RV1126B — 显示诊断")
        print("=" * 50)

        # framebuffer
        for fb in ("/dev/fb0", "/dev/fb1"):
            if os.path.exists(fb):
                try:
                    fd = os.open(fb, os.O_RDONLY)
                    var = bytearray(160)
                    fcntl.ioctl(fd, _FBIOGET_VSCREENINFO, var)
                    w = struct.unpack_from("I", var, 0)[0]
                    h = struct.unpack_from("I", var, 4)[0]
                    bpp = struct.unpack_from("I", var, 24)[0]
                    os.close(fd)
                    print(f"  {fb}: {w}x{h} @ {bpp}bpp")
                except Exception:
                    print(f"  {fb}: 存在但无法读取参数")

        # DRM
        for d in os.listdir("/sys/class/drm"):
            if d.startswith("card"):
                try:
                    with open(f"/sys/class/drm/{d}/dev") as f:
                        print(f"  /dev/dri/{d}: dev={f.read().strip()}")
                except Exception:
                    pass

        print("=" * 50)
