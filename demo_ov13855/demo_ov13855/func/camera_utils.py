"""
飞凌 ELFBoard RV1126B — OV13855 MIPI CSI 摄像头管理模块

特性：
- 自动探测 OV13855 设备节点（rkisp_mainpath）
- GStreamer pipeline 优先后端（videoconvert 自动格式转换）
- 回退到 V4L2 直接捕获 + 软件格式转换
- 启动时打印诊断信息
"""
import os
import sys
import cv2
import numpy as np
import subprocess

# rkisp 在 RV1126B 上的候选节点（按优先级排序）
_CANDIDATE_NODES = [
    "/dev/video23",         # rkisp_mainpath 真实节点（优先）
    "/dev/video24",         # rkisp_selfpath
    "/dev/video-camera0",   # rkisp_mainpath（飞凌 BSP 默认，可能指向 video31）
    "/dev/video0",          # 部分 BSP 的 mainpath
    "/dev/video1",          # 部分 BSP 的 selfpath
]


def _run(cmd):
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _probe_sensor_name(video_dev):
    """通过 v4l2-ctl 或 sysfs 读取 driver/card 名称"""
    out = _run(f"v4l2-ctl -d {video_dev} --all 2>/dev/null")
    if out:
        lines = []
        for line in out.split("\n"):
            s = line.strip()
            if "Driver name" in s or "Card type" in s:
                lines.append(s)
        if lines:
            return " | ".join(lines)

    dev_name = os.path.basename(video_dev)
    sysfs_path = f"/sys/class/video4linux/{dev_name}/name"
    if os.path.exists(sysfs_path):
        try:
            with open(sysfs_path, "r") as f:
                return f.read().strip()
        except Exception:
            pass
    return None


def _detect_device():
    """探测可用摄像头设备节点"""
    print("[Camera] 正在探测摄像头设备...")

    # 第一轮：扫描候选节点，匹配 rkisp / ov13855 / rkaiisp
    for node in _CANDIDATE_NODES:
        if not os.path.exists(node):
            continue
        name = _probe_sensor_name(node)
        print(f"  {node}: {name or '(无法读取)'}")
        if name:
            nl = name.lower()
            if any(k in nl for k in ("rkisp", "ov13855", "rkaiisp")):
                print(f"[Camera] 确认设备: {node} ({name})")
                return node

    # 第二轮：大范围扫描 /dev/video*
    print("[Camera] 候选节点未命中，扫描全部 /dev/video*...")
    for i in range(50):
        node = f"/dev/video{i}"
        if not os.path.exists(node):
            continue
        name = _probe_sensor_name(node)
        if not name:
            continue
        print(f"  {node}: {name}")
        nl = name.lower()
        if any(k in nl for k in ("rkisp", "ov13855", "rkaiisp")):
            print(f"[Camera] 确认设备: {node}")
            return node

    # 兜底：返回第一个存在的候选节点
    for node in _CANDIDATE_NODES:
        if os.path.exists(node):
            print(f"[Camera] 回退使用: {node}")
            return node
    return None


def _is_opencv_gstreamer():
    """检测 OpenCV 是否编译了 GStreamer 支持"""
    try:
        info = cv2.getBuildInformation()
        for line in info.split("\n"):
            if "GStreamer" in line:
                return "YES" in line
        return False
    except Exception:
        return False


# ====================================================================
class OV13855Camera:
    """OV13855 摄像头采集器"""

    def __init__(self, width=800, height=600, fps=30, capture_width=1280, capture_height=960):
        self.width = width
        self.height = height
        self.fps = fps
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.cap = None
        self.device = None
        self._use_gst = False
        self._cvt_nv12 = False
        self._need_resize = False

    # ----------------------------------------------------------------
    def open(self):
        self.device = _detect_device()
        if self.device is None:
            print("[Camera] 未找到摄像头设备", file=sys.stderr)
            return False

        # 优先 GStreamer（自动格式协商，适配 multiplanar）
        if _is_opencv_gstreamer():
            if self._open_gstreamer():
                return True
            print("[Camera] GStreamer 失败，尝试 V4L2...")

        # 回退 V4L2
        if self._open_v4l2():
            return True

        print("[Camera] 所有后端均失败", file=sys.stderr)
        return False

    def _open_gstreamer(self):
        """GStreamer: 原生分辨率采集 → videoscale 缩放 → BGR"""
        pipeline = (
            f"v4l2src device={self.device} "
            f"! video/x-raw,framerate={self.fps}/1 "
            f"! videoscale "
            f"! video/x-raw,width={self.width},height={self.height} "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink drop=1 max-buffers=2"
        )
        try:
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                self._use_gst = True
                aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[Camera] GStreamer 就绪 → {self.device} {aw}x{ah}")
                return True
        except Exception as e:
            print(f"[Camera] GStreamer 异常: {e}")
        return False

    def _open_v4l2(self):
        """V4L2 直接打开（回退方案）— 高分辨率采集 + 软件缩放到目标分辨率"""
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            return False

        # 请求更大分辨率：ISP 取更宽画幅缩放到该尺寸，再软件缩放到 target
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"UYVY"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_fmt = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fmt_str = "".join(chr((actual_fmt >> i) & 0xFF) for i in (0, 8, 16, 24))
        self._cvt_nv12 = ("NV12" in fmt_str.upper())

        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._need_resize = (aw != self.width or ah != self.height)
        note = ""
        if self._cvt_nv12:
            note += " (需NV12转换)"
        if self._need_resize:
            note += f" (resize → {self.width}x{self.height})"
        print(f"[Camera] V4L2 就绪 → {self.device} {aw}x{ah} fmt={fmt_str}{note}")
        return True

    # ----------------------------------------------------------------
    def read(self):
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None
        if self._cvt_nv12:
            frame = self._nv12_to_bgr(frame)
        if self._need_resize:
            frame = cv2.resize(frame, (self.width, self.height))
        return True, frame

    def _nv12_to_bgr(self, nv12):
        """NV12(YUV420SP) → BGR（软件转换，GStreamer 不可用时的回退方案）"""
        img_h = nv12.shape[0] * 2 // 3
        y_plane = nv12[:img_h, :]
        uv_plane = nv12[img_h:, :]
        u_ch = cv2.resize(uv_plane[:, ::2], (nv12.shape[1], img_h))
        v_ch = cv2.resize(uv_plane[:, 1::2], (nv12.shape[1], img_h))
        yuv = cv2.merge([y_plane, u_ch, v_ch])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # ----------------------------------------------------------------
    @property
    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.release()
