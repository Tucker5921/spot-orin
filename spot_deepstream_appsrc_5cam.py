#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import math
import time
import signal
import argparse
import threading
from typing import Dict, List, Optional

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import numpy as np
import cv2
from scipy import ndimage

import pyds

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2


# 與 deepstream-test3 類似的 class id 命名；若你換模型，請自行調整
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

MAX_DISPLAY_LEN = 64
MUXER_BATCH_TIMEOUT_USEC = 33000

SPOT_SOURCES = [
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
    "back_fisheye_image",
]

ROTATION_ANGLE = {
    "back_fisheye_image": 0,
    "frontleft_fisheye_image": -78,
    "frontright_fisheye_image": -102,
    "left_fisheye_image": 0,
    "right_fisheye_image": 180,
}


def cb_newpad(decodebin, decoder_src_pad, data):
    """保留函式名稱習慣，這版 source bin 不用動態 pad，僅作佔位。"""
    return


def decodebin_child_added(child_proxy, obj, name, user_data):
    """保留與 deepstream-test3 類似的函式外觀，這版不使用。"""
    return


def is_aarch64() -> bool:
    return os.uname().machine == "aarch64"


class SpotDeepStreamApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pipeline: Optional[Gst.Pipeline] = None
        self.loop: Optional[GLib.MainLoop] = None
        self.streammux: Optional[Gst.Element] = None
        self.pgie: Optional[Gst.Element] = None
        self.tiler: Optional[Gst.Element] = None
        self.nvvidconv: Optional[Gst.Element] = None
        self.nvosd: Optional[Gst.Element] = None
        self.sink: Optional[Gst.Element] = None

        self.appsrc_map: Dict[str, Gst.Element] = {}
        self.frame_count_map: Dict[str, int] = {name: 0 for name in SPOT_SOURCES}
        self.stop_event = threading.Event()
        self.producer_thread: Optional[threading.Thread] = None

        self.robot = None
        self.image_client = None
        self.requests = []

    def _jpeg_response_to_bytes(self, response) -> Optional[bytes]:
        """直接回傳原始 JPEG，或先旋轉再重新編碼成 JPEG。"""
        jpeg_bytes = bytes(response.shot.image.data)
        if not self.args.rotate:
            return jpeg_bytes

        np_buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[WARN] 解碼失敗: {response.source.name}")
            return None

        angle = ROTATION_ANGLE.get(response.source.name, 0)
        if angle != 0:
            frame = ndimage.rotate(frame, angle)

        ok, enc = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)],
        )
        if not ok:
            print(f"[WARN] 重新編碼失敗: {response.source.name}")
            return None
        return enc.tobytes()

    def _push_jpeg_to_appsrc(self, source_name: str, jpeg_bytes: bytes) -> bool:
        appsrc = self.appsrc_map[source_name]
        frame_num = self.frame_count_map[source_name]

        buf = Gst.Buffer.new_allocate(None, len(jpeg_bytes), None)
        if buf is None:
            print(f"[WARN] 無法建立 Gst.Buffer: {source_name}")
            return False

        buf.fill(0, jpeg_bytes)

        pts = (frame_num * Gst.SECOND) // self.args.fps
        duration = Gst.SECOND // self.args.fps

        buf.pts = pts
        buf.dts = pts
        buf.duration = duration
        buf.offset = frame_num

        ret = appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            print(f"[WARN] push-buffer 失敗: source={source_name}, ret={ret}")
            return False

        self.frame_count_map[source_name] += 1
        return True

    def _spot_capture_loop(self):
        print("[INFO] Spot capture thread started.")
        while not self.stop_event.is_set():
            try:
                responses = self.image_client.get_image(self.requests)
            except Exception as e:
                print(f"[ERROR] Spot get_image 失敗: {e}")
                time.sleep(0.2)
                continue

            for resp in responses:
                if self.stop_event.is_set():
                    break

                source_name = resp.source.name
                if source_name not in self.appsrc_map:
                    continue

                jpeg_bytes = self._jpeg_response_to_bytes(resp)
                if jpeg_bytes is None:
                    continue

                self._push_jpeg_to_appsrc(source_name, jpeg_bytes)

        print("[INFO] Spot capture thread stopping.")
        for source_name, appsrc in self.appsrc_map.items():
            try:
                appsrc.emit("end-of-stream")
            except Exception:
                pass

    def init_spot(self):
        sdk = bosdyn.client.create_standard_sdk("spot_deepstream_appsrc_5cam")
        robot = sdk.create_robot(self.args.robot_ip)
        robot.authenticate(self.args.username, self.args.password)
        robot.sync_with_directory()
        robot.time_sync.wait_for_sync()

        self.robot = robot
        self.image_client = robot.ensure_client(ImageClient.default_service_name)

        self.requests = [
            build_image_request(
                source,
                image_format=image_pb2.Image.FORMAT_JPEG,
                quality_percent=int(self.args.quality_percent),
                resize_ratio=float(self.args.resize_ratio),
            )
            for source in SPOT_SOURCES
        ]

    def create_appsrc_source_bin(self, index: int, source_name: str) -> Gst.Bin:
        """
        參考 deepstream-test3.py 的 create_source_bin() 風格，
        但把 uridecodebin 改成 appsrc + JPEG decode chain。
        """
        bin_name = f"source-bin-{index:02d}"
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            raise RuntimeError(f"Unable to create source bin: {bin_name}")

        appsrc = Gst.ElementFactory.make("appsrc", f"appsrc-{index}")
        jpegparse = Gst.ElementFactory.make("jpegparse", f"jpegparse-{index}")
        decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder-{index}")
        conv = Gst.ElementFactory.make("nvvideoconvert", f"conv-{index}")
        capsfilter = Gst.ElementFactory.make("capsfilter", f"capsfilter-{index}")
        queue = Gst.ElementFactory.make("queue", f"queue-{index}")

        if not all([appsrc, jpegparse, decoder, conv, capsfilter, queue]):
            raise RuntimeError(f"Failed to create source elements for {source_name}")

        # appsrc 吃 Spot JPEG bytes
        appsrc_caps = Gst.Caps.from_string(f"image/jpeg,framerate={self.args.fps}/1")
        appsrc.set_property("caps", appsrc_caps)
        appsrc.set_property("is-live", True)
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property("block", True)

        # MJPEG / JPEG 解碼常用設定
        decoder.set_property("mjpeg", 1)

        # 確保進 streammux 前是 NVMM memory
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
        capsfilter.set_property("caps", caps)

        for elem in [appsrc, jpegparse, decoder, conv, capsfilter, queue]:
            nbin.add(elem)

        if not Gst.Element.link_many(appsrc, jpegparse, decoder, conv, capsfilter, queue):
            raise RuntimeError(f"Failed to link source chain for {source_name}")

        ghost_pad = Gst.GhostPad.new("src", queue.get_static_pad("src"))
        if not ghost_pad:
            raise RuntimeError(f"Failed to create ghost pad for {source_name}")
        nbin.add_pad(ghost_pad)

        self.appsrc_map[source_name] = appsrc
        return nbin

    def pgie_src_pad_buffer_probe(self, pad, info, u_data):
        """參考 deepstream-test3.py 的 probe 方式，直接印每路 bbox。"""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("[WARN] Unable to get GstBuffer")
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            obj_counter = {
                PGIE_CLASS_ID_VEHICLE: 0,
                PGIE_CLASS_ID_BICYCLE: 0,
                PGIE_CLASS_ID_PERSON: 0,
                PGIE_CLASS_ID_ROADSIGN: 0,
            }

            l_obj = frame_meta.obj_meta_list
            num_rects = 0

            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                class_id = int(obj_meta.class_id)
                obj_counter[class_id] = obj_counter.get(class_id, 0) + 1
                num_rects += 1

                print(
                    f"[BBOX] stream={frame_meta.pad_index} "
                    f"frame={frame_meta.frame_num} "
                    f"class={class_id} "
                    f"conf={float(obj_meta.confidence):.3f} "
                    f"left={float(obj_meta.rect_params.left):.1f} "
                    f"top={float(obj_meta.rect_params.top):.1f} "
                    f"width={float(obj_meta.rect_params.width):.1f} "
                    f"height={float(obj_meta.rect_params.height):.1f}"
                )

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            print(
                "Frame Number={} Stream={} Number of Objects={} Vehicle_count={} "
                "Person_count={}".format(
                    frame_meta.frame_num,
                    frame_meta.pad_index,
                    num_rects,
                    obj_counter.get(PGIE_CLASS_ID_VEHICLE, 0),
                    obj_counter.get(PGIE_CLASS_ID_PERSON, 0),
                )
            )

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def build_pipeline(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("spot-deepstream-appsrc-pipeline")
        self.loop = GLib.MainLoop()

        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        self.pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        self.tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

        if self.args.no_display:
            self.sink = Gst.ElementFactory.make("fakesink", "fake-sink")
        else:
            if is_aarch64():
                self.sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            else:
                self.sink = Gst.ElementFactory.make("nveglglessink", "egl-sink")

        elems = [self.streammux, self.pgie, self.tiler, self.nvvidconv, self.nvosd, self.sink]
        if not all(elems):
            raise RuntimeError("One or more pipeline elements could not be created.")

        self.pipeline.add(self.streammux)
        self.pipeline.add(self.pgie)
        self.pipeline.add(self.tiler)
        self.pipeline.add(self.nvvidconv)
        self.pipeline.add(self.nvosd)
        self.pipeline.add(self.sink)

        # 建立 5 路 source bin，風格上接近 deepstream-test3.py
        for i, source_name in enumerate(SPOT_SOURCES):
            source_bin = self.create_appsrc_source_bin(i, source_name)
            self.pipeline.add(source_bin)

            sinkpad = self.streammux.request_pad_simple(f"sink_{i}")
            if not sinkpad:
                raise RuntimeError(f"Unable to create sink pad for streammux sink_{i}")

            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                raise RuntimeError(f"Unable to get src pad of source bin {i}")

            if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link source bin {i} to streammux")

        # streammux
        self.streammux.set_property("width", int(self.args.streammux_width))
        self.streammux.set_property("height", int(self.args.streammux_height))
        self.streammux.set_property("batch-size", len(SPOT_SOURCES))
        self.streammux.set_property("live-source", 1)
        self.streammux.set_property("batched-push-timeout", int(self.args.mux_timeout_usec))

        # pgie
        self.pgie.set_property("config-file-path", self.args.configfile)
        try:
            pgie_batch_size = self.pgie.get_property("batch-size")
            if pgie_batch_size != len(SPOT_SOURCES):
                print(
                    f"WARNING: Overriding infer-config batch-size "
                    f"{pgie_batch_size} with {len(SPOT_SOURCES)}"
                )
                self.pgie.set_property("batch-size", len(SPOT_SOURCES))
        except Exception:
            pass

        # tiler
        tiler_rows = int(math.sqrt(len(SPOT_SOURCES)))
        tiler_columns = int(math.ceil(len(SPOT_SOURCES) / tiler_rows))
        self.tiler.set_property("rows", tiler_rows)
        self.tiler.set_property("columns", tiler_columns)
        self.tiler.set_property("width", int(self.args.tiled_width))
        self.tiler.set_property("height", int(self.args.tiled_height))

        self.sink.set_property("sync", False)
        if self.sink.find_property("qos") is not None:
            self.sink.set_property("qos", False)

        if not Gst.Element.link_many(
            self.streammux, self.pgie, self.tiler, self.nvvidconv, self.nvosd, self.sink
        ):
            raise RuntimeError("Failed to link core pipeline.")

        # 與 deepstream-test3.py 類似：在推論後掛 probe 讀 bbox
        pgie_src_pad = self.pgie.get_static_pad("src")
        if not pgie_src_pad:
            raise RuntimeError("Unable to get src pad of pgie")
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.pgie_src_pad_buffer_probe, 0)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

    def bus_call(self, bus, message, loop):
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            print("End-of-stream")
            loop.quit()
        elif msg_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"[WARNING] {err}: {debug}")
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[ERROR] {err}: {debug}")
            loop.quit()
        return True

    def start(self):
        self.init_spot()
        self.build_pipeline()

        self.pipeline.set_state(Gst.State.PLAYING)
        print("Pipeline started.")
        print("Now playing Spot sources:")
        for i, name in enumerate(SPOT_SOURCES):
            print(f"  source[{i}] = {name}")
        print(f"fps = {self.args.fps}")
        print(f"rotate = {self.args.rotate}")

        self.producer_thread = threading.Thread(target=self._spot_capture_loop, daemon=True)
        self.producer_thread.start()

        self._install_signal_handlers()
        self.loop.run()

    def stop(self):
        if self.stop_event.is_set():
            return

        print("Stopping application...")
        self.stop_event.set()

        if self.producer_thread is not None:
            self.producer_thread.join(timeout=2.0)

        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)

        if self.loop is not None and self.loop.is_running():
            self.loop.quit()

    def _install_signal_handlers(self):
        def _handler(sig, frame):
            self.stop()

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spot 5-camera JPEG -> DeepStream appsrc Python pipeline"
    )
    parser.add_argument("--robot-ip", default=os.getenv("SPOT_IP", "192.168.80.3"))
    parser.add_argument("--username", default=os.getenv("SPOT_USERNAME", "admin"))
    parser.add_argument("--password", default=os.getenv("SPOT_PASSWORD", ""))

    parser.add_argument(
        "--configfile",
        required=True,
        help="nvinfer config file path, e.g. dstest_appsrc_config.txt",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--quality-percent", type=int, default=75)
    parser.add_argument("--resize-ratio", type=float, default=1.0)

    parser.add_argument("--streammux-width", type=int, default=1920)
    parser.add_argument("--streammux-height", type=int, default=1080)
    parser.add_argument("--tiled-width", type=int, default=1920)
    parser.add_argument("--tiled-height", type=int, default=1080)
    parser.add_argument("--mux-timeout-usec", type=int, default=MUXER_BATCH_TIMEOUT_USEC)

    parser.add_argument(
        "--rotate",
        action="store_true",
        help="先解 JPEG、旋轉、再重新編碼為 JPEG 後送入 DeepStream；較耗 CPU",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--no-display", action="store_true")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if not args.password:
        print("請用 --password 或環境變數 SPOT_PASSWORD 提供密碼。")
        sys.exit(1)

    app = SpotDeepStreamApp(args)
    try:
        app.start()
    except Exception as e:
        print(f"[FATAL] {e}")
        app.stop()
        sys.exit(1)
    finally:
        app.stop()


if __name__ == "__main__":
    sys.exit(main())