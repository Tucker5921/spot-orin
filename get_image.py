#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np
from scipy import ndimage

import bosdyn.client
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2

# Spot 連線
robot_ip = "192.168.80.3"
username = "admin"
password = "eqyqp33u8i74"

# Spot camera source
sources = [
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
    "back_fisheye_image",
]

# 相機角度校正
ROTATION_ANGLE = {
    "back_fisheye_image": 0,
    "frontleft_fisheye_image": -78,
    "frontright_fisheye_image": -102,
    "left_fisheye_image": 0,
    "right_fisheye_image": 180,
}


def jpeg_response_to_bgr(response, auto_rotate=True):
    jpeg_bytes = response.shot.image.data
    np_buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

    if frame is None:
        return None

    if auto_rotate:
        angle = ROTATION_ANGLE.get(response.source.name, 0)
        frame = ndimage.rotate(frame, angle)

    return frame


def main():
    sdk = bosdyn.client.create_standard_sdk("image_capture")
    robot = sdk.create_robot(robot_ip)
    robot.authenticate(username, password)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)

    requests = [
        build_image_request(
            source,
            image_format=image_pb2.Image.FORMAT_JPEG,
            quality_percent=75,
            resize_ratio=1.0,
        )
        for source in sources
    ]

    for source in sources:
        cv2.namedWindow(source, cv2.WINDOW_NORMAL)

    print("開始監看 Spot 五路影像（含轉正），按 q 或 Esc 結束。")

    try:
        while True:
            responses = image_client.get_image(requests)

            for resp in responses:
                frame = jpeg_response_to_bgr(resp, auto_rotate=True)
                if frame is None:
                    print(f"[WARN] 解碼失敗: {resp.source.name}")
                    continue

                cv2.imshow(resp.source.name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()