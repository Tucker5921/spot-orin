# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.

import argparse
import io
import logging
import os
import queue
import sys
import threading
import time
from concurrent import futures

# --- [關鍵設定] 加速啟動 ---
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['ULTRALYTICS_OFFLINE'] = 'True' 

import cv2
import grpc
import numpy as np
from google.protobuf import wrappers_pb2
from PIL import Image

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import (header_pb2, image_pb2, network_compute_bridge_pb2,
                        network_compute_bridge_service_pb2_grpc)
from ultralytics import YOLO

kServiceAuthority = "fetch-tutorial-worker.spot.robot"

from bosdyn.client.directory_registration import DirectoryRegistrationClient

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

class YoloTRTModel:
    def __init__(self, model_path, label_path=None):
        print(f"Loading YOLO Model: {model_path}")
        self.model = YOLO(model_path, task='detect') 
        self.name = os.path.basename(model_path)
        self.category_index = {
            k: {'name': v} for k, v in self.model.names.items()
        }

    def predict(self, image):
        results = self.model(image, conf=0.5, iou=0.45, verbose=False)
        result = results[0] 
        boxes = result.boxes.xyxyn.cpu().numpy()  
        scores = result.boxes.conf.cpu().numpy()  
        classes = result.boxes.cls.cpu().numpy()  
        
        detections = {
            'detection_boxes': boxes,      
            'detection_scores': scores,    
            'detection_classes': classes,  
            'num_detections': len(boxes)
        }
        return detections

class SpotYoloBridgeNode(Node):
    def __init__(self):
        super().__init__('spot_yolo_bridge')
        # 建立 Publisher
        self.image_pub = self.create_publisher(Image, 'yolo/debug_image', 10)
        self.det_pub = self.create_publisher(Detection2DArray, 'yolo/detections', 10)
        self.bridge = CvBridge()

    def publish_results(self, cv_image, detections, labels):
        # 1. 發布影像
        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        self.image_pub.publish(img_msg)

        # 2. 發布結構化偵測數據 (供其他 ROS 節點使用)
        det_array = Detection2DArray()
        # 這裡可以根據 detections 內容填充 Detection2DArray...
        self.det_pub.publish(det_array)

# 全域節點變數，方便 thread 調用
ros_node = None

def process_thread(args, request_queue):
    # Load models
    models = {}
    for model in args.model:
        this_model = YoloTRTModel(model[0], model[1])
        models[this_model.name] = this_model

    print(f'\nService {args.name} running on port: {args.port}')
    print('Loaded models:')
    for model_name in models:
        print('    ' + model_name)

    # --- [Warmup] ---
    print('正在暖機 TensorRT Engine...')
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    for m in models.values():
        m.predict(dummy_img)
    print('暖機完成，準備接收請求。')
    
    while True:
        # [關鍵修正 1] 這裡改成接收一個 Tuple: (請求, 專屬的回傳通道)
        request, return_queue = request_queue.get()

        try:
            if isinstance(request, network_compute_bridge_pb2.ListAvailableModelsRequest):
                out_proto = network_compute_bridge_pb2.ListAvailableModelsResponse()
                for model_name in models:
                    out_proto.models.data.append(
                        network_compute_bridge_pb2.ModelData(model_name=model_name))
                return_queue.put(out_proto)
                continue
            
            # 以下是 NetworkComputeRequest 的處理
            out_proto = network_compute_bridge_pb2.NetworkComputeResponse()
            out_proto.status = network_compute_bridge_pb2.NETWORK_COMPUTE_STATUS_SUCCESS

            # Find model logic
            target_model_name = request.input_data.model_name
            if target_model_name not in models:
                # Fallback logic
                found = False
                for m_name in models:
                    if target_model_name in m_name:
                        target_model_name = m_name
                        found = True
                        break
                if not found:
                    target_model_name = list(models.keys())[0]

            model = models[target_model_name]

            # Decode Image
            if request.input_data.image.format == image_pb2.Image.FORMAT_RAW:
                pil_image = Image.open(io.BytesIO(request.input_data.image.data))
                if request.input_data.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_GRAY2RGB)
                else:
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            elif request.input_data.image.format == image_pb2.Image.FORMAT_JPEG:
                dtype = np.uint8
                jpg = np.frombuffer(request.input_data.image.data, dtype=dtype)
                image = cv2.imdecode(jpg, cv2.IMREAD_COLOR)

            # [關鍵修正 2] 寬高修正：OpenCV Shape 是 (H, W)
            image_height = image.shape[0]
            image_width = image.shape[1]

            # Predict
            detections = model.predict(image)

            num_objects = 0
            boxes = detections['detection_boxes']
            scores = detections['detection_scores']
            classes = detections['detection_classes']

            for i in range(detections['num_detections']):
                if scores[i] < request.input_data.min_confidence:
                    continue

                box_raw = boxes[i]
                # 轉成 Pixel 座標 [x1, y1, x2, y2]
                box = [
                    box_raw[0] * image_width, 
                    box_raw[1] * image_height, 
                    box_raw[2] * image_width,
                    box_raw[3] * image_height
                ]

                score = scores[i]
                class_id = int(classes[i])
                label = model.category_index.get(class_id, {'name': 'N/A'})['name']

                num_objects += 1
                print(f'Found: "{label}" ({score:.2f})')

                # Pack protobuf
                point1 = np.array([box[0], box[1]])
                point2 = np.array([box[2], box[1]])
                point3 = np.array([box[2], box[3]])
                point4 = np.array([box[0], box[3]])

                out_obj = out_proto.object_in_image.add()
                out_obj.name = "obj" + str(num_objects) + "_label_" + label

                for p in [point1, point2, point3, point4]:
                    v = out_obj.image_properties.coordinates.vertexes.add()
                    v.x, v.y = p[0], p[1]

                out_obj.additional_properties.Pack(wrappers_pb2.FloatValue(value=score))

                # Debug Image Drawing
                if not args.no_debug:
                    pts = np.array([point1, point2, point3, point4], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], True, (0, 255, 0), 2)
                    cv2.putText(image, f"{label}: {score:.2f}", (int(box[0]), int(box[1]-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # if not args.no_debug:
            #     cv2.imwrite('network_compute_server_output.jpg', image)
            if not args.no_debug and ros_node is not None:
                # 透過 ROS2 發布
                # ros_node.publish_results(image, detections, model.category_index)
                if num_objects > 0:
                    # 只有在辨識到物品時才透過 ROS2 發布影像與數據
                    ros_node.publish_results(image, detections, model.category_index)
                    # print(f"偵測到 {num_objects} 個目標，已發布至 ROS2。") # 可選：Debug 用
                else:
                    # 如果沒偵測到東西，可以選擇靜默，或是發布空數據
                    # 通常為了省頻寬與 RViz 乾淨，我們在這裡什麼都不做
                    pass

            # Put Success Result
            return_queue.put(out_proto)

        except Exception as e:
            print(f"!!! Error in process_thread: {e}")
            # 即使出錯，也要回傳一個錯誤狀態，不然 Client 會卡死
            if not isinstance(request, network_compute_bridge_pb2.ListAvailableModelsRequest):
                err_proto = network_compute_bridge_pb2.NetworkComputeResponse()
                err_proto.status = network_compute_bridge_pb2.NETWORK_COMPUTE_STATUS_EXTERNAL_SERVER_ERROR
                return_queue.put(err_proto)
            else:
                # ListAvailableModels 出錯就回傳空的
                return_queue.put(network_compute_bridge_pb2.ListAvailableModelsResponse())


class NetworkComputeBridgeWorkerServicer(
        network_compute_bridge_service_pb2_grpc.NetworkComputeBridgeWorkerServicer):

    def __init__(self, thread_input_queue):
        super(NetworkComputeBridgeWorkerServicer, self).__init__()
        self.thread_input_queue = thread_input_queue

    def NetworkCompute(self, request, context):
        # [關鍵修正 3] 每個請求創建自己的 Queue
        my_response_queue = queue.Queue()
        # 把 (請求, 我的Queue) 丟給 worker
        self.thread_input_queue.put((request, my_response_queue))
        # 只從我的 Queue 等待結果
        out_proto = my_response_queue.get()
        return out_proto

    def ListAvailableModels(self, request, context):
        my_response_queue = queue.Queue()
        self.thread_input_queue.put((request, my_response_queue))
        out_proto = my_response_queue.get()
        return out_proto

def register_with_robot(options):
    ip = bosdyn.client.common.get_self_ip(options.hostname)
    print(f'偵測到本機 IP: {ip}')

    sdk = bosdyn.client.create_standard_sdk("yolo_server")
    robot = sdk.create_robot(options.hostname)
    # 請確保密碼正確
    robot.authenticate("admin", "eqyqp33u8i74")

    registration_client = robot.ensure_client(
        bosdyn.client.directory_registration.DirectoryRegistrationClient.default_service_name)

    # --- [新增邏輯]：先檢查並踢掉舊服務 ---
    try:
        print(f'正在檢查並清理舊的 "{options.name}" 服務...')
        registration_client.unregister(options.name)
        time.sleep(0.5) # 給 Directory 服務一點反應時間
    except Exception:
        # 如果本來就沒這個服務，會報錯，我們直接跳過即可
        pass

    # --- 執行正式註冊 ---
    print(f'正在將 {ip}:{options.port} 註冊為 {options.name}...')
    try:
        registration_client.register(options.name, "bosdyn.api.NetworkComputeBridgeWorker",
                                    kServiceAuthority, ip, int(options.port))
        print(f"服務 {options.name} 註冊成功！")
    except Exception as e:
        print(f"註冊失敗: {e}")
        # 如果還是失敗，可能是 IP 衝突或網路問題，建議直接中斷
        sys.exit(1)

def main(argv):
    FIXED_ROBOT_IP = "10.0.0.3"
    # FIXED_ROBOT_IP = "192.168.80.3"
    DEFAULT_MODEL = "best.engine"
    DEFAULT_LABELS = "labels.txt"
    SERVICE_NAME = "fetch-server"
    PORT = "50051"

    rclpy.init()
    global ros_node
    ros_node = SpotYoloBridgeNode()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', action='append', nargs=2, 
                        default=[[DEFAULT_MODEL, DEFAULT_LABELS]])
    parser.add_argument('-p', '--port', default=PORT)
    parser.add_argument('-n', '--name', default=SERVICE_NAME)
    parser.add_argument('-d', '--no-debug', action='store_true')
    parser.add_argument('hostname', nargs='?', default=FIXED_ROBOT_IP)
    
    options = parser.parse_args(argv)
    
    if not os.path.exists(options.model[0][0]):
        print(f"錯誤: 找不到模型文件 {options.model[0][0]}")
        sys.exit(1)
    
    # 註冊服務
    register_with_robot(options)

    request_queue = queue.Queue()
    thread = threading.Thread(target=process_thread, args=([options, request_queue]), daemon=True)
    thread.start()

    # 啟動 gRPC Server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    network_compute_bridge_service_pb2_grpc.add_NetworkComputeBridgeWorkerServicer_to_server(
        NetworkComputeBridgeWorkerServicer(request_queue), server)
    
    # 這裡用 0.0.0.0 確保外部連得進來
    server.add_insecure_port('0.0.0.0:' + options.port)
    server.start()

    print(f'YOLO 伺服器運行中 (Port {options.port})...')
    
    try:
        # 恢復原本的 thread.join()，這對 gRPC 服務器比較友善
        # thread.join()
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        print("\n偵測到停止訊號")
    finally:
        try:
            print("\n正在註銷服務...")
            # 建立臨時清理客戶端
            sdk = bosdyn.client.create_standard_sdk("cleanup")
            robot = sdk.create_robot(options.hostname)
            robot.authenticate("admin", "eqyqp33u8i74")
            reg_client = robot.ensure_client(
                bosdyn.client.directory_registration.DirectoryRegistrationClient.default_service_name)
            reg_client.unregister(options.name)
            print("成功註銷服務。")
        except Exception as e:
            print(f"註銷失敗: {e}")
            
        # 關閉 ROS2
        ros_node.destroy_node()
        rclpy.shutdown()
        server.stop(0)
        
        # 強制退出，避免殘留進程
        os._exit(0) 

    return True
    
if __name__ == '__main__':
    logging.basicConfig()
    if not main(sys.argv[1:]):
        sys.exit(1)