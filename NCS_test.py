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

            if not args.no_debug:
                cv2.imwrite('network_compute_server_output.jpg', image)

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


# def register_with_robot(options):
#     ip = bosdyn.client.common.get_self_ip(options.hostname)
#     print('Detected IP address as: ' + ip)

#     sdk = bosdyn.client.create_standard_sdk("yolo_server")
#     robot = sdk.create_robot(options.hostname)
#     robot.authenticate("admin", "eqyqp33u8i74")

#     directory_registration_client = robot.ensure_client(
#         bosdyn.client.directory_registration.DirectoryRegistrationClient.default_service_name)

#     print(f'Registering {ip}:{options.port}...')
#     directory_registration_client.register(options.name, "bosdyn.api.NetworkComputeBridgeWorker",
#                                            kServiceAuthority, ip, int(options.port))
    
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

# def main(argv):
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('-m', '--model', action='append', nargs=2, required=True)
#     parser.add_argument('-m', '--model', action='append', nargs=2, 
#                         default=[['best.engine', 'labels.txt']])
#     parser.add_argument('-p', '--port', default='50051')
#     parser.add_argument('-d', '--no-debug', action='store_true')
#     parser.add_argument('-n', '--name', default='fetch-server')
#     bosdyn.client.util.add_base_arguments(parser)
#     options = parser.parse_args(argv)

#     for model in options.model:
#         if not os.path.exists(model[0]):
#              print(f'Error: model path ({model[0]}) not found.')
#              sys.exit(1)

#     register_with_robot(options)

#     # 只需要一個 Request Queue，Response Queue 會動態建立
#     request_queue = queue.Queue()

#     thread = threading.Thread(target=process_thread, args=([options, request_queue]))
#     thread.start()

#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     network_compute_bridge_service_pb2_grpc.add_NetworkComputeBridgeWorkerServicer_to_server(
#         NetworkComputeBridgeWorkerServicer(request_queue), server)
#     server.add_insecure_port('[::]:' + options.port)
#     server.start()

#     print('Running... (Press Ctrl+C to stop)')
#     try:
#         thread.join()
#     except KeyboardInterrupt:
#         pass

#     return True


# def main(argv):
#     # 1. 預設參數設定
#     FIXED_ROBOT_IP = "10.0.0.3"
#     DEFAULT_MODEL = "best.engine"
#     DEFAULT_LABELS = "labels.txt"
#     SERVICE_NAME = "fetch-server"
#     PORT = "50051"

#     parser = argparse.ArgumentParser()
#     # 雖然硬編碼了，但保留 parser 讓你以後想改還能改，不改就用預設值
#     parser.add_argument('-m', '--model', action='append', nargs=2, 
#                         default=[[DEFAULT_MODEL, DEFAULT_LABELS]])
#     parser.add_argument('-p', '--port', default=PORT)
#     parser.add_argument('-n', '--name', default=SERVICE_NAME)
#     parser.add_argument('-d', '--no-debug', action='store_true')
#     parser.add_argument('hostname', nargs='?', default=FIXED_ROBOT_IP)
    
#     options = parser.parse_args(argv)

#     # 檢查模型文件
#     if not os.path.exists(options.model[0][0]):
#         print(f"錯誤: 找不到模型文件 {options.model[0][0]}")
#         sys.exit(1)

#     # 2. 註冊服務
#     try:
#         register_with_robot(options)
#     except Exception as e:
#         print(f"註冊失敗: {e}")
#         sys.exit(1)

#     # 3. 啟動 YOLO 處理線程
#     request_queue = queue.Queue()
#     thread = threading.Thread(target=process_thread, args=([options, request_queue]))
#     thread.start()

#     # 4. 啟動 gRPC Server
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     network_compute_bridge_service_pb2_grpc.add_NetworkComputeBridgeWorkerServicer_to_server(
#         NetworkComputeBridgeWorkerServicer(request_queue), server)
#     server.add_insecure_port('0.0.0.0:' + options.port)
#     server.start()

#     print(f'YOLO 伺服器已在 Port {options.port} 啟動')
#     print('按下 Ctrl+C 可停止伺服器並自動註銷...')

#     # 5. [關鍵] 捕捉 Ctrl+C 進行自動註銷
#     try:
#         while True:
#             time.sleep(1) # 保持主執行緒存活
#     except KeyboardInterrupt:
#         print("\n\n偵測到停止訊號，正在自動從 Spot 註銷服務...")
#         try:
#             # 建立臨時客戶端來註銷服務
#             sdk = bosdyn.client.create_standard_sdk("cleanup_client")
#             robot = sdk.create_robot(options.hostname)
#             robot.authenticate("admin", "eqyqp33u8i74")
#             reg_client = robot.ensure_client(
#                 bosdyn.client.directory_registration.DirectoryRegistrationClient.default_service_name)
            
#             if reg_client.unregister(options.name):
#                 print(f"成功註銷服務: {options.name}")
#             else:
#                 print(f"服務 {options.name} 可能早已被移除。")
#         except Exception as cleanup_e:
#             print(f"註銷過程中出錯: {cleanup_e}")
        
#         print("程式安全退出。")
#         sys.exit(0)

#     return True

def main(argv):
    # FIXED_ROBOT_IP = "10.0.0.3"
    FIXED_ROBOT_IP = "192.168.80.3"
    DEFAULT_MODEL = "best.engine"
    DEFAULT_LABELS = "labels.txt"
    SERVICE_NAME = "fetch-server"
    PORT = "50051"

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

    # 啟動處理線程
    thread = threading.Thread(target=process_thread, args=([options, request_queue]))
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
        thread.join()
    except KeyboardInterrupt:
        print("\n偵測到停止訊號，正在註銷服務...")
        try:
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
        
        # 強制退出，避免殘留進程
        os._exit(0) 

    return True

if __name__ == '__main__':
    logging.basicConfig()
    if not main(sys.argv[1:]):
        sys.exit(1)