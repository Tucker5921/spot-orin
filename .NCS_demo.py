# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).
import argparse
import io
import logging
import os
import queue
import sys
import threading
import time
from concurrent import futures
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['ULTRALYTICS_OFFLINE'] = 'True' # 強制離線模式
import cv2
import grpc
import numpy as np
# import tensorflow as tf  <-- 不需要了
from google.protobuf import wrappers_pb2
# from object_detection.utils import label_map_util <-- 不需要了
from PIL import Image
import bosdyn.client
import bosdyn.client.util
from bosdyn.api import (header_pb2, image_pb2, network_compute_bridge_pb2,
                        network_compute_bridge_service_pb2_grpc)
from ultralytics import YOLO
kServiceAuthority = "fetch-tutorial-worker.spot.robot"

class YoloTRTModel:
    def __init__(self, model_path, label_path=None):
        # model_path 預期是 "your_model.engine" 或 "your_model.pt"
        print(f"Loading YOLO Model: {model_path}")
        self.model = YOLO(model_path, task='detect') 
        self.name = os.path.basename(model_path)

        # [修正 2] 建立 category_index 以相容原本的邏輯
        # YOLO 的 model.names 是一個 dict {0: 'person', 1: 'dogtoy'...}
        # 我們把它轉成原本程式期待的格式： {0: {'name': 'person'}, ...}
        self.category_index = {
            k: {'name': v} for k, v in self.model.names.items()
        }

    def predict(self, image):
        # 執行推論
        # verbose=False 減少 log 輸出
        results = self.model(image, conf=0.5, iou=0.45, verbose=False)
        
        result = results[0] 
        
        # 提取數據 (轉回 CPU 處理)
        # xyxyn: Normalized [x1, y1, x2, y2]
        boxes = result.boxes.xyxyn.cpu().numpy()  
        scores = result.boxes.conf.cpu().numpy()  
        classes = result.boxes.cls.cpu().numpy()  
        
        # 重新封裝
        detections = {
            'detection_boxes': boxes,      # [N, 4]
            'detection_scores': scores,    # [N]
            'detection_classes': classes,  # [N]
            'num_detections': len(boxes)
        }
        
        return detections

def process_thread(args, request_queue):
    # Load the model(s)
    models = {}
    for model in args.model:
        # model[0] 是路徑, model[1] 原本是 label 檔，現在 YOLO 不需要，但為了參數相容保留
        this_model = YoloTRTModel(model[0], model[1])
        models[this_model.name] = this_model

    print('')
    print('Service ' + args.name + ' running on port: ' + str(args.port))

    print('Loaded models:')
    for model_name in models:
        print('    ' + model_name)
    # --- [新增] Warmup 暖機 ---
    print('正在暖機 TensorRT Engine...')
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    for m in models.values():
        m.predict(dummy_img)
    print('暖機完成，準備接收請求。')
    while True:
        data = request_queue.get()
        if data is None: break
        request, private_queue = data

        if isinstance(request, network_compute_bridge_pb2.ListAvailableModelsRequest):
            out_proto = network_compute_bridge_pb2.ListAvailableModelsResponse()
            for model_name in models:
                out_proto.models.data.append(
                    network_compute_bridge_pb2.ModelData(model_name=model_name))
            private_queue.put(out_proto)
            continue
        else:
            out_proto = network_compute_bridge_pb2.NetworkComputeResponse()
            out_proto.status = network_compute_bridge_pb2.NETWORK_COMPUTE_STATUS_SUCCESS
        try:
            # Find the model
            # 這裡做一個簡單的 fallback，如果請求的模型名字對不上，就用第一個載入的模型 (方便測試)
            # 原本邏輯較嚴格
            target_model_name = request.input_data.model_name
            if target_model_name not in models:
                # 嘗試只對比檔名
                found = False
                for m_name in models:
                    if target_model_name in m_name:
                        target_model_name = m_name
                        found = True
                        break
                
                if not found:
                    # 如果真的找不到，預設使用第一個模型 (為了讓 Spot 不需要精確輸入檔名也能跑)
                    target_model_name = list(models.keys())[0]
                    print(f"Warning: Requested model '{request.input_data.model_name}' not found. Using '{target_model_name}' instead.")

            model = models[target_model_name]

            # Unpack the incoming image.
            if request.input_data.image.format == image_pb2.Image.FORMAT_RAW:
                pil_image = Image.open(io.BytesIO(request.input_data.image.data))
                if request.input_data.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                    image = cv2.cvtColor(pil_image, cv2.COLOR_GRAY2RGB)
                elif request.input_data.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                    image = pil_image
                else:
                    print('Error: image input in unsupported pixel format: ',
                        request.input_data.image.pixel_format)
                    out_proto.status = network_compute_bridge_pb2.NETWORK_COMPUTE_STATUS_EXTERNAL_SERVER_ERROR
                    private_queue.put(out_proto)
                    continue

            elif request.input_data.image.format == image_pb2.Image.FORMAT_JPEG:
                dtype = np.uint8
                jpg = np.frombuffer(request.input_data.image.data, dtype=dtype)
                image = cv2.imdecode(jpg, -1)
                if len(image.shape) < 3:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # image_width = image.shape[0]
            # image_height = image.shape[1]
            image_height, image_width = image.shape[:2] # 這樣才對：0是高，1是寬

            # [修正 1] 這裡開始大改
            # 取得 YOLO 的結果 (已經是乾淨的 Numpy Array，不需要再 pop 或做 tensor 轉換)
            detections = model.predict(image)

            if not args.no_debug:
                # --- 新增偵測調試資訊 ---
                raw_num = len(detections['detection_scores'])
                if raw_num > 0:
                    max_score = np.max(detections['detection_scores'])
                    print(f"DEBUG: YOLO 原始偵測到 {raw_num} 個物體, 最高信心值: {max_score:.4f}")
                else:
                    print("DEBUG: YOLO 什麼都沒看到")
                print(f"DEBUG: 機器狗要求的門檻 (min_confidence): {request.input_data.min_confidence}")
                # -----------------------
            
            num_objects = 0
            
            boxes = detections['detection_boxes']
            scores = detections['detection_scores']
            classes = detections['detection_classes']
            num_detections = detections['num_detections']

            for i in range(num_detections):
                if scores[i] < request.input_data.min_confidence:
                    continue

                # YOLO 輸出 normalized 座標 [x1, y1, x2, y2]
                box_raw = boxes[i]

                # 轉成 Pixel 座標 [x1_px, y1_px, x2_px, y2_px]
                box = [
                    box_raw[0] * image_width, 
                    box_raw[1] * image_height, 
                    box_raw[2] * image_width,
                    box_raw[3] * image_height
                ]

                score = scores[i]
                class_id = int(classes[i])

                if class_id in model.category_index:
                    label = model.category_index[class_id]['name']
                else:
                    label = 'N/A'

                num_objects += 1

                print('Found object with label: "' + label + '" and score: ' + str(score))

                # [修正 3] 座標順序修正
                # YOLO Box 是 [xmin, ymin, xmax, ymax] -> [x1, y1, x2, y2]
                # Spot Protobuf 需要 (x, y)
                # 原本 TF 邏輯是因為 TF 輸出 [y, x, y, x] 所以它才寫 [box[1], box[0]]
                # 現在 YOLO 輸出 [x, y, x, y]，所以直接對應即可
                
                point1 = np.array([box[0], box[1]]) # Top-Left
                point2 = np.array([box[2], box[1]]) # Top-Right
                point3 = np.array([box[2], box[3]]) # Bottom-Right
                point4 = np.array([box[0], box[3]]) # Bottom-Left

                # Add data to the output proto.
                out_obj = out_proto.object_in_image.add()
                out_obj.name = "obj" + str(num_objects) + "_label_" + label

                vertex1 = out_obj.image_properties.coordinates.vertexes.add()
                vertex1.x = point1[0]
                vertex1.y = point1[1]

                vertex2 = out_obj.image_properties.coordinates.vertexes.add()
                vertex2.x = point2[0]
                vertex2.y = point2[1]

                vertex3 = out_obj.image_properties.coordinates.vertexes.add()
                vertex3.x = point3[0]
                vertex3.y = point3[1]

                vertex4 = out_obj.image_properties.coordinates.vertexes.add()
                vertex4.x = point4[0]
                vertex4.y = point4[1]

                # Pack the confidence value.
                confidence = wrappers_pb2.FloatValue(value=score)
                out_obj.additional_properties.Pack(confidence)

                if not args.no_debug:
                    polygon = np.array([point1, point2, point3, point4], np.int32)
                    polygon = polygon.reshape((-1, 1, 2))
                    cv2.polylines(image, [polygon], True, (0, 255, 0), 2)

                    caption = "{}: {:.3f}".format(label, score)
                    # 簡單的防呆，確保文字不會跑出畫面
                    left_x = max(0, min(point1[0], point4[0]))
                    top_y = max(20, min(point1[1], point2[1]))
                    
                    cv2.putText(image, caption, (int(left_x), int(top_y)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

            print('Found ' + str(num_objects) + ' object(s)')

            if not args.no_debug:
                debug_image_filename = 'network_compute_server_output.jpg'
                cv2.imwrite(debug_image_filename, image)
                print('Wrote debug image output to: "' + debug_image_filename + '"')
            out_proto.status = network_compute_bridge_pb2.NETWORK_COMPUTE_STATUS_SUCCESS
        except Exception as e:
            print(f"!!! 推論執行緒發生錯誤: {e}")
            out_proto.status = network_compute_bridge_pb2.NETWORK_COMPUTE_STATUS_EXTERNAL_SERVER_ERROR
        
        finally:
            # [關鍵]：確保不論成功或是噴錯，一定要把 proto 塞回去給主執行緒
            private_queue.put(out_proto)


class NetworkComputeBridgeWorkerServicer(
        network_compute_bridge_service_pb2_grpc.NetworkComputeBridgeWorkerServicer):

    def __init__(self, thread_input_queue):
        super(NetworkComputeBridgeWorkerServicer, self).__init__()

        self.thread_input_queue = thread_input_queue

    def NetworkCompute(self, request, context):
        print('Got NetworkCompute request')
        single_res_queue = queue.Queue()
        self.thread_input_queue.put((request, single_res_queue))
        return single_res_queue.get()


    def ListAvailableModels(self, request, context):
        print('Got ListAvailableModels request')
        single_res_queue = queue.Queue()
        self.thread_input_queue.put((request, single_res_queue))
        return single_res_queue.get()


def register_with_robot(options):
    """ Registers this worker with the robot's Directory."""
    ip = bosdyn.client.common.get_self_ip(options.hostname)
    print('Detected IP address as: ' + ip)

    sdk = bosdyn.client.create_standard_sdk("tensorflow_server")
    robot = sdk.create_robot(options.hostname)
    robot.authenticate("admin", "eqyqp33u8i74")
    # Authenticate robot before being able to use it
    directory_client = robot.ensure_client(
        bosdyn.client.directory.DirectoryClient.default_service_name)
    directory_registration_client = robot.ensure_client(
        bosdyn.client.directory_registration.DirectoryRegistrationClient.default_service_name)

    # Check to see if a service is already registered with our name
    services = directory_client.list()
    for s in services:
        if s.name == options.name:
            print("WARNING: existing service with name, \"" + options.name + "\", removing it.")
            directory_registration_client.unregister(options.name)
            break
    # Register service
    print('Attempting to register ' + ip + ':' + options.port + ' onto ' + options.hostname +
          ' directory...')
    directory_registration_client.register(options.name, "bosdyn.api.NetworkComputeBridgeWorker",
                                           kServiceAuthority, ip, int(options.port))


def main(argv):
    default_port = '50051'

    parser = argparse.ArgumentParser()
    # 這裡保留 nargs=2，為了相容。但你可以只傳一個 dummy 字串作為第二個參數
    # 例如: python3 script.py -m my_model.engine dummy_label
    parser.add_argument(
        '-m', '--model', help=
        '[MODEL_FILE] [IGNORED]: Path to the .engine file and a dummy string',
        action='append', nargs=2, required=True)
    parser.add_argument('-p', '--port', help='Server\'s port number, default: ' + default_port,
                        default=default_port)
    parser.add_argument('-d', '--no-debug', help='Disable writing debug images.',
                        action='store_true')
    parser.add_argument('-n', '--name', help='Service name', default='fetch-server')
    bosdyn.client.util.add_base_arguments(parser)

    options = parser.parse_args(argv)
    print("0")
    print(options.model)

    # [修正 4] 允許檔案 (Engine) 通過檢查，而不只是資料夾
    for model in options.model:
        if not os.path.exists(model[0]):
             print('Error: model path (' + model[0] + ') not found.')
             sys.exit(1)

    # Perform registration.
    register_with_robot(options)

    # Thread-safe queues for communication between the GRPC endpoint and the ML thread.
    request_queue = queue.Queue()

    # Start server thread
    thread = threading.Thread(target=process_thread, args=(options, request_queue))
    thread.start()

    # Set up GRPC endpoint
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    network_compute_bridge_service_pb2_grpc.add_NetworkComputeBridgeWorkerServicer_to_server(
        NetworkComputeBridgeWorkerServicer(request_queue), server)
    server.add_insecure_port('[::]:' + options.port)
    server.start()
    print('Running...')
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Shutting down...")
        server.stop(0)

    return True


if __name__ == '__main__':
    logging.basicConfig()
    if not main(sys.argv[1:]):
        sys.exit(1)