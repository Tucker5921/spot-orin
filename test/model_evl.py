# import cv2
# import time
# import numpy as np
# from ultralytics import YOLO
# import argparse
# import os

# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pt', type=str, required=True, help='Path to .pt model')
#     parser.add_argument('--engine', type=str, required=True, help='Path to .engine model')
#     parser.add_argument('--source', type=str, default='0', help='Path to image, video, or 0 for webcam')
#     parser.add_argument('--imgsz', type=int, default=640, help='Inference size')
#     return parser.parse_args()

# def benchmark_model(model, name, img_size=(640, 640), runs=50):
#     print(f"\n--- 正在測試 {name} 基準速度 (純運算，跑 {runs} 次) ---")
#     dummy_input = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    
#     # 暖機
#     print(f"正在暖機 {name}...")
#     for _ in range(10):
#         model(dummy_input, verbose=False)
        
#     # 測試
#     print(f"開始計時...")
#     t_start = time.time()
#     for _ in range(runs):
#         model(dummy_input, verbose=False)
#     t_end = time.time()
    
#     total_time = t_end - t_start
#     avg_time = (total_time / runs) * 1000
#     fps = 1 / (total_time / runs)
    
#     print(f"結果 [{name}]: 平均延遲 {avg_time:.2f} ms | FPS: {fps:.2f}")
#     return fps

# def main():
#     args = parse_opt()

#     # 1. 載入模型
#     print(f"載入 PyTorch 模型: {args.pt}")
#     model_pt = YOLO(args.pt)
    
#     print(f"載入 TensorRT 模型: {args.engine}")
#     model_trt = YOLO(args.engine, task='detect')

#     # 2. 執行基準測試 (可選，如果你只想看圖片結果可以註解掉這段)
#     fps_pt = benchmark_model(model_pt, "PyTorch (.pt)", img_size=(args.imgsz, args.imgsz))
#     fps_trt = benchmark_model(model_trt, "TensorRT (.engine)", img_size=(args.imgsz, args.imgsz))
#     print(f"\n>>> 加速倍率: {fps_trt / fps_pt:.2f} 倍 <<<")

#     # 3. 處理輸入來源
#     source = args.source
#     is_image = source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))

#     print(f"\n--- 正在對來源 '{source}' 進行視覺化測試 ---")

#     if is_image:
#         # --- 單張圖片模式 ---
#         if not os.path.exists(source):
#             print(f"錯誤：找不到圖片檔案 {source}")
#             return

#         frame = cv2.imread(source)
#         if frame is None:
#             print("錯誤：無法讀取圖片")
#             return

#         # 執行推論 (使用 TensorRT Engine)
#         # 第一次跑可能會慢一點點 (Warmup)
#         results = model_trt(frame, imgsz=args.imgsz)

#         # 畫圖
#         annotated_frame = results[0].plot()

#         # 顯示結果
#         cv2.imshow("TensorRT Result (Press any key to exit)", annotated_frame)
#         print("圖片已顯示，按任意鍵離開...")
#         cv2.waitKey(0) # 0 代表無限等待，直到按鍵
#         cv2.destroyAllWindows()

#     else:
#         # --- 影片/鏡頭模式 ---
#         cap_source = int(source) if source.isnumeric() else source
#         cap = cv2.VideoCapture(cap_source)

#         if not cap.isOpened():
#             print("錯誤: 無法開啟攝影機或影片")
#             return

#         while True:
#             ret, frame = cap.read()
#             if not ret: break

#             start = time.time()
#             results = model_trt(frame, imgsz=args.imgsz, verbose=False)
#             end = time.time()
            
#             annotated_frame = results[0].plot()
#             fps_real = 1 / (end - start)

#             cv2.putText(annotated_frame, f"FPS: {fps_real:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.imshow("TensorRT Video Test", annotated_frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
        
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
import cv2
import numpy as np
from ultralytics import YOLO
import os

# --- 設定區 ---
MODEL_PATH = 'best.engine'  # 你的模型檔名
IMAGE_PATH = 'network_compute_server_output.jpg'     # 你要測試的圖片檔名
CONF_THRESHOLD = 0.25       # 測試用的信心值門檻

def test_model():
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤: 找不到模型檔案 {MODEL_PATH}")
        return

    if not os.path.exists(IMAGE_PATH):
        print(f"錯誤: 找不到測試圖片 {IMAGE_PATH}")
        return

    # 1. 載入模型
    print(f"正在載入模型: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH, task='detect')

    # 2. 讀取圖片 (OpenCV 預設讀取是 BGR)
    img_bgr = cv2.imread(IMAGE_PATH)
    # YOLO 訓練通常使用 RGB，建議轉換後送入辨識
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. 執行推論
    print(f"開始推論 (門檻設定: {CONF_THRESHOLD})...")
    results = model(img_rgb, conf=CONF_THRESHOLD, verbose=True)
    
    result = results[0]
    
    # 4. 解析結果
    boxes = result.boxes.xyxy.cpu().numpy()   # 像素座標 [xmin, ymin, xmax, ymax]
    scores = result.boxes.conf.cpu().numpy()  # 信心值
    classes = result.boxes.cls.cpu().numpy()  # 類別 ID
    
    print(f"\n--- 偵測報告 ---")
    print(f"共偵測到 {len(boxes)} 個物體")

    # 5. 在原圖上畫框 (使用 BGR 圖畫，方便 cv2 顯示)
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i].astype(int)
        score = scores[i]
        label = model.names[int(classes[i])]
        
        print(f"[{i}] 標籤: {label}, 信心值: {score:.4f}")
        
        # 畫矩形框 (綠色)
        cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        
        # 標註文字
        text = f"{label}: {score:.2f}"
        cv2.putText(img_bgr, text, (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 6. 顯示與存檔
    output_name = 'test_result.jpg'
    cv2.imwrite(output_name, img_bgr)
    print(f"\n結果已存至: {output_name}")
    
    # 如果你有顯示器環境，可以打開視窗
    # cv2.imshow('Detection Result', img_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    test_model()