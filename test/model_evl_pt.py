import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# --- 設定區 ---
MODEL_PATH = 'best.pt'      # 你的 PT 模型檔名
IMAGE_PATH = 'network_compute_server_output.jpg'     # 你要測試的圖片檔名
CONF_THRESHOLD = 0.25       # 信心值門檻

def test_pt_model():
    # 檢查檔案
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型檔案: {MODEL_PATH}")
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 找不到測試圖片: {IMAGE_PATH}")
        return

    # 1. 載入模型 (PT 模型會自動載入到 GPU，如果有的話)
    print(f"🔄 正在載入 PyTorch 模型: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # 2. 讀取與處理影像
    img_bgr = cv2.imread(IMAGE_PATH)
    # 強烈建議轉 RGB，因為 YOLO 訓練時通常看的是 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. 執行推論並計時
    print(f"🚀 開始推論...")
    start_time = time.time()
    results = model(img_rgb, conf=CONF_THRESHOLD, verbose=False)
    end_time = time.time()
    
    result = results[0]
    
    # 4. 輸出偵測資訊
    inference_ms = (end_time - start_time) * 1000
    print(f"⏱️ 推理耗時: {inference_ms:.2f} ms")
    
    # 取得偵測結果
    boxes = result.boxes.xyxy.cpu().numpy()   # 像素座標
    scores = result.boxes.conf.cpu().numpy()  # 信心值
    classes = result.boxes.cls.cpu().numpy()  # 類別索引
    
    if len(boxes) == 0:
        print("⚠️ 沒偵測到任何物體！請嘗試降低 CONF_THRESHOLD。")
    else:
        print(f"✅ 偵測到 {len(boxes)} 個物體：")

    # 5. 繪製並列印細節
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        score = scores[i]
        cls_id = int(classes[i])
        label = model.names[cls_id]
        
        print(f"   - [{i}] {label}: {score:.4f} @ [{x1}, {y1}, {x2}, {y2}]")
        
        # 畫框與文字 (BGR 空間畫圖)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2) # PT 用藍色框區分
        cv2.putText(img_bgr, f"{label} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 6. 存檔
    output_path = 'pt_test_result.jpg'
    cv2.imwrite(output_path, img_bgr)
    print(f"\n💾 結果已儲存至: {output_path}")

if __name__ == '__main__':
    test_pt_model()