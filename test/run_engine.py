#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

class YoloEngineNode(Node):
    def __init__(self):
        super().__init__('yolo_engine_node')
        
        # --- 設定區 ---
        # 這是您剛剛用 trtexec 轉出來的神器
        model_path = 'best.engine' 
        #camera_topic = '/camera/color/image_raw' # 請確認您的相機 Topic
        camera_topic = '/camera/frontleft/image'
        # -------------

        self.get_logger().info(f'正在載入 TensorRT 引擎: {model_path} ...')
        
        # 關鍵魔法：Ultralytics 會自動辨識 .engine 檔並使用 TensorRT 後端
        # task='segment' 很重要，告訴它我們要的是 Mask  
        self.model = YOLO(model_path, task='segment')
        
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, camera_topic, self.callback, 10)
        
        # 發布兩種結果：
        # 1. 畫好框跟顏色的漂亮圖片 (給人看)
        self.vis_pub = self.create_publisher(Image, '/yolo/visualization', 10)
        # 2. 純黑白的 Mask (給機器人看，用來算座標)
        self.mask_pub = self.create_publisher(Image, '/yolo/mask', 10)
        
        self.get_logger().info('模型載入完成，開始推論！')

    def callback(self, msg):
        # 1. ROS Image -> OpenCV Image
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 2. 推論 (Inference)
        # verbose=False 讓它不要一直要在終端機印字，會變快
        results = self.model(cv_img, verbose=False)
        
        for r in results:
            # --- A. 處理視覺化結果 ---
            # r.plot() 會幫您把 Mask 和 Box 畫在圖上
            vis_img = r.plot()
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding='bgr8')
            self.vis_pub.publish(vis_msg)
            
            # --- B. 處理 Mask (給手臂用的資料) ---
            if r.masks is not None:
                # 這裡取出所有偵測到的物件 Mask
                # data 是 (N, H, W) 的 Tensor
                masks = r.masks.data.cpu().numpy()
                
                # 簡單範例：把所有 Mask 疊在一起變成一張圖
                # 如果有垃圾，這張圖對應垃圾的位置就是 255 (白)，背景是 0 (黑)
                combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
                
                # 因為 YOLO 的 Mask 有時會縮放，保險起見 Resize 回原圖大小
                combined_mask = cv2.resize(combined_mask, (cv_img.shape[1], cv_img.shape[0]))
                
                mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, encoding='mono8')
                mask_msg.header = msg.header # 時間戳記要對齊，這對之後跟 Depth 同步很重要
                self.mask_pub.publish(mask_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloEngineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
