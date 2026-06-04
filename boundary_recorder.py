import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import math

# 引入 TF2 相關模組
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class TfRtkBoundaryRecorder(Node):
    def __init__(self):
        super().__init__('tf_rtk_boundary_recorder')
        
        # 1. 初始化 TF2 監聽器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 2. 訂閱 /rtk 僅用於擷取起點的全球座標 (Datum)
        self.rtk_subscription = self.create_subscription(
            NavSatFix,
            '/rtk',
            self.rtk_callback,
            10)

        # 座標與狀態暫存
        self.current_lat = None
        self.current_lon = None
        self.origin_lat = None
        self.origin_lon = None
        
        self.polygon_points = []
        self.last_x = 0.0
        self.last_y = 0.0
        
        # 設定：每移動 1.0 公尺記錄一個點
        self.dist_threshold = 1.0

        # 建立定時器，定期檢查 TF（10Hz）
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(f"TF 混合 RTK 記錄器已啟動。間隔設定：{self.dist_threshold}米")
        self.get_logger().info("機制：由 TF 紀錄平滑軌跡，並鎖定第一個點的 GPS 作為基準點 (Datum)。")

    def rtk_callback(self, msg):
        # 持續更新當前的經緯度（如果有需要，可取消註解狀態檢查）
        # if msg.status.status < 2: return
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude

    def timer_callback(self):
        try:
            # 尋找從 map 到 base_link 的最新轉換
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                now)
            
            x = trans.transform.translation.x
            y = trans.transform.translation.y

            # 處理第一個點
            if not self.polygon_points:
                # 確保必須先收到至少一個 GPS 訊號才允許建立起點
                if self.current_lat is None:
                    self.get_logger().warn("等待首次 RTK 訊號以鎖定地圖基準點 (Datum)...", throttle_duration_sec=2.0)
                    return
                
                # 鎖定當前經緯度作為這張地圖邊界的 Datum
                self.origin_lat = self.current_lat
                self.origin_lon = self.current_lon
                self.get_logger().info(f"[成功鎖定基準點] Lat: {self.origin_lat}, Lon: {self.origin_lon}")
                
                self.add_point(x, y)
                return

            # 計算與上一個記錄點的距離 (使用 TF 座標)
            dist = math.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)

            if dist >= self.dist_threshold:
                self.add_point(x, y)

        except TransformException:
            # 忽略未準備好的 TF 異常
            pass

    def add_point(self, x, y):
        self.polygon_points.append([round(x, 4), round(y, 4)])
        self.last_x = x
        self.last_y = y
        self.get_logger().info(f"已記錄點 {len(self.polygon_points)}: [X={x:.2f}, Y={y:.2f}] (相對於 map 原點)")

    def save_polygon(self):
        # 1. 打印鎖定的原點經緯度資訊 (Datum)
        print("origin_info:")
        print(f"  latitude, longitude: {self.origin_lat}, {self.origin_lon}")
        print("---")
        
        # 2. 打印多邊形邊界
        self.get_logger().info("\n--- 採集完成！YAML 格式如下 ---")
        print("map_config:")
        print("  # 搜尋範圍區域 (不規則多邊形，座標系：map)")
        print("  search_boundary:")
        print("    points:")
        for p in self.polygon_points:
            print(f"      - [{p[0]}, {p[1]}]")
            
        self.get_logger().info(f"總共記錄了 {len(self.polygon_points)} 個點。")

def main():
    rclpy.init()
    node = TfRtkBoundaryRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_polygon()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()