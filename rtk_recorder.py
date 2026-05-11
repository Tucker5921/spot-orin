import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
import math

class RtkRecorder(Node):
    def __init__(self):
        super().__init__('rtk_recorder')
        
        # 訂閱 /rtk
        self.subscription = self.create_subscription(
            NavSatFix,
            '/rtk',
            self.rtk_callback,
            10)

        self.origin_lat = None
        self.origin_lon = None
        self.polygon_points = []
        self.last_x = 0.0
        self.last_y = 0.0
        
        # 修改：每移動 5.0 公尺記錄一個點
        self.dist_threshold = 1.0

        self.get_logger().info(f"RTK {self.dist_threshold}米間隔記錄器已啟動。")
        self.get_logger().info("當前設定：X軸=正東(0度), Y軸=正北(90度)")

    def latlon_to_meters(self, lat, lon):
        """
        採用 WGS84 橢球體局部投影公式
        """
        if self.origin_lat is None:
            self.origin_lat = lat
            self.origin_lon = lon
            self.get_logger().info(f"設定起始基準點 (Datum): Lat={lat}, Lon={lon}")
            return 0.0, 0.0
        
        # 台灣緯度 (約25度) 下的經緯度長度換算
        lat_rad = math.radians(self.origin_lat)
        
        # 緯度每度約 111132 公尺
        meters_per_lat = 111132.0 
        # 經度每度約 111319 * cos(lat) 公尺
        meters_per_lon = 111319.0 * math.cos(lat_rad)
        
        y = (lat - self.origin_lat) * meters_per_lat
        x = (lon - self.origin_lon) * meters_per_lon
        
        return x, y

    def rtk_callback(self, msg):
        # 檢查 RTK 狀態 (4 代表 RTK Fixed，最精準)
        # 注意：某些設備在 ROS2 message 中固定解會回傳 2
        print(
            '1'
        )
        # if msg.status.status < 2:
        #     return

        x, y = self.latlon_to_meters(msg.latitude, msg.longitude)

        # 第一個點
        if not self.polygon_points:
            self.add_point(x, y)
            return

        # 計算與上一個記錄點的距離
        dist = math.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)

        if dist >= self.dist_threshold:
            self.add_point(x, y)

    def add_point(self, x, y):
        self.polygon_points.append([round(x, 4), round(y, 4)])
        self.last_x = x
        self.last_y = y
        self.get_logger().info(f"已記錄點 {len(self.polygon_points)}: [X={x:.2f}, Y={y:.2f}] (距離起點約 {math.sqrt(x**2+y**2):.1f}m)")

    def save_polygon(self):
        # 1. 打印原點資訊 (Datum)
        print("origin_info:")
        print(f"  latitude, longitude: {self.origin_lat}, {self.origin_lon}")
        print("---")
        self.get_logger().info("\n--- 採集完成！YAML 格式如下 ---")
        print("map_config:")
        print("  # 搜尋範圍區域 (不規則多邊形)")
        print("  search_boundary:")
        print("    points:")
        for p in self.polygon_points:
            # 輸出格式： - [x, y]
            print(f"      - [{p[0]}, {p[1]}]")
        
        # 提示：為了讓多邊形閉合，Nav2 通常會自動連回第一個點，
        # 但有些插件需要手動重複第一個點，如有需要可取消下方註解：
        # if self.polygon_points:
        #     p = self.polygon_points[0]
        #     print(f"      - [{p[0]}, {p[1]}]")
            
        self.get_logger().info(f"總共記錄了 {len(self.polygon_points)} 個點。")

def main():
    rclpy.init()
    node = RtkRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_polygon()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()