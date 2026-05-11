import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import yaml
import os
from geometry_msgs.msg import Point32, Polygon
from opennav_coverage_msgs.action import NavigateCompleteCoverage
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class CoverageManager(Node):
    def __init__(self):
        super().__init__('coverage_manager')
        self._action_client = ActionClient(self, NavigateCompleteCoverage, '/navigate_complete_coverage')
        
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization/forbidden_zones', 10)
        
        # 讀取配置 (確保路徑正確)
        self.config_path = 'zone_config.yaml'
        self.config = self.load_config(self.config_path)
        
        # 修正：將 timer 賦值給 self.one_off_timer 才能在 callback 裡取消
        self.one_off_timer = self.create_timer(1.0, self.timer_callback)

    def load_config(self, path):
        if not os.path.exists(path):
            self.get_logger().error(f"找不到設定檔: {path}")
            return None
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def timer_callback(self):
        """定時器只執行一次發送指令"""
        self.one_off_timer.cancel() # 停止定時器
        self.send_goal()

    def send_goal(self):
        if self.config is None: 
            self.get_logger().error("配置為空，取消發送")
            return

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("找不到 OpenNav Coverage Action Server")
            return

        self.publish_forbidden_markers() # 同時發布禁區的 Marker 到 RViz
        
        goal_msg = NavigateCompleteCoverage.Goal()
        # goal_msg.header.stamp = rclpy.time.Time().to_msg()
        goal_msg.frame_id = 'map'
        
        all_polygons = []

        # --- 處理邊界 ---
        boundary_points = self.config['map_config']['search_boundary']['points']
        all_polygons.append(self.create_poly_msg(boundary_points))

        # --- 處理禁區 ---
        if 'forbidden_zones' in self.config['map_config']:
            for zone in self.config['map_config']['forbidden_zones']:
                all_polygons.append(self.create_poly_msg(zone['points']))
                self.get_logger().info(f"已加入禁區: {zone.get('name', '未命名')}")

        goal_msg.polygons = all_polygons
        self.get_logger().info(f"🚀 發送請求：1 邊界 + {len(all_polygons)-1} 禁區")
        
        # 修正：必須將 callback 掛載到 async 呼叫上
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)
        
    def create_poly_msg(self, points):
        # """輔助函式：將 list 轉為 Polygon 訊息"""
        # poly = Polygon()
        # for pt in points:
        #     p = Point32()
        #     p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.0
        #     poly.points.append(p)
        # return poly
        """輔助函式：將 list 轉為 Polygon 訊息，並確保首尾相連"""
        if not points:
            return Polygon()

        poly = Polygon()
        # 轉換所有點
        for pt in points:
            p = Point32()
            p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.0
            poly.points.append(p)

        # --- 關鍵修正：檢查是否需要閉合 ---
        # 比較第一個點與最後一個點的坐標
        first = points[0]
        last = points[-1]
        
        # 如果最後一點不等於第一點，則重複第一點
        if first[0] != last[0] or first[1] != last[1]:
            p_close = Point32()
            p_close.x, p_close.y, p_close.z = float(first[0]), float(first[1]), 0.0
            poly.points.append(p_close)
            self.get_logger().debug("多邊形未閉合，已自動補上首點以封閉區域。")

        return poly

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal 請求被拒絕')
            return
        self.get_logger().info('Goal 已接受，正在規劃路徑...')

    def publish_forbidden_markers(self):
        if 'forbidden_zones' not in self.config['map_config']:
            return

        marker_array = MarkerArray()
        
        for i, zone in enumerate(self.config['map_config']['forbidden_zones']):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "forbidden_zones"
            marker.id = i
            marker.type = Marker.LINE_STRIP # 線條模式，適合多邊形
            marker.action = Marker.ADD
            
            # 設定顏色 (紅色，半透明)
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.6)
            marker.scale.x = 0.05 # 線條粗細
            
            # 轉換頂點
            points = zone['points']
            # 為了閉合多邊形，我們把最後一點連回第一點
            closed_points = points + [points[0]]
            
            for pt in closed_points:
                p = Point32() # 這裡 Marker 使用的是 geometry_msgs/Point，但功能類似
                from geometry_msgs.msg import Point
                p_msg = Point()
                p_msg.x, p_msg.y, p_msg.z = float(pt[0]), float(pt[1]), 0.01 # 稍微浮起來一點點
                marker.points.append(p_msg)
                
            marker_array.markers.append(marker)
            
        self.marker_pub.publish(marker_array)
    
def main(args=None):
    rclpy.init(args=args)
    node = CoverageManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()