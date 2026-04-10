# 執行這個小腳本 list_images.py
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient

sdk = bosdyn.client.create_standard_sdk('ListImageSources')
robot = sdk.create_robot("10.0.0.3") # 換成你的 Spot IP
robot.authenticate("admin", "eqyqp33u8i74")

image_client = robot.ensure_client(ImageClient.default_service_name)
sources = image_client.list_image_sources()

print(f"{'Source Name':<30} | {'Type':<15}")
print("-" * 50)
for source in sources:
    print(f"{source.name:<30} | {source.image_type}")