import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image

'''def initialize_camera(width=640, height=480, format=rs.format.bgr8, fps=5):
    """
    初始化 RealSense 相机管道并返回管道对象。
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, format, fps)
    pipeline.start(config)
    return pipeline
'''

def initialize_camera(camera1, camera2):
    """
    初始化所有相机
    根据实际情况替换 serial_numbers 中的序列号。
    返回一个字典，键为相机型号，值为相机的管道对象。
    """
    # 创建上下文对象
    context = rs.context()
    connected_devices = [device.get_info(rs.camera_info.serial_number) for device in context.devices]
    print("Connected camera serials:", connected_devices)
    
    # 映射相机型号到序列号（请替换为您的相机实际序列号）
    serial_numbers = {
        'L515': camera1, # 'f0210138', # 'f0265239', f0210138    # 替换为 L515 的序列号 f0265239 f0210138
        'D435i': camera2, # '332522071841',
        # 'D455': '', # 替换为 D455 的序列号
        # 'D435': ''  # 替换为 D435 的序列号
    }

    # 初始化每个相机的管道
    pipelines = {}
    for model, serial in serial_numbers.items():
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        # 这里分辨率和帧率不要改 改了可能会出现问题
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(cfg)
        pipelines[model] = pipeline
        print(f"Initialized camera {model} with serial number {serial}.")
    
    return pipelines
'''
def get_frame(pipeline, target_width=384, target_height=384):
    """
    从已经启动的管道中获取一帧图像，返回调整为指定分辨率的 PIL.Image 格式。
    """
    # 获取图像帧
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        return None

    # 将彩色图像转换为 numpy 格式
    color_image = np.asanyarray(color_frame.get_data())

    # 调整图像为目标分辨率
    resized_image = cv2.resize(color_image, (target_width, target_height))

    # 将 BGR (OpenCV 格式) 转换为 RGB (PIL 格式)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # 将 numpy 数组转换为 PIL.Image 格式
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image
'''

def get_L515_image(pipelines):
    """
    从 pipelines 中获取 L515 相机的图像，转换为 PIL Image 并调整为 (384, 384) 大小。
    """
    if 'L515' not in pipelines:
        print("L515 camera not found in pipelines.")
        return None
    pipeline = pipelines['L515']
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("No color frame captured from L515 camera.")
        return None
    color_image = np.asanyarray(color_frame.get_data())
    image = Image.fromarray(color_image)
    image = image.resize((224, 224)) # size for openvla
    # image = image.resize((640, 480)) # resize for pi0

    return image


def get_D435_image(pipelines):
    """
    从 pipelines 中获取 D435 相机的图像，转换为 PIL Image 并调整为 (384, 384) 大小。
    """
    if 'D435i' not in pipelines:
        print("D435i camera not found in pipelines.")
        return None
    pipeline = pipelines['D435i']
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("No color frame captured from D435i camera.")
        return None
    color_image = np.asanyarray(color_frame.get_data())
    image = Image.fromarray(color_image)
    image = image.resize((224, 224)) # size for openvla
    # image = image.resize((640, 480)) # resize for pi0

    return image



def stop_camera(pipeline):
    """
    停止相机管道。
    """
    pipeline.stop()
