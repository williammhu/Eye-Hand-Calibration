from openni import openni2
from openni import _openni2 as c_api



def initialize_openni(redist_path):
    openni2.initialize(redist_path)
    return openni2.Device.open_any()

def configure_depth_stream(device, width, height, fps):
    depth_stream = device.create_depth_stream()
    depth_stream.set_video_mode(c_api.OniVideoMode(
        pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
        resolutionX=width, resolutionY=height, fps=fps
    ))
    depth_stream.start()
    return depth_stream
def convert_depth_to_xyz(u, v, depth_value, fx, fy, cx, cy):
    depth = depth_value
    z = float(depth)
    x = float((u - cx) * z) / fx
    y = float((v - cy) * z) / fy
    return x, -y, z
