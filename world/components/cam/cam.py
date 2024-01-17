from yarok import component, interface, ConfigBlock
from yarok.platforms.mjc import InterfaceMJC
from yarok.platforms.mjc import InterfaceMJC

import numpy as np
import cv2
import os

@interface(
    defaults={
        'resolution': (640, 480)
    }
)
class CameraInterfaceMJC:

    def __init__(self, mjc: InterfaceMJC, config: ConfigBlock):
        self.interface = mjc
        self.resolution = config['resolution']

    def read(self, depth=False, shape=None):
        shape = (self.resolution[1], self.resolution[0]) if shape is None else shape
        return self.interface.read_camera('cam', shape=shape, depth=depth, rgb=True)



@interface()
class CameraInterfaceHW:

    def __init__(self, config: ConfigBlock):
        self.cap = cv2.VideoCapture(config['cam_id'])
        if not self.cap.isOpened():
            raise Exception('Camera with id ' + str(config['cam_id']) + ' not found')

        self.fake_depth = np.zeros((480, 640), np.float32)

    def read(self):
        [self.cap.read() for _ in range(10)]  # skip frames in the buffer.
        ret, frame = self.cap.read()
        return frame

@component(
    defaults={
        'interface_mjc': CameraInterfaceMJC,
        'interface_hw': CameraInterfaceHW,
        'probe': lambda c: {'camera': c.read()}
    },
    # language=xml
    template="""
        <mujoco>
            <asset>
                <material name="black_plastic" rgba=".3 .3 .3 1"/>
                <material name="gray_ring" rgba="1 1 1 1"/>
            </asset>
            <worldbody>
                <body>
                    <geom type='box' size='0.08 0.04 0.02' pos='0 0 -0.005' material='black_plastic'/>
                    <geom type='cylinder' size='0.02 0.02' material='gray_ring'/>
                    <camera name="cam" zaxis='0 0 -1'/>
                </body>
            </worldbody>
        </mujoco>
    """
)
class Cam:

    def __init__(self):
        pass

    def read(self, depth=False, shape=(480, 640)):
        # Implemented by the interface
        pass


if __name__ == '__main__':
    from yarok import Platform
    from yarok.comm.worlds.empty_world import EmptyWorld
    from yarok.comm.plugins.cv2_inspector import Cv2Inspector


    @component(
        extends=EmptyWorld,
        components=[
            Cam
        ],
        # language=xml
        template="""
            <mujoco>
                <worldbody>
                    <body pos='0 0 0.5' zaxis='0 0 -1'>
                        <cam name='test_cam'/>
                    </body>
                </worldbody>
            </mujoco>
        """
    )
    class CamTestWorld:
        pass


    Platform.create({
        'world': CamTestWorld,
        'defaults': {
            'plugins': [
                (Cv2Inspector, {})
            ]
        }
    }).run()
