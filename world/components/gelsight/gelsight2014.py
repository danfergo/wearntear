from yarok import component, interface
from yarok.comm.components.cam.cam import Cam
from yarok.platforms.mjc.interface import InterfaceMJC

import cv2
import os
import numpy as np
from .model.simulation_model_wear import SimulationApproach


from yarok import ConfigBlock
@interface()
class GelSight2014InterfaceMJC:

    def __init__(self, interface: InterfaceMJC):
        self.interface = interface
        self.raw_depth = np.zeros((480, 640))
        self.raw_rgb = np.zeros((480, 640, 3))
        self.tactile_rgb = np.zeros((480, 640))

        self.approach = SimulationApproach(**{
            'light_sources': [
                {'position': [0, 1, 0.25], 'color': (255, 255, 255), 'kd': 0.6, 'ks': 0.5},  # white, top
                {'position': [-1, 0, 0.25], 'color': (255, 130, 115), 'kd': 0.5, 'ks': 0.3},  # blue, right
                {'position': [0, -1, 0.25], 'color': (108, 82, 255), 'kd': 0.6, 'ks': 0.4},  # red, bottom
                {'position': [1, 0, 0.25], 'color': (120, 255, 153), 'kd': 0.1, 'ks': 0.1}  # green, left
                # {'position': [-1, 0, 0.25], 'color': (255, 255, 255), 'kd': 0.5, 'ks': 0.3},  # white, top
                # {'position': [0, 0, 1], 'color': (255, 255, 255), 'kd': 0.5, 'ks': 0.3},  # blue, right
            ],
            'background_img': cv2.imread(os.path.dirname(__file__) + '/background_gelsight2014.png'),
            'ka': 0.8,
            'px2m_ratio': 5.4347826087e-05,
            'elastomer_thickness': 0.004,
            'min_depth': 0.03,
            'texture_sigma': 0.000002
        })

    def read(self, shape=(480, 640)):
        self.raw_rgb, self.raw_depth = self.interface.read_camera('camera', shape, True)
        self.tactile_rgb = self.approach.generate(self.raw_depth, self.raw_rgb)
        return self.tactile_rgb

    def read_wear(self):
        return self.approach.wear_mask

    def read_tear(self):
        return self.approach.tear_mask

    def read_depth(self):
        return self.raw_depth

@interface()
class GelSight2014InterfaceHW:

    def __init__(self, config: ConfigBlock):
        self.cap = cv2.VideoCapture(config['cam_id'])
        if not self.cap.isOpened():
            raise Exception('GelTip cam ' + str(config['cam_id']) + ' not found')

        self.fake_depth = np.zeros((480, 640), np.float32)

    def read(self):
        [self.cap.read() for _ in range(10)]  # skip frames in the buffer.
        ret, frame = self.cap.read()
        return frame

    def read_depth(self):
        return self.fake_depth

@component(
    tag='gelsight2014',
    components=[
        Cam
    ],
    defaults={
        'interface_mjc': GelSight2014InterfaceMJC,
        'interface_hw': GelSight2014InterfaceHW,
        'probe': lambda c: {'gelsight camera': c.read()}
    },
    # language=xml
    template="""
<mujoco>
    <asset>
        <material name="black_resin" rgba="0.1 0.1 0.1 1"/>
        <material name="gray_elastomer" rgba="0.8 0.8 0.8 1"/>
        <material name="transparent_glass" rgba="0.9 0.95 1 0.7"/>

        <!-- gelsight models-->
        <mesh name="gelsight_front" file="meshes/gelsight2014_front.stl" scale="0.001 0.001 0.001"/>
        <mesh name="gelsight_back" file="meshes/gelsight2014_back.stl" scale="0.001 0.001 0.001"/>
        <mesh name="gelsight_glass" file="meshes/glass.stl" scale="0.0009 0.00125 0.001"/>

        <mesh name="front_cover" file="meshes/mountable_gelsight.stl" scale="0.001 0.001 0.001"/>
        <mesh name="back_cover" file="meshes/back_cover.stl" scale="0.001 0.001 0.001"/>

    </asset>
    <worldbody>
        <body name="sensor_body" pos="0 0 0">
            <!--Front and Back-->
            <geom name="front" type="mesh" material="black_resin" mesh="front_cover" mass="0.05" contype="32"
                  conaffinity="32" friction="1 0.05 0.01" solimp="1.1 1.2 0.001 0.5 2" solref="0.02 1"/>
            <geom name="back" type="mesh" material="black_resin" mesh="back_cover" mass="0.05" contype="32"
                  conaffinity="32" friction="1 0.05 0.01" solimp="1.1 1.2 0.001 0.5 2" solref="0.02 1"/>

            <!-- Glass Cover-->
             <!--<geom name="glass0" type="mesh" material="transparent_glass" mesh="gelsight_glass" mass="0.005"
                  contype="32" conaffinity="32" pos="-0.011 0 0.029"/>
             
            <geom name="glass1" type="mesh" material="transparent_glass" mesh="gelsight_glass" mass="0.005"
                  contype="32" conaffinity="32" pos="0.0115 0 0.029" quat="0 0 0 1"/> -->

            <!-- Elastomer -->
<!--            <geom name="elastomer" type="box" size="0.013 0.013 0.001" euler="0 0 0" pos="0 0 0.033"
                  contype="0"
                  conaffinity="32" rgba="0.9 0.95 1 0.1"/>
-->
            <!-- Elastomer Cover -->
            
          <!--  <geom name="elast_cover" type="box" size="0.013 0.013 0.00001" euler="0 0 0" pos="0 0 0.034001"
                  contype="0" conaffinity="32" material="black_resin"
                  friction="1 0.05 0.01" solimp="1.1 1.2 0.001 0.5 2" solref="0.02 1"/> -->

            <!-- Gel Camera -->
            <camera name="camera" mode="fixed" pos="0 0 0.001" zaxis="0 0 -1" fovy="20"/>

            <!-- Friction placholder -->
            <geom name="friction" type="box"
                 size="0.013 0.013 0.00001"
                 euler="0 0 0"
                 pos="0 0 0.032"
                 contype="32"
                 conaffinity="32" rgba="0 0 0 0"
                 friction="1 0.05 0.01" solimp="1.1 1.2 0.001 0.5 2" solref="0.02 1"/>
        </body>
    </worldbody>
</mujoco>
    """
)
class GelSight2014:

    def __init__(self):
        """
            GelSight 2014 driver as proposed in "Generation of GelSight Tactile Images for Sim2Real Learning＂
            https://danfergo.github.io/gelsight-simulation/

            The frame method gets the depth map from the simula
        """

    def read(self):
        pass

    def read_depth(self):
        pass

    def read_wear(self):
        pass
