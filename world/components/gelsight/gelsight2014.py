import time

from yarok import component, interface
from yarok.comm.components.cam.cam import Cam
from yarok.platforms.mjc.interface import InterfaceMJC

import cv2
import os
import numpy as np

from ..sim_model.wnt_model import WNTSimulationModel
# from .model.simulation_model_wear import SimulationApproach

from yarok import ConfigBlock

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

from threading import Thread


def run_simulation(sensor_interface):
    while True:

        sensor_interface.tactile_rgb = sensor_interface.model.generate(sensor_interface.raw_depth, sensor_interface.raw_rgb)
        # time.sleep(0.0001)

@interface(defaults={
    'frame_size': (480, 640),
    'field_size': (120, 160),
    'field_name': 'geodesic',
    'elastic_deformation': True,
    'texture_sigma': 0.000005,
    'ia': 0.8,
    'fov': 40,
    'light_constants': [
        # {'color': [196, 94, 255], 'id': 0.5, 'is': 0.1},  # red # [108, 82, 255]
        # {'color': [154, 144, 255], 'id': 0.5, 'is': 0.1},  # green # [255, 130, 115]
        # {'color': [104, 175, 255], 'id': 0.5, 'is': 0.1},  # blue  # [120, 255, 153]
        {'direction': [0, 1, 0], 'color': (255, 255, 255), 'id': 0.6, 'is': 0.5},  # white, top
        {'direction': [-1, 0, 0], 'color': (255, 130, 115), 'id': 0.5, 'is': 0.3},  # blue, right
        {'direction': [0, -1, 0], 'color': (108, 82, 255), 'id': 0.6, 'is': 0.4},  # red, bottom
        {'direction': [1, 0, 0], 'color': (120, 255, 153), 'id': 0.1, 'is': 0.1}  # green, left
    ],
})
class GelSight2014InterfaceMJC:

    def __init__(self, interface: InterfaceMJC, config: ConfigBlock):
        self.config = config
        self.interface = interface
        self.raw_depth = np.zeros((480, 640))
        self.raw_rgb = None
        self.tactile_rgb = np.zeros((480, 640))
        self.step_count = 0
        self.raw_rgb, self.raw_depth = self.interface.read_camera('camera', (480, 640), True)
        self.cam_hz = 0.005
        self.next_update = time.time() + self.cam_hz

        # try:
        real_bkg = cv2.imread(os.path.join(__location__, 'background_2024.jpg'))
        real_bkg = cv2.cvtColor(real_bkg, cv2.COLOR_BGR2RGB) / 255.0
        self.real_bkg = real_bkg

        self.fields = [np.ones((480, 640, 3)) * np.array(c['direction']) for c in config['light_constants']]

        self.model = WNTSimulationModel(**{
            'ia': config['ia'],
            'fov': config['fov'],
            'light_sources': [{
                'field': self.fields[l],
                **config['light_constants'][l]}
                for l in range(len(config['light_constants']))
            ],
            # the depth map is obtained by running the sim model once and saving the depth map with no contact.
            'background_depth': np.load(f'{__location__}/bkg_depth.npy'),
            # 'cloud_map': cloud,
            'rectify_fields': True,
            'background_img': self.real_bkg,
            # bkg_rgb if use_bkg_rgb else
            'texture_sigma': config['texture_sigma'],
            'elastic_deformation': config['elastic_deformation']
        })

        Thread(target=run_simulation, args=[self]).start()

    def read(self, shape=(480, 640)):
        # self.raw_rgb, self.raw_depth = self.interface.read_camera('camera', shape, True)
        # t = time.time()
        self.tactile_rgb = self.model.generate(self.raw_depth, self.raw_rgb)
        # print('elapsed', time.time() - t)
        # self.tactile_rgb = self.model.generate(self.raw_depth, self.raw_rgb)
        return self.tactile_rgb

    def read_wear(self):
        return self.model.wear_mask

    def read_tear(self):
        return self.model.tear_mask

    def read_depth(self):
        print('read depth ??')
        return self.raw_depth

    def step(self):
        if time.time() > self.next_update:
            print('timestep ... !!')
            self.raw_rgb, self.raw_depth = self.interface.read_camera('camera', (480, 640), True)
            cv2.imshow('cam', self.raw_rgb)
            cv2.waitKey(1)
            self.next_update = time.time() + self.cam_hz


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

    def read_wear(self):
        return None

    def read_tear(self):
        return None

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
        <material name="translucid_glass" rgba="0.9 0.95 1 0.7"/>
        <material name="transparent_glass" rgba="1 1 1 1.0"/>

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
            <geom name="elastomer" type="box" 
                  material="translucid_glass"
                  size="0.013 0.013 0.001" 
                  euler="0 0 0" pos="0 0 0.033"
                  contype="0"
                  conaffinity="32" rgba="0.9 0.95 1 0.1"/>

            <!-- Depth-map Thresholding -->
            <!-- <geom name="elast_cover" type="box" 
                  size="0.013 0.013 0.00001" euler="0 0 0" pos="0 0 0.034001"
                  contype="0" conaffinity="32" material="transparent_glass"
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

            The frame method gets the depth map from the simulation
        """

    def read(self):
        pass

    def read_depth(self):
        pass

    def read_wear(self):
        pass
