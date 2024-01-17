import time
from random import sample, randint, choice

import yaml

from worlds.shared.cross_spawn import run_parallel, run_sequential, run_all
from worlds.shared.memory import Memory
from worlds.shared.robotbody import RobotBody
from .components.geltip.geltip import GelTip
from .components.tumble_tower.tumble_tower import TumbleTower
from .components.ur5e.ur5e import UR5e
from .components.robotiq_2f85.robotiq_2f85 import Robotiq2f85
from .components.cam.cam import Cam

from yarok import Platform, PlatformMJC, PlatformHW, Injector, component, ConfigBlock
from yarok.comm.worlds.empty_world import EmptyWorld

from math import pi

colors = {
    'red': [1, 0, 0],
    'green': [0, 1, 0],
    'blue': [0, 0, 1],
    'yellow': [1, 1, 0],
    'magenta': [1, 0, 1],
    'cyan': [0, 1, 1]
}


def color_map(c, s=0.8):
    if c in colors:
        color = colors[c]
        return f'{color[0] * s} {color[1] * s} {color[2] * s}'
    return c


@component(
    extends=EmptyWorld,
    components=[
        GelTip,
        TumbleTower,
        UR5e,
        Robotiq2f85,
        Cam
    ],
    defaults={
        'color_map': color_map,
        'bs': 0.03,
        'sx': 0.11,
        'sy': 0,
        'ex': 0,
        'ey': 0,
        'pick_blocks': ['red', 'green', 'blue', 'cyan'],
        # 'placed_blocks': ['yellow', 'green'],
    },
    template="""
        <mujoco>
            <option impratio="50" noslip_iterations="15"/>
            <asset>
                <texture type="skybox" 
                    file="assets/robot_lab.png"
                    rgb1="0.6 0.6 0.6" 
                    rgb2="0 0 0"/>
                <texture 
                    name="wood_texture"
                    type="cube" 
                    file="assets/white_wood.png"
                    width="400" 
                    height="400"/>
                <material name="wood" texture="wood_texture" specular="0.1"/>
                <material name="gray_wood" texture="wood_texture" rgba="0.6 0.4 0.2 1" specular="0.1"/>
                <material name="white_wood" texture="wood_texture" rgba="0.6 0.6 0.6 1" specular="0.1"/>
                
                <mesh name="female_threads" file="assets/female_threads.stl" scale="0.001 0.001 0.001"/>
                <mesh name="threaded_flask" file="assets/threaded_flask.stl" scale="0.001 0.001 0.001"/>
                <material name="glass_material" rgba="1 1 1 0.3" specular="1.0"/>

            </asset>
            <default>
                <default class='pp-block'>
                     <geom type="box" 
                           size="{bs} {bs} {bs}"
                           mass="0.0001"
                           material="wood"
                           zaxis="0 1 0"/>
                </default>
            </default>
            <worldbody>
                <light directional="true" 
                    diffuse="{color_map(light, 0.1)}" 
                    specular="{color_map(light, 0.1)}" 
                    pos="1.0 1.0 5.0" 
                    dir="0 -1 -1"/>
                <body pos="{0.3 + p_cam[0]*0.1} -1.3 {0.6 + p_cam[0]*0.1}" euler="1.57 -3.14 0">
                    <cam name="cam" />
                </body>
                
                <!-- pick blocks -->
                <!-- <for each="range(len(pick_blocks))" as="i">
                    <body>
                        <freejoint/>
                        <geom 
                            class="pp-block" 
                            pos="{0.13 + sx*0.01} {-0.135 + sy*0.01} {0.131 + i*2*bs}" 
                            rgba="{color_map(pick_blocks[i])} 1"/>
                    </body>
                </for>-->
                
               <!-- <for each="range(len(placed_blocks))" as="i">
                    <body>
                        <freejoint/>
                        <geom 
                            class="pp-block"
                            pos="{0.43 + ex*0.01} {-0.135 + ey*0.01} {0.131 + i*2*bs}" 
                            rgba="{color_map(pick_blocks[i])} 1"/>
                    </body>
                </for> -->
                
               <body pos="0.3 0.11 0" name="table_base">
                    <geom type="box" pos="-0.45 0.29 0" size="0.1 0.1 0.3" material="gray_wood"/>
                    <geom type="box" pos="0 0 0" size="0.4 0.4 0.1" material="white_wood"/>
               </body>  
               
               <body pos="0.3 0.11 0" name="glassware">
                    <!--<freejoint />-->
                    <!--<geom  
                          density="0.1" 
                          type="mesh" 
                          mesh="female_threads" 
                          pos="0.1 0.1 0.3"
                          zaxis="0 1 0"
                          solimp="1.0 1.2 0.001 0.5 2" 
                          solref="0.02 1"
                          mass="0.1"
                          material="glass_material" /> -->
                      <geom  
                          density="1" 
                          type="mesh" 
                          mesh="threaded_flask" 
                          pos="0.0 0.0 0.1"
                          zaxis="0 -1 0"
                          material="glass_material" /> 
               </body>  

                
                <body euler="0 0 1.57" pos="-0.15 0.4 0.3">
                    <ur5e name='arm'>
                       <robotiq-2f85 name="gripper" left_tip="{True}" right_tip="{True}" parent="ee_link"> 
                          <body pos="0.02 -0.017 0.053" xyaxes="0 -1 0 1 0 0" parent="right_tip">
                                <geltip name="left_geltip" cubic_core="{True}" label_color="255 0 0"/>
                            </body>
                           <body pos="-0.02 -0.017 0.053" xyaxes="0 1 0 -1 0 0" parent="left_tip">
                                <geltip name="right_geltip" cubic_core="{True}" label_color="0 255 0"/>
                            </body>
                        </robotiq-2f85> 
                    </ur5e> 
                </body>
            </worldbody>
        </mujoco>
    """
)
class BlocksTowerTestWorld:
    pass


def z(pos, delta):
    new_pos = pos.copy()
    new_pos[2] += delta
    return new_pos

import cv2

class PickAndPlaceBehaviour:

    def __init__(self, injector: Injector, config: ConfigBlock):
        self.body = RobotBody(injector)
        self.body.arm.set_ws([
            [- pi, pi],  # shoulder pan
            [- pi, -pi / 2],  # shoulder lift,
            [- 2 * pi, 2 * pi],  # elbow
            [-2 * pi, 2 * pi],  # wrist 1
            [0, pi],  # wrist 2
            [- 2 * pi, 2 * pi]  # wrist 3
        ])
        self.body.arm.set_speed(pi / 24)
        self.pl: Platform = injector.get(Platform)
        self.config = config

        self.memory = Memory(self.body, self.config, skip_right_sensor=False)


        self.DOWN = [3.11, 1.6e-7, 3.11]
        self.BLOCK_SIZE = 0.06
        print('-------> ', config['sx'])
        sx = config['sx'] * 0.01
        sy = config['sy'] * 0.01
        ex = config['ex'] * 0.01
        ey = config['ey'] * 0.01
        self.pick_blocks = config['pick_blocks']

        self.START_POS_UP = [0.3 + sx, -0.5 + sy, 0.21]
        self.START_POS_DOWN = [0.3 + sx, -0.5 + sy, 0.11]
        self.END_POS_UP = [0.6 - ex, -0.5 - ey, 0.21]
        self.END_POS_DOWN = [0.6 - ex, -0.5 - ey, 0.115]

    def wait(self, arm=None, gripper=None):
        def cb():
            self.memory.save()

            self.pl.wait_seconds(0.5)

            if arm is not None:
                return self.body.arm.is_at(arm)
            else:
                return self.body.gripper.is_at(gripper)

        self.pl.wait(cb)

    def on_start(self):
        def move_arm(p):
            q = self.body.arm.ik(xyz=p, xyz_angles=self.DOWN)
            if q is None:
                print(p)

            self.body.arm.move_q(q)
            self.wait(arm=q)

        def move_gripper(q):
            self.body.gripper.close(q)
            self.wait(gripper=q)

        # set the arm in the initial position
        print('MOVED ORIGIN', [0, 0, 0, 0, 0, 0])
        self.pl.wait(self.body.arm.move_q([0, 0, 0, 0, 0, 0]))
        print('MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMOVED ORIGIN')
        self.pl.wait_seconds(10)
        self.pl.wait(self.body.arm.move_xyz(xyz=self.START_POS_UP, xyz_angles=self.DOWN))


        # cv2.imwrite(f'bkg.jpg', self.memory.prepare_frame(self.body.cam.read()))
        # exit(-1)

        self.memory.prepare()


        # do the pick and place.
        for i in range(len(self.pick_blocks)):
            # before grasping.
            move_arm(self.START_POS_UP)

            # grasps block.
            # move_arm(z(self.START_POS_DOWN, -i * self.BLOCK_SIZE))
            # move_gripper(0.26)
            #
            # # moves.
            # move_arm(self.START_POS_UP)
            # move_arm(self.END_POS_UP)

            # places.
            # move_arm(z(self.END_POS_DOWN, -(2 - i) * self.BLOCK_SIZE))
            # move_gripper(0)

            # # moves back
            # move_arm(self.END_POS_UP)
            # move_arm(self.START_POS_UP)


def launch_world(**kwargs):
    Platform.create({
        'world': BlocksTowerTestWorld,
        'behaviour': PickAndPlaceBehaviour,
        'defaults': {
            'environment': 'real',
            'behaviour': kwargs,
            'components': {
                '/': kwargs
            }
        },
        'environments': {
            'sim': {
                'platform': {
                    'class': PlatformMJC
                },
            },
            'real': {
                'platform': {
                    'class': PlatformHW,
                    'interfaces': {
                        '/left_geltip': {
                            'cam_id': 0
                        },
                        '/right_geltip': {
                            'cam_id': "/dev/video2"
                        },
                        '/cam': {
                            'cam_id': 5
                        }
                    }
                },
                
            }
        }
        

    }).run()


if __name__ == '__main__':
    launch_world(**{
        'data_path': '/home/danfergo/data',
        'dataset_name': 'grasp_rw',
        'it': 1,
        'sx': 0.12,
        'sy': 0,
        'ex': 0,
        'ey': 0,
        'p_cam': (0, 0),
        'pick_blocks': ['red'],
        'placed_blocks': [],
        'light': 'red'
    })