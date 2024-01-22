import time
from random import sample, randint, choice

from world.components.gelsight.gelsight2014 import GelSight2014
from world.shared.colors import colors_names, color_map
from world.shared.cross_spawn import run_parallel, run_sequential, run_all
from world.shared.memory import Memory
from world.shared.robotbody import RobotBody

from yarok.comm.worlds.empty_world import EmptyWorld
from yarok import Platform, PlatformMJC, PlatformHW, Injector, component, ConfigBlock

from .components.geltip.geltip import GelTip
from .components.tumble_tower.tumble_tower import TumbleTower
from .components.ur5e.ur5e import UR5e
from .components.robotiq_2f85.robotiq_2f85 import Robotiq2f85
from .components.cam.cam import Cam

from math import pi


@component(
    extends=EmptyWorld,
    components=[
        GelTip,
        GelSight2014,
        TumbleTower,
        UR5e,
        Robotiq2f85,
        Cam
    ],
    defaults={
        # 'color_map': color_map,
        # 'bs': 0.03,
        # 'sx': 0,
        # 'sy': 0,
        # 'ex': 0,
        # 'ey': 0,
        # 'pick_blocks': ['red', 'green', 'blue', 'cyan'],
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
                <texture 
                    name="brown_wood_texture"
                    type="cube" 
                    file="assets/brown_wood.png"
                    width="400"
                    height="400"/>    
                
                <mesh name="mount" file="assets/wear_mount.stl" />
                              
                <material name="wood" texture="wood_texture" specular="0.1"/>
                <material name="gray_wood" texture="wood_texture" rgba="0.6 0.4 0.2 1" specular="0.1"/>
                <material name="white_wood" texture="wood_texture" rgba="0.6 0.6 0.6 1" specular="0.1"/>
                <material name="brown_wood" texture="brown_wood_texture" rgba="0.6 0.6 0.6 1" specular="0.1"/>
                <material name="black_metal" rgba="0.05 0.05 0.05 1" specular="1.0"/>
                <material name="black_plastic" rgba="1 0.2 0.2 1" specular="0.1"/>
            </asset>
            <default>
                <!-- <default class='pp-block'>
                     <geom type="box" 
                           size="{bs} {bs} {bs}"
                           mass="0.0001"
                           material="wood"
                           zaxis="0 1 0"/>
                </default> -->
            </default>
            <worldbody>
                <!-- 
                <light directional="true" 
                    diffuse="{color_map(light, 0.1)}" 
                    specular="{color_map(light, 0.1)}" 
                    pos="1.0 1.0 5.0" 
                    dir="0 -1 -1"/>
                -->
                <body pos="{0.3 + p_cam[0]*0.1} -1.3 {0.6 + p_cam[0]*0.1}" euler="1.57 -3.14 0">
                    <cam name="cam" />
                </body> 
                
                <!-- pick blocks -->
                <!-- <for each="range(len(pick_blocks))" as="i">
                    <body>
                        <freejoint/>
                        <geom 
                            class="pp-block" 
                            pos="{0.13 + sx*0.01} {-0.135 + sy*0.01} {0.431 + i*2*bs}" 
                            rgba="{color_map(pick_blocks[i])} 1"/>
                    </body>
                </for>
                
                <for each="range(len(placed_blocks))" as="i">
                    <body>
                        <freejoint/>
                        <geom 
                            class="pp-block"
                            pos="{0.43 + ex*0.01} {-0.135 + ey*0.01} {0.131 + i*2*bs}" 
                            rgba="{color_map(pick_blocks[i])} 1"/>
                    </body>
                </for> -->
                
               <body pos="0 0 0" name="wall">
                    <geom type="box" pos="0 0.45 0" size="0.8 0.05 1.0" material="white_wood"/>
               </body>  
               
               <body pos="0 0 0" name="mount_body">
                   <geom pos="0.5 -0.5 0.5" mesh="mount" material="black_plastic"/>
               </body>  
                
               <body pos="0 0 0" name="table_base">
                    <geom type="box" pos="0.5 -0.5 0.3" size="0.2 0.2 0.01" material="black_metal"/>
                    <geom type="box" pos="0 0 0.2" size="0.8 0.8 0.1" material="white_wood"/>
               </body>  
                
                <body euler="0 0 3.14" pos="0 -0.2 0.3">
                    <ur5e name='arm'>
                       <robotiq-2f85 name="gripper" left_tip="{False}" right_tip="{False}" parent="ee_link"> 
                          <body euler="1.57079 0 -1.57079" parent="left_tip">
                            <body pos="-0.07 0 0">
                                <gelsight2014 name="left_geltip" label_color="255 0 0"/>
                            </body>
                           </body>
                        </robotiq-2f85> 
                    </ur5e> 
                </body>
            </worldbody>
        </mujoco>
    """
)
class WearNTearWorld:
    pass


def z(pos, delta):
    new_pos = pos.copy()
    new_pos[2] += delta
    return new_pos


class SlideToWear:

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
        self.memory = Memory(config['data_path'],
                             config['dataset_name'],
                             self.body,
                             self.config,
                             skip_right_sensor=True)

        # self.DOWN_DIRECTION = [3.11, 1.6e-7, 3.11]
        # Zhuo's horizontal -0.006517287775752023, 1.586883858342641, 0.02436554436467914
        self.DOWN_DIRECTION = [0.0, 1.595, 0.03]

        self.speed = config['speed']
        self.hardness = config['hardness']
        self.load_dz = config['load_dz']

        # Zhuo's points
        # p1 = [-0.44401382678115653, -0.44730236014909425, 0.0406047505555867]
        # p2, p3, p4 = [-0.4277706049994471, -0.4472889688144456, 0.062198323173402986]
        # p0 = [-0.42777175443488147, -0.4472821869932463, 0.12572038820870515]
        self.START_POS_UP = [-0.44, -0.447, 0.12]
        self.START_POS_DOWN = [-0.44, -0.447, 0.11 + self.load_dz * 0.001]
        self.END_POS_DOWN = [-0.44, -0.47, 0.11 + self.load_dz * 0.001]
        self.END_POS_UP = [-0.44, -0.47, 0.12]

    def wait_and_record(self, arm=None, gripper=None):
        def cb():
            self.memory.save()

            self.pl.wait_seconds(0.1)

            if arm is not None:
                return self.body.arm.is_at(arm)
            else:
                return self.body.gripper.is_at(gripper)

        self.pl.wait(cb)

    def on_start(self):

        self.memory.prepare()

        print('-------------')
        print(f'Collecting data for {self.hardness} !!!!')

        # Moves above start position.
        self.pl.wait(self.body.gripper.close())
        self.pl.wait(self.body.arm.move_xyz(xyz=self.START_POS_UP, xyz_angles=self.DOWN_DIRECTION))

        # Moves to start position.
        self.pl.wait(self.body.arm.move_xyz(xyz=self.START_POS_DOWN, xyz_angles=self.DOWN_DIRECTION))
        self.pl.wait_seconds(2)

        # Moves and records.
        print('Move and record...')
        q = self.body.arm.ik(xyz=self.END_POS_DOWN, xyz_angles=self.DOWN_DIRECTION)
        self.body.arm.move_q(q)
        self.wait_and_record(arm=q)
        self.pl.wait_seconds(2)
        print('End recording.')

        # moves back
        self.pl.wait(self.body.arm.move_xyz(xyz=self.END_POS_UP, xyz_angles=self.DOWN_DIRECTION))
        print('end.')
        self.pl.wait_seconds(2)

    def on_update(self):
        return True


def launch_world(**kwargs):
    Platform.create({
        'world': WearNTearWorld,
        'behaviour': SlideToWear,
        'defaults': {
            'environment': 'sim',  # update this to sim for running in sim environment.
            'behaviour': kwargs,
            'components': {
                '/': kwargs
            }
        },
        'environments': {
            'sim': {
                'platform': {
                    'class': PlatformMJC,
                    'interfaces': {
                        '/arm': {
                            # it seems that there's a bug in yarok,
                            # so I've repeated in the interface
                            'initial_position': ([-0.44, -0.44, 0.3], [0.0, 1.6, 0.0])
                        }
                    }
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


def main():
    # multi = False
    #
    # if multi:
    #     run_all(launch_world, {
    #         'hardness': ['SMOOTH', 'HARD'],
    #         'speed': range(-1, -1 + 3),
    #         'load_dz': [1, 0, -1, -2],
    #     }, parallel=4)
    # else:
    launch_world(**{
        'data_path': './data/',
        'dataset_name': 'grasp_rw',
        'hardness': 'SMOOTH',
        'speed': 0,
        'load_dz': 0,
        'p_cam': (0, 0),
        'it': 1,
    })


if __name__ == '__main__':
    main()
