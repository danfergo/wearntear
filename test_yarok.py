from yarok import Platform, PlatformMJC, PlatformHW, component, ConfigBlock, Injector

from yarok.comm.worlds.empty_world import EmptyWorld
from yarok.comm.components.robotiq_2f85.robotiq_2f85 import Robotiq2f85
# from yarok.comm.components.geltip.geltip import GelTip
from yarok.comm.components.ur5e.ur5e import UR5e
from yarok.comm.components.cam.cam import Cam

from yarok.comm.plugins.cv2_inspector import Cv2Inspector
# from yarok.comm.plugins.cv2_waitkey import Cv2WaitKey

from math import pi

import cv2


@component(
    tag="template",
    defaults={

    },
    template="""
        <mujoco>
        <worldbody>
        
        <body if="i > 0">
            <geom type="box" size="0.1 0.1 0.1"/>
        
            <template name="somename" x="${x - 1}"> </template>
        </body>
        
        </worldbody>
        </mujoco>
    """
)
class Template:
    pass


@component(
    extends=EmptyWorld,
    components=[
        UR5e,
        Robotiq2f85,
        Template,
        Cam
    ],
    # language=xml
    template="""
        <mujoco>
            <default>
                <default class='marker'>
                    <geom type="box" 
                        size="0.02 0.02 0.001" 
                        conaffinity='32' 
                        contype='32'/>
                </default>
            </default>
            <worldbody>


                
                <template name="template" x="${10}">
                
                </template>
                
                
            </worldbody>        
        </mujoco>
    """
)
class GraspRopeWorld:
    pass


class GraspRoleBehaviour:

    def __init__(self, pl: Platform, injector: Injector):
        # self.arm: UR5e = injector.get('arm')
        # self.gripper: Robotiq2f85 = injector.get('gripper')
        self.pl = pl

    def on_update(self):
        return True
        # # self.pl.wait_seconds(60)
        # # print('------------------------------------------------------------------------------------------->')
        #
        # self.gripper.close(0.5)
        #
        # self.pl.wait(
        #     self.arm.move_xyz(xyz=[0.45, 0.5, 0.2], xyz_angles=[3.11, -1.6e-7, -pi / 2])
        # )
        #
        # self.pl.wait(
        #     self.arm.move_xyz(xyz=[0.45, 0.5, 0.17], xyz_angles=[3.11, -1.6e-7, -pi / 2])
        # )
        # self.pl.wait_seconds(2)
        # self.pl.wait(
        #     self.gripper.close(0.75)
        # )
        # self.pl.wait_seconds(10)
        # print('closed gripper')
        #
        # self.pl.wait(
        #     self.arm.move_xyz(xyz=[0.45, 0.5, 0.4], xyz_angles=[3.11, -1.6e-7, -pi / 2])
        # )
        #
        # self.pl.wait(
        #     self.arm.move_xyz(xyz=[0.45, 0.2, 0.4], xyz_angles=[3.11, -1.6e-7, -pi / 2])
        # )
        # self.pl.wait(
        #     self.arm.move_xyz(xyz=[0.45, 0.2, 0.25], xyz_angles=[3.11, -1.6e-7, -pi / 2])
        # )
        # self.pl.wait(
        #     self.gripper.close(0.5)
        # )
        #
        # print('ended.')
        # self.pl.wait_seconds(100)


conf = {
    'world': GraspRopeWorld,
    'behaviour': GraspRoleBehaviour,
    'defaults': {
        'environment': 'sim',
        'behaviour': {
        },
        'components': {
            '/gripper': {
                'left_tip': False,
                'right_tip': False,
            },
            '/right_geltip': {
                'label_color': '255 0 0'
            },
            '/left_geltip': {
                'label_color': '0 255 0'
            },
        },
    },
    'environments': {
        'sim': {
            'platform': {
                'class': PlatformMJC,
                'width': 800,
                'height': 600
            },
            'plugins': [
                (Cv2Inspector, {})
            ]
        }
    },
}

if __name__ == '__main__':
    Platform.create(conf).run()
