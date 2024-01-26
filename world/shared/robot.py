from yarok import Injector, ConfigBlock, Platform
from yarok.comm.components.cam.cam import Cam
from yarok.comm.components.robotiq_2f85.robotiq_2f85 import Robotiq2f85
from yarok.comm.components.ur5e.ur5e import UR5e

from world.components.geltip.geltip import GelTip
from world.shared.memory import Memory

from math import pi


class Robot:

    def __init__(self, injector: Injector, config: ConfigBlock):
        self.pl: Platform = injector.get(Platform)
        self.cam: Cam = injector.get('cam')
        self.arm: UR5e = injector.get('arm')
        self.gripper: Robotiq2f85 = injector.get('gripper')
        self.l_ots: GelTip = injector.get('left_geltip')
        # self.right_geltip: GelTip = injector.get('right_geltip')

        self.memory = Memory(config['data_path'],
                             config['dataset_name'],
                             self,
                             config,
                             skip_right_sensor=True)

        self.arm.set_ws([
            [- pi, pi],  # shoulder pan
            [- pi, -pi / 2],  # shoulder lift,
            [- 2 * pi, 2 * pi],  # elbow
            [-2 * pi, 2 * pi],  # wrist 1
            [0, pi],  # wrist 2
            [- 2 * pi, 2 * pi]  # wrist 3
        ])
        self.arm.set_speed(pi / 12)

        # self.DOWN_DIRECTION = [3.11, 1.6e-7, 3.11]
        # Zhuo's horizontal -0.006517287775752023, 1.586883858342641, 0.02436554436467914
        self.TCP_DIRECTION = [3.140, 1.6, -3.114]  # pointing forward / y direction

    def move_to(self, position, record=False):
        if record:
            q = self.arm.ik(xyz=position, xyz_angles=self.TCP_DIRECTION)
            self.arm.move_q(q)
            self.wait_and_record(arm=q)
        else:
            self.pl.wait(self.arm.move_xyz(xyz=position, xyz_angles=self.TCP_DIRECTION))

    def wait_and_record(self, arm=None, gripper=None):
        def cb():
            self.memory.save()

            self.pl.wait_seconds(0.3)

            if arm is not None:
                return self.arm.is_at(arm)
            else:
                return self.gripper.is_at(gripper)

        self.pl.wait(cb)
