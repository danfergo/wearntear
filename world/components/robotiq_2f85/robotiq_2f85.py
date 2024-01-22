from yarok import component, interface
from yarok.platforms.mjc import InterfaceMJC

from yarok.comm.utils.PID import PID
from math import pi

from .robotiq_gripper import RobotiqGripper


@interface()
class Robotiq2f85InterfaceHW:

    def __init__(self):
        self.gripper = RobotiqGripper()
        self.gripper.connect("10.40.101.100", 63352, socket_timeout=10)

        # if not gripper.is_active():
        # print('gripper not active, activating ...')
        self.gripper.activate()
        # print('gripper is active now.')
        # else:
        self.min_p = self.gripper._min_position
        self.max_p = self.gripper._max_position

    def _percentage2range(self, p):
        return round(p * (self.max_p - self.min_p) + self.min_p)

    def is_ready(self):
        # required by PlatformHW/interfaceHW,
        # should return true when the component is initialized.
        # it can be used to activate components asyncronously.
        print('gripper is ready.')
        return True

    def at(self):
        return self.gripper.get_current_position()

    def is_at(self, pos):
        current_p = self.gripper.get_current_position()
        return abs(pos - current_p) <= 1

    def move(self, pos):
        self.gripper.move(pos)
        return lambda: self.is_at(pos)

    def close(self, percentage=1.0):
        self.move(self._percentage2range(percentage))

    def step(self):
        pass


@interface()
class Robotiq2f85MJCInterfaceMJC:

    def __init__(self, interface: InterfaceMJC):
        self.interface = interface
        self.target_q = 0
        self.q = 0
        # P = 0.001
        # I = 0
        # D = 0
        # self.gear = 100
        # self.PIDs = [PID(P, I, D) for a in interface.actuators]
        # self.move(0.9)
        #
        # self.last_query_position = None
        # self.at_target_hit_counter = 0

    def at(self):
        return self.q

    def is_at(self, q):
        return abs(self.q - self.target_q) <= 1

        # def similar_q(a1, a2):
        #     return abs(a1 - a2) < 0.02
        #
        # at = self.at()
        #
        # if self.last_query_position is not None:
        #     if similar_q(a, self.last_query_position):
        #         self.at_target_hit_counter += 1
        #
        #         if self.at_target_hit_counter > 30:
        #             self.last_query_position = None
        #             self.at_target_hit_counter = 0
        #             return True
        #
        #     else:
        #         self.last_query_position = None
        #         self.at_target_hit_counter = 0
        #
        # self.last_query_position = at
        # return False
        pass

    def move(self, q):
        self.target_q = q
        return lambda: self.is_at(q)
        # k = 3.14
        # q = [-1 * a * k * self.gear, a * k * self.gear]
        # [self.PIDs[qa].setTarget(q[qa] * self.gear) for qa in range(2)]
        # pass

    def close(self, percentage=1.0):
        self.move(round(percentage * 255))

    def step(self):
        self.q = round(self.interface.sensordata()[0] * (255 / 0.8))
        # scale =  255 / (0.8 - 0.002)
        # self.q =
        # print(self.q, self.target_q )
        # self.q = data
        # print(self.q)
        # [self.PIDs[a].update(data[a]) for a in range(2)]
        self.interface.set_ctrl(0, self.target_q)
        # print('')

        # m  = ['finger0_joint1_vel', 'finger1_joint1_vel']
        # [self.interface.set_ctrl(m[a], abs(self.PIDs[a].SetPoint - data[a])) for a in range(2)]


@component(
    tag='robotiq-2f85',
    defaults={
        'interface_mjc': Robotiq2f85MJCInterfaceMJC,
        'interface_hw': Robotiq2f85InterfaceHW,
        'left_tip': True,
        'right_tip': True
    },
    template_path='2f85.xml',
)
class Robotiq2f85:

    def __init__(self):
        pass

    def is_open(self):
        pass

    def is_closed(self):
        pass

    def at(self):
        pass

    def is_at(self, a):
        pass

    def move(self, a):
        pass

    def close(self, percentage=1.0):
        pass

    def open(self):
        self.close(0)
