from world.shared.robot import Robot

from yarok import Platform, PlatformMJC, PlatformHW, Injector, component, ConfigBlock

from world.world import shared_config



class SlideToWear:

    def __init__(self, injector: Injector, config: ConfigBlock):
        self.robot = Robot(injector, config)
        self.config = config

        self.pl: Platform = injector.get(Platform)

        self.speed = config['speed']
        self.hardness = config['hardness']
        self.load_dz = config['load_dz']
        self.load_dy = config['load_dy']

        self.START_POS_UP = [-0.468, -0.556 + self.load_dy * 0.001, -0.0168]
        self.START_POS_DOWN = [-0.468, -0.556 + self.load_dy * 0.001, -0.0168 + self.load_dz * 0.001]
        self.END_POS_DOWN = [-0.448, -0.556 + self.load_dy * 0.001, -0.0168 + self.load_dz * 0.001]
        self.END_POS_UP = [-0.448, -0.556 + self.load_dy * 0.001, -0.0168]

    def on_start(self):
        self.robot.memory.prepare()
        self.pl.wait(self.robot.gripper.close())
        self.pl.wait_seconds(100)

        print('-------------')
        print(f'Collecting data for {self.hardness} !!!!')
        # xyz, _ = self.body.arm.at_xyz()
        # print('xyz',xyz)
        # xyz = [xyz[0],xyz[1],xyz[2]-0.01]
        # self.pl.wait(self.body.arm.move_xyz(xyz=xyz, xyz_angles=self.DOWN_DIRECTION))
        # print(self.body.arm.at_xyz())

        # Moves above start position.
        # self.pl.wait(self.body.gripper.close())
        self.robot.move_to(self.START_POS_UP)
        print("moved 0")

        # Moves to start position.
        self.robot.move_to(self.START_POS_DOWN)
        self.pl.wait_seconds(2)
        print("moved 1")

        # Moves and records.
        print('Move and record...')
        self.robot.move_to(self.START_POS_DOWN)
        self.robot.move_to(self.END_POS_DOWN, record=True)
        self.pl.wait_seconds(2)
        print('End recording.')
        print("moved 2")

        # moves back
        self.robot.move_to(self.END_POS_UP)
        print('end.')
        self.pl.wait_seconds(2)
        print("moved 3")


def run(**kwargs):
    Platform.create(shared_config(SlideToWear, 'sim', **kwargs)).run()


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
    # speeds: slow, medium, fast
    run(**{
        'dataset_name': 'wear',
        'data_path': './data/',
        'hardness': 'hard',
        'speed': 1,
        'load_dz': -0.8,
        'load_dy': +2,
        'p_cam': (0, 0),
        'it': 9,
    })


if __name__ == '__main__':
    main()
