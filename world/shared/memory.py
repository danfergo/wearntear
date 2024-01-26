import numpy as np

import os
import cv2
import shutil
import yaml


class Memory:
    def __init__(self, base_path, dataset_name, body, config, skip_right_sensor=False, resolution=(640, 480)):
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.body = body
        self.i = 0
        self.ps = []
        self.config = config
        self.skip_right_sensor = skip_right_sensor
        self.resolution = resolution
        self.min_side = min(*self.resolution)
        self.side_crop = (max(*self.resolution) - self.min_side) // 2

    def prepare(self):
        self.rec = 'rec_' + str(self.config['it']).zfill(5)
        # self.dataset_path = f'data/{self.dataset_name}'
        self.dataset_path = f'{self.base_path}/{self.dataset_name}'
        self.data_path = f'{self.dataset_path}/{self.rec}/'

        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        if os.path.exists(self.data_path):
            shutil.rmtree(self.data_path)
        os.mkdir(self.data_path)
        os.mkdir(f'{self.data_path}/c')
        os.mkdir(f'{self.data_path}/l')

        if not self.skip_right_sensor:
            os.mkdir(f'{self.data_path}/r')

        with open(f'{self.data_path}/config.yml', 'w') as outfile:
            yaml.dump(self.config, outfile)

    def prepare_frame(self, frame):
        # print('frame.shape',frame.shape)
        frame = cv2.resize(frame, self.resolution)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = frame[:, self.side_crop:self.side_crop + self.min_side, :]
        # frame = cv2.resize(frame, (self.min_side, self.min_side))
        return frame

    def save(self):
        # visual camera
        cam_frame = self.prepare_frame(self.body.cam.read())
        # cam_frame = cv2.rotate(cam_frame, cv2.ROTATE_180)
        cv2.imwrite(f'{self.data_path}/c/frame_{str(self.i).zfill(5)}.jpg', cam_frame)

        # left touch sensor
        left_touch_frame = self.prepare_frame(self.body.l_ots.read())
        cv2.imwrite(f'{self.data_path}/l/frame_{str(self.i).zfill(5)}.jpg', left_touch_frame)

        # right touch sensor
        if not self.skip_right_sensor:
            right_touch_frame = self.prepare_frame(self.body.right_geltip.read())
            cv2.imwrite(f'{self.data_path}/r/frame_{str(self.i).zfill(5)}.jpg', right_touch_frame)

        # arm position(s)
        q_xyz = self.body.arm.at_xyz()
        p = np.array(q_xyz[0].tolist() + q_xyz[1].tolist() + [self.body.gripper.at()])
        self.ps = np.concatenate([self.ps, p.T], axis=0) if self.ps is not None else p.T

        with open(f'{self.data_path}/p.npy', 'wb') as f:
            np.save(f, self.ps)

        self.i += 1
