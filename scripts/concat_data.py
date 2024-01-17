import os
import subprocess
from os.path import dirname

import cv2
import numpy as np


def gen_frame(base_path, frame_name):
    frames = [cv2.imread(os.path.join(base_path, s, frame_name)) for s in ['l', 'c', 'r']]
    # print(frames)
    concat_frame = np.concatenate(frames, axis=1)
    legend = np.zeros((50, concat_frame.shape[1]), dtype=np.uint8)

    legend = cv2.putText(legend, 'Left sensor', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    legend = cv2.putText(legend, 'Visual camera', (640 + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    legend = cv2.putText(legend, 'Right sensor', (1280 + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    legend_rgb = np.stack([legend, legend, legend], axis=2)

    concat_frame = np.concatenate([concat_frame, legend_rgb], axis=0)
    # cv2.imshow('frame', concat_frame)
    # cv2.waitKey(1)
    return concat_frame, frame_name


def main():
    data_path = dirname(__file__) + '/../data/wear/rec_00001'

    # sensors = [s for s in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, s))]

    frames = [f for f in os.listdir(os.path.join(data_path, 'c'))]
    concat_frames = [gen_frame(data_path, f) for f in frames]

    concat_path = os.path.join(data_path, '_concat')

    if not os.path.exists(concat_path):
        os.mkdir(concat_path)

    [cv2.imwrite(os.path.join(concat_path, frame_name), frame) for frame, frame_name in concat_frames]

    result = subprocess.run(f"cd {concat_path} && ffmpeg -f image2 -i frame_%005d.jpg out.gif", shell=True)


if __name__ == '__main__':
    main()
