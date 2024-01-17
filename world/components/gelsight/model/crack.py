import math
from random import random, randint

import numpy as np
import cv2


def inside(frame, pt):
    s = frame.shape
    return 0 <= pt[0] < s[0] and 0 <= pt[1] < s[1]


def normalize(v):
    # print(v)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def roundc(v):
    return np.array([round(x) for x in list(v)])


propagation_vectors = [
    np.array(x)
    for x in
    [(1, 0),
     (-1, 0),
     (0, 1),
     (0, -1),
     (-1, -1),
     (-1, 1),
     (1, -1),
     (1, 1)]
]


# def gen_crack():
#     crack = np.zeros((480, 640))
#     crack[:, :] = 1.0
#
#     crack_pt = np.array((240, 320))
#
#     propagation_pt = crack_pt
#
#     while True:
#         if propagation_pt is not None and inside(crack, propagation_pt):
#             i = randint(0, len(propagation_vectors) - 1)
#             crack[tuple(propagation_pt)] = 0.0
#             propagation_pt = propagation_pt + propagation_vectors[i]
#         else:
#             break
#             # crack[tuple(propagation_pt)] = 0.0
#
#         # print(propagation_pt)
#
#     #     print(propagation_pt)
#     return crack


def create_crack(crack_map, crack_pt, global_propagation_vector):
    propagation_pt = crack_pt
    prev_propagation_vector = np.array((0, 0))

    # theta = random() * 2 * math.pi
    # global_propagation_vector = (math.sin(theta), math.cos(theta))
    gamma = 0.5

    for _ in range(100):
        if propagation_pt is not None and inside(crack_map, propagation_pt):
            i = randint(0, len(propagation_vectors) - 1)
            d = np.linalg.norm(propagation_pt - crack_pt)

            thickness = max(1, round(5 - min(5, 0.02 * d)))  # round(2 * (1.0 if d < 1.0 else 1 / d))
            # print(1 / d)

            crack_map = cv2.circle(crack_map, tuple(propagation_pt[::-1]), thickness, 0.0, -1)
            # crack[] = 0.0

            propagation_vector = roundc(normalize(gamma * propagation_vectors[i] + global_propagation_vector))
            propagation_pt = propagation_pt + propagation_vector
            prev_propagation_vector = propagation_vector

        else:
            break
    return crack_map


def gen_crack_map():
    crack = np.zeros((480, 640))
    crack[:, :] = 1.0

    crack_pt = np.array(240 + (randint(0, 10), 320 + randint(0, 10)))

    # propagation_pt = crack_pt
    # prev_propagation_vector = np.array((0, 0))

    theta = random() * 2 * math.pi
    global_propagation_vector = (math.sin(theta), math.cos(theta))
    global_propagation_vector2 = (math.sin(theta + math.pi), math.cos(theta + math.pi))
    gamma = 0.5

    crack = create_crack(crack, crack_pt, global_propagation_vector)
    crack = create_crack(crack, crack_pt, global_propagation_vector2)

    # while True:
    #     if propagation_pt is not None and inside(crack, propagation_pt):
    #         i = randint(0, len(propagation_vectors) - 1)
    #         d = np.linalg.norm(propagation_pt - crack_pt)
    #
    #         thickness = max(1, round(5 - min(5, 0.02 * d))) # round(2 * (1.0 if d < 1.0 else 1 / d))
    #         print(1 / d)
    #
    #         crack = cv2.circle(crack, tuple(propagation_pt[::-1]), thickness, 0.0, -1)
    #         # crack[] = 0.0
    #
    #         propagation_vector = roundc(normalize(gamma * propagation_vectors[i] + global_propagation_vector))
    #         propagation_pt = propagation_pt + propagation_vector
    #         prev_propagation_vector = propagation_vector
    #
    #     else:
    #         break
    #         # crack[tuple(propagation_pt)] = 0.0

    # print(propagation_pt)

    #     print(propagation_pt)
    return crack

# for i in range(50):
#     crack = gen_crack_map()
#     cv2.imshow('crack', cv2.resize(crack, (640, 480)))
#     cv2.waitKey(-1)
