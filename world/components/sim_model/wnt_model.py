#!/usr/bin/env python
import math
import queue
import time
from threading import Thread

import cv2
import numpy as np

# import cv2
# import numpy as np
import open3d as o3d

from .utils.camera import get_camera_matrix, depth2cloud
from .utils.maths import normalize_vectors, gkern2, dot_vectors, partial_derivative, normals, proj_vectors, \
    norm_vectors

from .utils.vis_img import to_normed_rgb, show_normalized_img
from .utils.vis_mesh import show_field
import imutils
import matplotlib.pyplot as plt

""" 
    GelSight Simulation
"""


def radius_from_area(area):
    radius = math.sqrt(area / math.pi)
    return radius


def distort_image(image, displacement_map_x, displacement_map_y):
    # Create a grid of coordinates
    rows, cols = image.shape[:2]
    map_x = np.arange(0, cols, 1)
    map_y = np.arange(0, rows, 1)
    map_x, map_y = np.meshgrid(map_x, map_y)

    # Apply displacement maps
    distorted_map_x = map_x + displacement_map_x
    distorted_map_y = map_y + displacement_map_y

    # Remap the image using the distorted maps
    distorted_image = cv2.remap(image, distorted_map_x.astype(np.float32), distorted_map_y.astype(np.float32),
                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return distorted_image


def trapezoid(Ct_1, Ct, Rt_1, Rt, Ht_1, H_t, shape=(480, 640)):
    # Ct -> Ct_1 axis
    v = np.array([Ct_1[0] - Ct[0], Ct_1[1] - Ct[1]])

    # Perpendicular to v
    vp = np.array([-v[1], v[0]])
    vp = (vp / np.linalg.norm(vp)) if np.linalg.norm(vp) > 0 else vp

    # ...
    p1 = Ct_1 + Rt_1 * vp
    p2 = Ct_1 + Rt_1 * vp
    p3 = Ct_1 + Rt * vp
    p4 = Ct_1 - Rt_1 * vp

    mask = np.zeros(shape)
    mask = cv2.fillPoly(mask, pts=[p1, p2, p3, p4], color=255)
    return mask


def generate_dot_grid(grid_size, dot_size, spacing, image_shape=(480, 640)):
    # Unpack grid_size tuple
    grid_size_horizontal, grid_size_vertical = grid_size

    # Calculate the size of the image
    image_size_horizontal = grid_size_horizontal * (dot_size + spacing) - spacing
    image_size_vertical = grid_size_vertical * (dot_size + spacing) - spacing

    # Create an empty image with white background
    image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.float32)

    # Calculate the starting position to center the grid
    start_position_x = (image_shape[1] - image_size_horizontal) // 2
    start_position_y = (image_shape[0] - image_size_vertical) // 2

    # Draw dots on the image
    for i in range(grid_size_vertical):
        for j in range(grid_size_horizontal):
            x = start_position_x + j * (dot_size + spacing)
            y = start_position_y + i * (dot_size + spacing)
            cv2.circle(image, (x, y), dot_size // 2, (1, 1, 1), -1)

    return image
    # # Display the image
    # cv2.imshow('Dot Grid', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# Example usage:


def match_contours(contour1, contour2):
    # Match contours using cv2.matchShapes
    similarity_score = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)

    # Print the similarity score (lower score indicates a better match)
    print(f"Similarity Score: {similarity_score}")

    # Extract points from the matched contours
    points1 = np.squeeze(contour1)
    points2 = np.squeeze(contour2)

    # Ensure both contours have the same number of points
    min_points = min(len(points1), len(points2))
    points1 = points1[:min_points]
    points2 = points2[:min_points]

    # Calculate displacement vectors
    displacement_vectors = points2 - points1

    return displacement_vectors


class WNTSimulationModel:

    def __init__(self, **config):
        self.default_ks = 0.15
        self.default_kd = 0.5
        self.default_alpha = 100
        self.ia = config['ia'] or 0.8
        self.fov = config['fov'] or 90

        self.lights = config['light_sources']
        self.rectify_fields = config['rectify_fields']

        self.ref_depth = config['background_depth']
        self.cam_matrix = get_camera_matrix(self.ref_depth.shape[::-1], self.fov)
        self.internal_shadow = config['internal_shadow'] if 'internal_shadow' in config else 0.15

        self.background_img = config['background_img']
        self.s_ref = depth2cloud(self.cam_matrix, self.ref_depth)  # config['cloud_map']
        self.s_ref_n = normals(self.s_ref)

        self.apply_elastic_deformation = config['elastic_deformation'] if 'elastic_deformation' in config else False
        self.torn_depth = self.ref_depth.copy()

        # pre compute & defaults
        self.ambient = config['background_img']

        for light in self.lights:
            light['ks'] = light['ks'] if 'ks' in light else self.default_ks
            light['kd'] = light['kd'] if 'kd' in light else self.default_kd
            light['alpha'] = light['alpha'] if 'alpha' in light else self.default_alpha
            light['color_map'] = np.tile(np.array(np.array(light['color']) / 255.0)
                                         .reshape((1, 1, 3)), self.s_ref.shape[0:2] + (1,))

        self.texture_sigma = config['texture_sigma'] or 0.00001
        self.membrane_texture = self.gauss_texture((480, 640))[:, :, 0]

        self.t = config['t'] if 't' in config else 3
        self.sigma = config['sigma'] if 'sigma' in config else 7
        self.kernel_size = config['sigma'] if 'sigma' in config else 21
        # self.max_depth = self.min_depth + self.elastomer_thickness

        self.wear_mask = np.zeros((480, 640), dtype=np.float64)
        self.prev_contacts = []
        self.prev_contours = []
        self.tear_mask = np.zeros((480, 640), dtype=np.float64)

        # self.markers_mask = cv2.filter2D(generate_dot_grid(grid_size=(20, 15), dot_size=8, spacing=22, image_shape=(480, 640)), -1, gkern2(5, 0.5))
        self.markers_mask = generate_dot_grid(grid_size=(20, 15), dot_size=8, spacing=22, image_shape=(480, 640))
        self.markers_color = np.zeros((480, 640, 3), np.float64)

    @staticmethod
    def load_assets(assets_path, input_res, output_res, lf_method, n_light_sources):
        prefix = str(input_res[1]) + 'x' + str(input_res[0])

        # cloud = np.load(assets_path + '/' + prefix + '_ref_cloud.npy')
        # cloud = cloud.reshape((input_res[1], input_res[0], 3))
        # cloud = cv2.resize(cloud, output_res)

        # normals = np.load(assets_path + '/' + prefix + '_surface_normals.npy')
        # normals = normals.reshape((input_res[1], input_res[0], 3))
        # normals = cv2.resize(normals, output_res)
        light_fields = [
            # normalize_vectors(
            cv2.resize(
                cv2.GaussianBlur(
                    # cv2.resize(
                    np.load(assets_path + '/' + lf_method + '_' + prefix + '_field_' + str(l) + '.npy'),
                    # (60, 60),
                    # interpolation=cv2.INTER_LINEAR),
                    (25, 25), 0),
                output_res[::-1], interpolation=cv2.INTER_LINEAR)
            # )
            for l in range(n_light_sources)
        ]
        # normals,
        return light_fields

    # def protrusion_map(self, original, not_in_touch):
    #     protrusion_map = np.copy(original)
    #     protrusion_map[not_in_touch >= self.max_depth] = self.max_depth
    #     return protrusion_map

    # def segments(self, depth_map):
    #     not_in_touch = np.copy(depth_map)
    #     not_in_touch[not_in_touch < self.max_depth] = 0.0
    #     not_in_touch[not_in_touch >= self.max_depth] = 1.0
    #
    #     in_touch = 1 - not_in_touch
    #
    #     return not_in_touch, in_touch

    # def internal_shadow(self, elastomer_depth):
    #     elastomer_depth_inv = self.max_depth - elastomer_depth
    #     elastomer_depth_inv = np.interp(elastomer_depth_inv, (0, self.elastomer_thickness), (0.0, 1.0))
    #     return elastomer_depth_inv

    def gauss_texture(self, shape):
        row, col = shape
        mean = 0
        gauss = np.random.normal(mean, self.texture_sigma, (row, col))
        gauss = gauss.reshape(row, col)
        return np.stack([gauss, gauss, gauss], axis=2)

    def elastic_deformation(self, protrusion_depth):
        fat_gauss_size = 95
        thin_gauss_size = 95
        thin_gauss_pad = (fat_gauss_size - thin_gauss_size) // 2
        # - gkern2(gauss2_size, 12)
        fat_gauss_kernel = gkern2(55, 5)
        # thin_gauss_kernel = np.pad(gkern2(thin_gauss_size, 21), thin_gauss_pad)
        # dog_kernel = fat_gauss_kernel - thin_gauss_kernel
        # show_panel([fat_gauss_kernel, thin_gauss_kernel])

        return cv2.filter2D(protrusion_depth, -1, fat_gauss_kernel)

        # kernel = gkern2(self.kernel_size, self.sigma)
        # deformation = protrusion_depth
        #
        # deformation2 = protrusion_depth
        # kernel2 = gkern2(52, 9)
        #
        # for i in range(self.t):
        #     deformation_ = cv2.filter2D(deformation, -1, kernel)
        #     r = np.max(protrusion_depth) / np.max(deformation_) if np.max(deformation_) > 0 else 1
        #     deformation = np.maximum(r * deformation_, protrusion_depth)
        #
        #     deformation2_ = cv2.filter2D(deformation2, -1, kernel2)
        #     r = np.max(protrusion_depth) / np.max(deformation2_) if np.max(deformation2_) > 0 else 1
        #     deformation2 = np.maximum(r * deformation2_, protrusion_depth)
        #
        # for i in range(self.t):
        #     deformation_ = cv2.filter2D(deformation2, -1, kernel)
        #     r = np.max(protrusion_depth) / np.max(deformation_) if np.max(deformation_) > 0 else 1
        #     deformation2 = np.maximum(r * deformation_, protrusion_depth)
        #
        #
        # deformation_x = 2 * deformation  # - deformation2
        #
        # return deformation_x / 2
        # # return np.stack([deformation_x, deformation_x, deformation_x], axis=2) / 3

    def _spec_diff(self, lm_data, v, n, s):
        imd = lm_data['id']
        ims = lm_data['is']
        alpha = lm_data['alpha']

        lm = - lm_data['field']  # points in the direction of the light source,
        color = lm_data['color_map']

        if self.rectify_fields:
            lm = normalize_vectors(lm - proj_vectors(lm, self.s_ref_n))

        # - (self.s_ref - s)
        # print('-->', dot_vectors(lm, self.s_ref_n)[100, 100])
        # i.e. p(s) -> light source

        # show_field(cloud_map=s, field=lm, field_color='red', subsample=99)

        # Shared calculations
        lm_n = dot_vectors(lm, n)
        lm_n[lm_n < 0.0] = 0.0
        Rm = 2.0 * lm_n[:, :, np.newaxis] * n - lm

        # diffuse component
        diffuse_l = lm_n * imd

        # specular component
        spec_l = (dot_vectors(Rm, v) ** alpha) * ims

        return (diffuse_l + spec_l)[:, :, np.newaxis] * color

    def _parallel_spec_diff(self, args, q):
        q.put(self._spec_diff(*args))

    def track_motion(self, protrusion_map):
        high_protrusion_map = protrusion_map.copy()
        high_protrusion_map[high_protrusion_map < 0.000001] = 0.0
        high_protrusion_map[high_protrusion_map >= 0.00001] = 1.0
        # high_protrusion_map[high_protrusion_map < 0.001] = 0.0

        # cv2.imshow('protrusion map', to_normed_rgb(high_protrusion_map))
        # cv2.waitKey(1)

        if high_protrusion_map.max() > 0:
            gray = (high_protrusion_map * 255.0 / high_protrusion_map.max()).astype(np.uint8)
            cnts = imutils.grab_contours(cv2.findContours(
                gray,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            ))
            cnts = [c for c in cnts if cv2.contourArea(c) > 0]

            if len(cnts) > 0:
                M = cv2.moments(cnts[0])
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                contour_mask = cv2.drawContours(
                    np.zeros((480, 640), np.uint8),
                    [cnts[0]],
                    -1,
                    255,
                    cv2.FILLED
                )

                pressure = np.sum(contour_mask * protrusion_map) / M["m00"]

                prev_contacts = self.prev_contacts
                current_contacts = (cX, cY, pressure, M["m00"], radius_from_area(M["m00"]))

                if len(self.prev_contacts) > 0:
                    pcX, pcY, _, __, __ = self.prev_contacts[0]
                    # kernel_size = 15
                    # contour_mask = cv2.filter2D(contour_mask, -1, gkern2(55, 5))
                    # movement_mask = np.stack([contour_mask, contour_mask], axis=2) * np.array([cX - pcX, cY - pcY])
                    # movement_mask = np.sign(movement_mask) * (np.log(np.abs(movement_mask) + 1) - 1)
                    # background_img = self.background_img * (
                    #         1.0 - self.markers_mask) + self.markers_color * self.markers_mask
                    # # distorted_image = distort_image(background_img, movement_mask[:, :, 0], movement_mask[:, :, 1])
                    # # cv2.imshow('mask', contour_mask)
                    # # cv2.imshow('distorted', distorted_image)
                    # # cv2.waitKey(1)

                self.prev_contacts = [current_contacts]
                self.prev_contours = cnts

                # print('p > ', pressure, (cX, cY, pressure), self.prev_contacts[0])

                return [current_contacts, prev_contacts[0]] if len(prev_contacts) > 0 else []
        else:
            self.prev_contacts = []
            self.prev_contours = []
        return []

    def _update_wear_mask(self, protrusion_map, contacts_movement):
        distance = 0 if len(contacts_movement) < 2 else math.dist(contacts_movement[0][0:2], contacts_movement[1][0:2])
        # print(distance)
        Q = 0.5 * protrusion_map * distance
        # self.wear_mask += Q // TODO bug
        self.wear_mask += cv2.filter2D(Q, -1, gkern2(55, 5))
        self.wear_mask = np.clip(self.wear_mask, 0, 1.0)
        # self.tear_mask = np.maximum(self.tear_mask, Q)

    def compute_tear_volume(self, protrusion_map, contacts_movement):
        # high_protrusion_map = protrusion_map.copy()
        # high_protrusion_map[high_protrusion_map < 0.001] = 0.0
        TQ_mask = np.zeros((480, 640))

        # cv2.imshow('x >> x', self.prev_contacts)
        # cv2.waitKey(-1)

        if len(contacts_movement) > 0:
            (pcX, pcY, p1, a1, r1), (cX, cY, pressure, a2, r2) = contacts_movement
            if pressure > 0.15:  # round(r2)
                TQ_mask = cv2.line(TQ_mask, (pcX, pcY), (cX, cY), 0.003, 15)
                TQ_mask = cv2.filter2D(TQ_mask, -1, gkern2(15, 1))
                TQ_mask = TQ_mask.astype(np.float32)

        return TQ_mask
        # self.tear_mask = np.maximum(self.tear_mask, TQ_mask)

    def _compute_spec_diff(self, *args, parallel=True):
        if parallel:
            q = queue.Queue()
            threads = [Thread(target=self._parallel_spec_diff, args=((lm, *args), q)) for lm in self.lights]
            [t.start() for t in threads]
            [t.join() for t in threads]
            return np.sum([q.get() for _ in range(len(self.lights))], axis=0)
        else:
            return np.sum([self._spec_diff(lm, *args) for lm in self.lights], axis=0)

    def generate(self, depth, rgb):
        rgb = rgb.astype(np.float64) / 255.0

        # Calculate the protrusion_map
        m_depth = np.minimum(self.ref_depth, depth)
        protrusion_map = self.ref_depth - m_depth

        # Track contacts movement update wear and tear
        contacts_movement = self.track_motion(protrusion_map)
        self._update_wear_mask(protrusion_map, contacts_movement)
        tear_volume = self.compute_tear_volume(protrusion_map, contacts_movement)

        # Tearing the membrane
        self.ref_depth = np.minimum(self.ref_depth, self.ref_depth - tear_volume)

        # Wear membrane
        wm3 = np.stack([self.wear_mask, self.wear_mask, self.wear_mask], axis=2)

        # Elastic deformation (as in the GelTip Sim paper, submitted to RSS)
        if self.apply_elastic_deformation:
            elastic_deformation = cv2.filter2D(protrusion_map, -1, gkern2(55, 5))
            m_depth = np.minimum(m_depth, self.ref_depth - elastic_deformation).astype(np.float32)

        # Tear membrane
        m_depth = m_depth - self.membrane_texture.astype(np.float32)

        # Surface point-cloud
        s = depth2cloud(self.cam_matrix, m_depth)
        ss = s.reshape((480 * 640, 3)).tolist()

        # Optical Rays = s - 0
        optical_rays = normalize_vectors(s)

        # Compute illumination with Phong's model

        # Illumination vectors (n, v) calculations
        n = - normals(s)
        v = - optical_rays
        spec_diffuse = self._compute_spec_diff(v, n, s)
        reduction_coeff = 0.2

        # Ambient_component = self.background_img * (self.ia - shadow_factor)[:, :, np.newaxis]
        background_img = self.background_img \
            if self.markers_mask is None else \
            self.background_img * (1.0 - self.markers_mask) + self.markers_color * self.markers_mask
        worn_bkg_img = wm3 * rgb + (1.0 - wm3) * background_img
        ambient_component = worn_bkg_img.astype(np.float64) * self.ia

        # Reduces specular reflections in worn areas.
        I = ambient_component + (1.0 - (reduction_coeff * wm3)) * spec_diffuse

        # Cuts membrane in torn areas
        # tm1 = self.tear_mask.copy()
        # tm1[tm1 > 0.002] = 1.0
        # tm3 = np.stack([tm1, tm1, tm1], axis=2)

        # I = ((1.0 - tm3) * I) + (tm3 * (rgb / 255.0))

        return np.clip(I * 255.0, 0, 255).astype(np.uint8)
