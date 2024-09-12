import numpy as np
import matplotlib.pyplot as plt


def generate_radial_field(center, heightmap, mask_shape):
    # Create an empty numpy array to store the displacement vectors
    radial_field = np.zeros(mask_shape + (2,))

    # Generate the grid of coordinates for the mask
    y, x = np.indices(mask_shape)

    # Calculate displacement vectors from the center to each point
    # condition_x = (x - center[0]) != 0
    # condition_y = (y - center[1]) != 0

    radial_field[:, :, 0] = 1 / (x - center[0] + np.finfo(float).eps)
    radial_field[:, :, 1] = 1 / (y - center[1] + np.finfo(float).eps)

    # Compute the distance from the center to each point
    # distance = np.sqrt(radial_field[:, :, 0] ** 2 + radial_field[:, :, 1] ** 2)

    # Set values outside the mask to zero
    radial_field[heightmap == 0] = 0

    return radial_field


def generate_elliptical_heightmap_mask(shape, a, b):
    # Generate a grid of coordinates
    y, x = np.indices(shape)

    # Calculate the distance from the center
    center_x, center_y = shape[1] // 2, shape[0] // 2
    distance = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2

    # Create a smooth mask based on the distance
    mask = 1 - np.sqrt(distance)

    # Clip values to ensure they are in the range [0, 1]
    mask = np.clip(mask, 0, 1)

    return mask


# Example usage:
shape = (200, 200)
a = 80  # Semi-major axis
b = 50  # Semi-minor axis
heightmap_mask = generate_elliptical_heightmap_mask(shape, a, b)

# Display the elliptical heightmap mask
# import matplotlib.pyplot as plt
# plt.imshow(heightmap_mask, cmap='gray')
# plt.title('Elliptical Heightmap Mask')
# plt.colorbar(label='Height')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# Example usage:
# Assuming `heightmap_mask` is your numpy heightmap mask and `center` is the center point
center = (heightmap_mask.shape[1] // 2, heightmap_mask.shape[0] // 2)
radial_field = generate_radial_field(center, heightmap_mask, heightmap_mask.shape) * 0.01

mask = np.linalg.norm(radial_field, axis=2) == 0

# Plot the radial field
plt.figure(figsize=(8, 8))
plt.quiver(np.ma.masked_array(radial_field[:, :, 0], mask),
           np.ma.masked_array(radial_field[:, :, 1], mask), scale=20)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Radial Field')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
