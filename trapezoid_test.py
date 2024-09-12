import numpy as np
import cv2


def generate_trapezoidal_path(circle1_center, circle1_radius, circle2_center, circle2_radius, num_points=50):
    # Calculate linear path between circle centers
    linear_path = circle2_center - circle1_center

    # Normalize the linear path vector
    linear_path /= np.linalg.norm(linear_path)

    # Calculate perpendicular vectors with lengths equal to circle diameters
    perpendicular_vector1 = np.array([-linear_path[1], linear_path[0]], dtype=np.float64) * 2 * circle1_radius
    perpendicular_vector2 = np.array([linear_path[1], -linear_path[0]], dtype=np.float64) * 2 * circle2_radius



    # Combine the four corners of the trapezoid
    corners = np.array([
        points[0],
        points[num_points],
        points[-1],
        points[-num_points]
    ])

    return corners


# Test the function
circle1_center = np.array([100, 100])
circle1_radius = 30
circle2_center = np.array([300, 300])
circle2_radius = 50

trapezoidal_path = generate_trapezoidal_path(circle1_center, circle1_radius, circle2_center, circle2_radius)

# Create a blank image
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Draw the trapezoidal path

# Display the result
cv2.imshow('Trapezoidal Path', image)
cv2.waitKey(0)
cv2.destroyAllWindows()