import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image, blur_size=7, lower_threshold=100, upper_threshold=200, blur=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur:
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    return edges

def sample_points_on_contour(contour, step=10):
    arc_length = cv2.arcLength(contour, closed=True)
    num_points = int(arc_length // step)
    t_values = np.linspace(0, 1, num_points, endpoint=False)
    sampled_points_indices = (t_values * len(contour)).astype(int) % len(contour)
    return contour[sampled_points_indices].reshape(-1, 2)

def segment_cup_and_find_points(image_path):
    image = cv2.imread(image_path)
    preprocessed_image = preprocess_image(image)
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    all_sampled_points = []

    for contour in contours:
        sampled_points = sample_points_on_contour(contour)
        all_sampled_points.append(sampled_points)

    return all_sampled_points

def plot_points_on_cup(image_path, all_points, highlight_points):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for points in all_points:
        for point in points:
            cv2.circle(image_rgb, tuple(point), 5, (255, 0, 0), -1)

    for point in highlight_points:
        cv2.circle(image_rgb, tuple(point), 10, (0, 255, 0), -1)

    plt.imshow(image_rgb)
    plt.show()

highlight_points = [
    [159, 73],
    [275, 133]
]

image_path = '/home/tyz/Desktop/LLM_For_Control/image_folder/cup4.png'
all_cup_coordinates = segment_cup_and_find_points(image_path)
plot_points_on_cup(image_path, all_cup_coordinates, highlight_points)
print([contour_points.tolist() for contour_points in all_cup_coordinates])
