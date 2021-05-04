import math
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

transform = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

# call this method to run the improved agent.
def multi_class(file_path: str):  # SGD: 1 input vecotr
    color_left, color_right, gray_left, gray_right, typical_colors = elbow(file_path)
    classes_matrix = np.array(typical_colors)

    row_num, col_num = int(color_left.shape[0]), int(color_left.shape[1])
    left_side_indices = list()
    for i in range(1, row_num - 1):
        for j in range(1, col_num - 1):
            left_side_indices.append((i, j))

    learning_rate = 0.01
    classes = classes_matrix.shape[0]
    vector_size = 10
    rgb_weights = np.full((classes, vector_size), 0.001, dtype=np.longdouble)

    min_mean_squared_error = math.inf
    for epoch in range(5000):
        indices = left_side_indices.copy()
        while len(indices) > 0:
            index_tuple = random.choice(indices)
            row, col = index_tuple
            x_vector = [1] + [gray_left[row + i][col + j] for (i, j) in transform]
            rgb = color_left[row][col]  # values for red, green and blue of the center pixel
            distances = [np.linalg.norm(np.subtract(classes_matrix[i], rgb))
                         for i in range(classes_matrix.shape[0])]
            rgb_index = np.argmin(distances)
            y_index = rgb_index
            indices.remove(index_tuple)
            # calculate e^(w*x) and stabilize the power
            dot_products = rgb_weights.dot(x_vector)
            stabilizer = np.max(dot_products)
            dot_products -= stabilizer  # stabilize e^(w*x)
            e_power_dot_products = np.zeros_like(dot_products, dtype=np.longdouble)
            for x in range(len(dot_products)):
                e_power_dot_products[x] = np.exp(dot_products[x])
            sums = np.sum(e_power_dot_products)  # sums = sum_{d=0..classes}(e^(w(d)*x))

            # update weight for each class.
            # p = [e^(w(j)*x_v)] / s
            # if j = c:  w(j, k+1) = w(j, k) - alpha * (p - 1) * x + lambda * R(w(j, k))
            # if j != c: w(j, k+1) = w(j, k) - alpha * p * x + lambda * R(w(j, k))
            for j in range(len(rgb_weights)):
                p = e_power_dot_products[j] / sums
                if y_index == j:
                    p = p - 1
                gradient = np.multiply(p, x_vector)
                l2_reg = rgb_weights[j]
                lambda_2 = 0.001
                change = gradient + lambda_2 * l2_reg
                rgb_weights[j] -= np.multiply(learning_rate, change)

        # display recolored image
        if epoch % 1 == 0:
            h, w, _ = color_right.shape
            recolor_matrix = np.zeros((h, w, 3))
            sum_squared_error = 0
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    x_vector = [1] + [gray_right[i + dx][j + dy] for (dx, dy) in transform]
                    dot_products = rgb_weights.dot(x_vector)
                    max_product = np.max(dot_products)
                    dot_products -= max_product  # shift power to avoid overflow of e^n
                    numerator = dot_products
                    for x in range(len(dot_products)):
                        numerator[x] = np.exp(numerator[x])
                    # find the max numerator without dividing by the same denominator to get the actual max(probability)
                    max_index = np.argmax(numerator)
                    recolor_matrix[i][j] = classes_matrix[max_index]
                    error_vector = np.subtract(recolor_matrix[i][j], color_right[i][j])
                    for val in error_vector:
                        sum_squared_error += val ** 2
            mean_squared_error = sum_squared_error / ((h - 1) * (w - 1) * 3)
            print(f'Epoch #: {epoch}; Mean Squared Error: {mean_squared_error}')
            recolor_matrix = np.hstack((color_left, recolor_matrix)).astype('uint8')
            if mean_squared_error < min_mean_squared_error: 
                plt.imsave(f'output/tiger/{epoch}-{learning_rate}-{int(mean_squared_error)}.png', recolor_matrix)
                plt.imshow(recolor_matrix, interpolation='nearest')
                plt.show()
            min_mean_squared_error = min(min_mean_squared_error, mean_squared_error)

        learning_rate = max(learning_rate / 2, 0.000001)  # decrease learning rate


def image_to_matrix(file_str):  # convert image to ndarray
    img = Image.open(file_str)
    pixels = img.load()
    width, height = img.size
    color_matrix = np.zeros((height, width, 3), dtype=np.uint8)
    gray_matrix = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            color_matrix[y][x][0] = r
            color_matrix[y][x][1] = g
            color_matrix[y][x][2] = b
            gray_matrix[y][x] = 0.21 * r + 0.72 * g + 0.07 * b
    return color_matrix, gray_matrix


def get_data(file_path: str):
    color_matrix, gray_matrix = image_to_matrix(file_path)
    [color_left, color_right] = np.hsplit(color_matrix, 2)
    [gray_left, gray_right] = np.hsplit(gray_matrix, 2)
    # use elbow method to get the typical colors by using the k-means from the basic agent.
    class_arr = np.array([[59.18342246, 59.03262032, 76.10481283],
                          [242.15928369, 242.60603205, 232.06597549],
                          [74.30608482, 71.3165335, 76.32452366],
                          [169.28809265, 205.2432139, 249.52750633],
                          [193.29177057, 190.40274314, 175.45760599],
                          [147.38100821, 147.25674091, 158.33059789],
                          [116.47982063, 123.32959641, 143.49327354],
                          [132.57314629, 134.29258517, 138.32865731],
                          [28.49697703, 19.81257557, 36.35187424],
                          [51.35787671, 55.54109589, 76.31335616],
                          [125.08729595, 124.38112136, 140.79418027],
                          [94.93195876, 92.72082474, 105.2156701],
                          [109.54182156, 108.08643123, 124.31319703],
                          [150.80952381, 152.17898194, 158.75123153],
                          [116.58915595, 103.11850196, 80.02124092],
                          [193.34016393, 201.06557377, 211.21311475]])
    return color_left, color_right, gray_left, gray_right, class_arr
