import copy
import heapq
import random

import maxflow
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.util import view_as_windows
from scipy.spatial.distance import cdist


def grayscale_conversion(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_transparent_pixels(image):
    if len(image.shape) == 3 and image.shape[2] == 4:
        transparent_pixels = set()

        height, width = image.shape[:2]
        for y in range(height):
            for x in range(width):
                if image[y, x, 3] == 0:
                    transparent_pixels.add((x, y))

        return transparent_pixels
    else:
        return set()


def is_pixel_transparent(transparent_pixels, x, y):
    return (x, y) in transparent_pixels


def quantization(img_gray):
    # hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    # n_shades = len([i for i in hist if i != 0])
    #img_gray = cv2.GaussianBlur(cv2.GaussianBlur(img_gray, (5, 5), 0), (5, 5), 0)

    n_shades = 5
    img_quantized = (img_gray // (256 // n_shades)) * (256 // n_shades)
    return img_quantized


def are_segments_neighbors(segment1, segment2):
    segment1_set = set(segment1)
    segment2_set = set(segment2)

    for x1, y1 in segment1_set:
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            x2, y2 = x1 + dx, y1 + dy
            if (x2, y2) in segment2_set:
                return True

    return False


def calculate_centroid(points):
    if not points:
        return None

    x_sum = sum(x for x, y in points)
    y_sum = sum(y for x, y in points)

    centroid_x = x_sum / len(points)
    centroid_y = y_sum / len(points)

    return centroid_x, centroid_y


def flood_fill(image, transparent_pixels, visited, x, y, segments):
    height, width = image.shape
    visited[y, x] = True
    segments.append((x, y))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy

        if (0 <= nx < width and 0 <= ny < height) and (not visited[ny, nx]) and \
                (not is_pixel_transparent(transparent_pixels, nx, ny)) \
                and (image[y, x] == image[ny, nx]):
            flood_fill(image, transparent_pixels, visited, nx, ny, segments)


def segmentation(image, transparent_pixels):
    height, width = image.shape
    visited = np.zeros((height, width), dtype=bool)
    segments = []

    for y in range(height):
        for x in range(width):
            if not visited[y, x] and not is_pixel_transparent(transparent_pixels, x, y):
                new_segment = []
                flood_fill(image, transparent_pixels, visited, x, y, new_segment)
                segments.append([new_segment, image[y, x], calculate_centroid(new_segment)])

    return segments


def find_closest_coordinates_indices(x, y, coordinates_list, n):
    distances = []
    for index, (x2, y2) in enumerate(coordinates_list):
        distance = ((x - x2) ** 2 + (y - y2) ** 2) ** 0.5
        distances.append((distance, index))

    closest_distances = heapq.nsmallest(n, distances)
    closest_indices = [index for _, index in closest_distances]

    return closest_indices

def remove_segments(sf, sb):
    coordinates = [item for sublist in sf for item in sublist[0]]
    set_coord = set(coordinates)
    for seg_b in sb:
        set_b = set(seg_b[0])
        if not (set_coord & set_b):
            sb.remove(seg_b)



def graph_construction(sf, sb):
    len_sf = len(sf)
    G_f = {index: [segment[1], []] for index, segment in enumerate(sf)}
    G_b = {index + len_sf: [segment[1], []] for index, segment in enumerate(sb)}
    G = {**G_f, **G_b}

    k_nearest_coeficient = 6

    for i in range(len_sf):
        for j in range(i + 1, len_sf):
            if are_segments_neighbors(sf[i][0], sf[j][0]):
                G[i][1].append(j)
                G[j][1].append(i)

    sb_centroids = [tup[2] for tup in sb]

    for i in range(len_sf):
        k_nearest = find_closest_coordinates_indices(sf[i][2][0], sf[i][2][1], sb_centroids, k_nearest_coeficient)
        for j in k_nearest:
            G[i][1].append(j + len_sf)
            G[j + len_sf][1].append(i)

    return G


def V_p_q(label1, label2):
    '''Definition of the potential'''
    # return 45*abs(label1-label2)
    return abs(label1 - label2)
    # return min(10,abs(label1-label2))


def D_p(label, actual_luminance):
    '''Returns the quadratic difference between label and real intensity of pixel'''
    return (abs(label ** 2 - actual_luminance ** 2)) ** 0.5  # best working D_p
    # return (label-graph[y][x])**2


def alpha_beta_swap(alpha, beta, work_graph, original_graph):
    # extract position of alpha or beta pixels to mapping
    filtered_graph = {k: v for k, v in work_graph.items() if v[0] in [alpha, beta]}
    index_to_node = {index: key for index, (key, _) in enumerate(filtered_graph.items())}
    node_to_index = {value: key for key, value in index_to_node.items()}

    len_graph = len(filtered_graph)
    # graph of maxflow
    graph_mf = maxflow.Graph[float](len_graph)
    # add nodes
    nodes = graph_mf.add_nodes(len_graph)

    # add n-link edges
    weight = V_p_q(alpha, beta)
    for i in range(len_graph):
        node = index_to_node[i]
        for neighbour in work_graph[node][1]:
            if neighbour in filtered_graph:
                graph_mf.add_edge(i, node_to_index[neighbour], weight, 0)

    # add all the terminal edges
    for i in range(len_graph):
        node = index_to_node[i]
        # find neighbours
        # consider only neighbours which are not having alpha or beta label
        fil_neigh = [work_graph[neighbour][0] for neighbour in work_graph[node][1] if neighbour not in filtered_graph]
        # calculation of weight
        t_weight_alpha = sum([V_p_q(alpha, v) for v in fil_neigh]) + D_p(alpha, original_graph[node][0])
        t_weight_beta = sum([V_p_q(beta, v) for v in fil_neigh]) + D_p(beta, original_graph[node][0])
        graph_mf.add_tedge(nodes[i], t_weight_alpha, t_weight_beta)

    # calculating flow
    flow = graph_mf.maxflow()
    res = [graph_mf.get_segment(nodes[i]) for i in range(len(nodes))]

    # depending on cut assign new label
    for i in range(0, len(res)):
        node = index_to_node[i]
        if res[i] == 1:
            work_graph[node][0] = alpha
        else:
            work_graph[node][0] = beta


# TODO fazer apenas alpha beta swap de luminancias do froeground com luminancias do background
def swap_minimization(original_graph, work_graph, cycles):
    labels = list(set(value[0] for value in work_graph.values()))

    for u in range(0, cycles):
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                alpha_beta_swap(labels[i], labels[j], work_graph, original_graph)
            # user output and interims result image


# def swap_minimization_modified(original_graph, work_graph, division_point, cycles):
#     values = list(work_graph.values())
#     sf_labels = list(set(value[0] for value in values[:division_point]))
#     sb_labels = list(set(value[0] for value in values[division_point:]))
#     #labels = list(set(value[0] for value in work_graph.values()))
#
#     for u in range(0, cycles):
#         for i in range(len(sf_labels)):
#             for j in range(len(sb_labels)):
#                 if(sf_labels[i]!=sb_labels[j]):
#                     alpha_beta_swap(sf_labels[i], sb_labels[j], work_graph, original_graph)
#                 # user output and interims result image

def update_image(graph, sf, image, background):
    for i in range(len(sf)):
        sf[i][1] = graph[i][0]
        for x, y in sf[i][0]:
            image[y, x] = sf[i][1]
            background[y, x, 1] = sf[i][1]

def update_gray_image(graph, sf, image, background):
    for i in range(len(sf)):
        sf[i][1] = graph[i][0]
        for x, y in sf[i][0]:
            image[y, x] = sf[i][1]
            background[y, x] = sf[i][1]

def texture_transfer(sb, sf, image_back, image_front):
    for f_seg in sf:
        elements = []
        for b_seg in sb:
            if(f_seg[1] == b_seg[1]):
               elements.append(b_seg)
        b_seg = random.choice(elements)
        len_b = len(b_seg[0])
        for i in range(len(f_seg[0])):
            image_front[f_seg[0][i][1], f_seg[0][i][0]] = \
                image_back[b_seg[0][i % len_b][1], b_seg[0][i % len_b][0]]

if __name__ == '__main__':
    sys.setrecursionlimit(1000000)

    # Load background and foreground images
    background = cv2.imread("test_images/Roche Rock.png", cv2.IMREAD_UNCHANGED)
    foreground = cv2.imread("test_images/Tiger.png", cv2.IMREAD_UNCHANGED)
    height, width, _ = background.shape
    foreground = cv2.resize(foreground, (height, width), interpolation=cv2.INTER_AREA)
    background_gray = quantization(grayscale_conversion(background))
    foreground_gray = quantization(grayscale_conversion(foreground))
    img2 = cv2.cvtColor(foreground_gray, cv2.COLOR_BGR2RGB)
    img4 = cv2.cvtColor(background_gray, cv2.COLOR_BGR2RGB)

    foreground_transparent = get_transparent_pixels(foreground)
    background_transparent = get_transparent_pixels(background)

    foreground_segments = segmentation(foreground_gray, foreground_transparent)
    background_segments = segmentation(background_gray, background_transparent)
    remove_segments(foreground_segments, background_segments)


    graph = graph_construction(foreground_segments, background_segments)
    work_graph = copy.deepcopy(graph)
    # swap_minimization_modified(graph, work_graph, len(foreground_segments), 5)
    swap_minimization(graph, work_graph, 5)

    hls_img = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
    hls_img = cv2.cvtColor(hls_img, cv2.COLOR_BGR2HLS)


    update_image(work_graph, foreground_segments, foreground_gray, hls_img)
    update_gray_image(work_graph, foreground_segments, foreground_gray, background_gray)
    img5 = cv2.cvtColor(background_gray, cv2.COLOR_BGR2RGB)

    foreground_segments = segmentation(foreground_gray, foreground_transparent)
    foreground_gray_color = cv2.cvtColor(foreground_gray, cv2.COLOR_GRAY2BGR)
    foreground_gray_hls = cv2.cvtColor(foreground_gray_color, cv2.COLOR_BGR2HLS)


    #texture_transfer(background_segments, foreground_segments, hls_img, foreground_gray_hls)



    img1 = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB)
    img3 = cv2.cvtColor(foreground_gray, cv2.COLOR_BGR2RGB)
    cv2.imwrite('test_images/Luminanve.png', img3)

    img1 = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)

    # Create a figure with two subplots, one for each image
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(10, 5))

    # Display the first image on the left subplot
    ax1.imshow(img1)
    ax1.set_title('Original')
    ax1.axis('off')
    # Display the second image on the right subplot
    ax2.imshow(img2)
    ax2.set_title('Quantizada')
    ax2.axis('off')
    ax3.imshow(img3)
    ax3.set_title('Otimizada')
    ax3.axis('off')
    ax4.axis('off')
    ax4.imshow(img4)
    ax4.set_title('Background quantizado')
    ax5.axis('off')
    ax5.imshow(img5)
    ax5.set_title('Resultado')

    plt.axis('off')
    # Show the plot
    plt.show()
