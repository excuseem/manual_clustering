from itertools import combinations
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from descartes.patch import PolygonPatch
from matplotlib import colors as mcolors
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
import string
import random

from satflow_algo.clustering import BoundedCluster, find_polygon_as_rectilinear_hull
from checker.layout import Pin, Board, Layer


def get_points_for_voronoi(pins: List[Pin]) -> np.array:
    """
    :param pins:
    :return: array of points for plotting Voronoi diagram
    """
    return np.array([[pin.center.x, pin.center.y] for pin in pins])

def plot_voronoi_diagram(points: np.array) -> None:
    """
    function that plots Voronoi diagram
    """
    diagram = Voronoi(points)
    voronoi_plot_2d(diagram)
    plt.show()

def distance_between_pins(pin1: Pin, pin2: Pin) -> float:
    """
    :param pin1:
    :param pin2:
    :return: the distance between pins (as the distance between their polygons)
    """
    return pin1.shape.distance(pin2.shape)

def distance_between_clusters(clusters: Tuple[BoundedCluster, BoundedCluster]) -> float:
    """
    :param clusters:
    :return: distance between clusters is min distance between pins from each cluster
    """
    cluster1, cluster2 = clusters
    return cluster1.polygon.distance(cluster2.polygon)

def pre_connection(board: Board, max_dist: float) -> List[Tuple[Pin, Pin]]:
    """
    Returns a list of all pairs of pins from the same network that are closer than |max_dist| to each other
    :param board:
    :param max_dist: max allowed distance between pins inside one cluster
    :return: list of pairs of pins that should be inside one cluster
    """
    pairs = []
    for network in board.networks.values():
        for pin1, pin2 in combinations(network.pins, 2):
            if distance_between_pins(pin1, pin2) < max_dist:
                pairs.append((pin1, pin2))

    return pairs

def index_of_cluster(clusters: List[BoundedCluster]) -> int:
    """
    finds index of cluster that pin belongs to
    """
    def get_index(pin: Pin):
        for i, cluster in enumerate(clusters):
            if pin in cluster.pins:
                return i
    return get_index

def unite_and_replace(
    index_cluster_a: int,
    index_cluster_b: int,
    clusters: List[BoundedCluster],
    max_number_of_pins: int,
    max_diam: float,
) -> Optional[BoundedCluster]:
    """
    tries to unite given clusters and replace the old clusters by a new one
    in this function list of clusters changes: if we unite cluster_a and cluster_b, we replace them by their union
    :param index_cluster_a:
    :param index_cluster_b:
    :param clusters:
    :param max_number_of_pins: maximal allowed number of pins in each cluster
    :param max_diam: maximal allowed diameter of cluster (approximately)
    :return: the cluster that constructs from cluster_a and cluster_b if all conditions are met else None
    """
    cluster_a, cluster_b = clusters[index_cluster_a], clusters[index_cluster_b]
    if len(cluster_a.pins) + len(cluster_b.pins) > max_number_of_pins:
        return None
    cluster_c = cluster_a.union(cluster_b)

    # boundary of union of two clusters
    cluster_c.update_polygon()
    minx, miny, maxx, maxy = cluster_c.polygon.minimum_rotated_rectangle.bounds

    # if diameter > max_diam: do NOT unite
    if Point(minx, miny).distance(Point(maxx, maxy)) > max_diam:
        return None
    # change clusters a and b to their union
    cluster_b.delete_polygon()
    cluster_a.delete_polygon()

    return cluster_c


def clusterization(
    board: Board,
    layer: Layer,
    number_of_clusters: int,
    max_number_of_pins: int,
    max_diam: float,
    max_dist: float,
    dist_between_clusters: float,
) -> List[BoundedCluster]:
    """
    returns list of clusters, algorithm consists of several parts:
    1) if two pins are in one net and are close to each other - unite clusters that contain these pins
    2) hierarchical clustering (unite the closest clusters until we get needed number of clusters
    (if all * conditions are met))
    *) do not unite clusters if diam of union cluster is bigger than max_diam
    *) do not unite clusters if number of pins in union cluster is bigger than max_number_of_pins

    :param board: selected board
    :param layer: selected layer
    :param number_of_clusters: desired number of clusters
    :param max_number_of_pins: maximal allowed number of pins in each cluster
    :param max_diam: maximal allowed diameter of cluster (approximately)
    :param max_dist: max allowed distance between pins inside one cluster
                    (used for uniting some clusters in which pins are connected)
    :return: list of clusters from the particular board and layer
    :param dist_between_clusters:
    """
    pins = [pin for pin in board.pins if pin.layer.compare(layer)]  # list of pins in the board on selected layer
    letters = string.ascii_lowercase
    clusters = [
        BoundedCluster(
            "CLUSTER" + str("".join(random.choice(letters) for _ in range(5))),
            [pin],
            find_polygon_as_rectilinear_hull,
        )
        for i, pin in enumerate(pins)
    ]  # list of clusters
    current_number_of_clusters = len(clusters)  # number of clusters (it changes while we are uniting clusters)
    pairs = pre_connection(board, max_dist)  # pairs of pins that should be connected before main clusterization

    # distances between all combinations of clusters
    dist = [(a, b) for a, b in combinations(clusters, 2)]
    list.sort(dist, key=distance_between_clusters)

    # if two pins are in one net and are close to each other - unite clusters that contain these pins
    for pin_a, pin_b in pairs:
        index_cluster_a, index_cluster_b = list(map(index_of_cluster(clusters), [pin_a, pin_b]))
        cluster_ab = (cluster_a, cluster_b) = clusters[index_cluster_a], clusters[index_cluster_b]
        # check if the distance between clusters A and B is smaller than given parameter
        if distance_between_clusters(cluster_ab) > dist_between_clusters:
            break
        cluster_c = unite_and_replace(index_cluster_a, index_cluster_b, clusters, max_number_of_pins, max_diam)
        if cluster_c:
            clusters = [cluster for cluster in clusters if cluster not in cluster_ab]
            clusters.append(cluster_c)
            for ind, (clust_a, clust_b) in enumerate(dist):
                # change clusters a and b to their union (in dist)
                if clust_a == cluster_b or clust_a == cluster_a:
                    dist[ind] = (cluster_c, dist[ind][1])
                if clust_b == cluster_b or clust_b == cluster_a:
                    dist[ind] = (dist[ind][0], cluster_c)

            current_number_of_clusters -= 1

        # if there are needed number of clusters: break
        if current_number_of_clusters == number_of_clusters:
            break

    for i, pair in enumerate(dist):
        cluster_a, cluster_b = pair
        if cluster_a == cluster_b:
            continue
        # check if the distance between clusters A and B is smaller than given
        # parameter
        if distance_between_clusters(dist[i]) > dist_between_clusters:
            break
        index_cluster_a, index_cluster_b = list(map(clusters.index, [cluster_a, cluster_b]))
        cluster_c = unite_and_replace(index_cluster_a, index_cluster_b, clusters, max_number_of_pins, max_diam)

        if cluster_c:
            clusters = [cluster for cluster in clusters if cluster not in [cluster_a, cluster_b]]
            clusters.append(cluster_c)

            for ind, (clust_a, clust_b) in enumerate(dist):
                # change clusters a and b to their union (in dist)
                if clust_a == cluster_b or clust_a == cluster_a:
                    dist[ind] = (cluster_c, dist[ind][1])
                if clust_b == cluster_b or clust_b == cluster_a:
                    dist[ind] = (dist[ind][0], cluster_c)

            current_number_of_clusters -= 1

        # if there are needed number of clusters: break
        if current_number_of_clusters == number_of_clusters:
            break

    return list(set(clusters))


def plot_clusters(clusters):
    fig, ax = plt.subplots()
    colors = list(mcolors.BASE_COLORS.keys())  # the list of colors
    for i, cluster in enumerate(clusters):
        # plotting convex hulls
        if cluster.polygon:
            patch = PolygonPatch(cluster.polygon, facecolor="None", edgecolor="green")
            ax.add_patch(patch)
        for pin in cluster.pins:
            patch = PolygonPatch(pin.shape, facecolor=colors[i % 6], edgecolor="None")
            ax.add_patch(patch)
    ax.autoscale()

    plt.show()
