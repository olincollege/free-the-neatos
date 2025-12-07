import numpy as np
from scipy.spatial import KDTree as kd

def icp(source, target, max_iterations=20, tolerance=1e-6):
    """
    Perform Iterative Closest Point (ICP) algorithm to align source to target.
    
    Args:
        source: Nx2 array representing the source 2D point cloud.
        target: Nx2 array representing the target 2D point cloud.
        max_iterations: maximum number of ICP iterations.
        tolerance: convergence tolerance.
    
    Returns:
        T: 3x3 array representing the transformation matrix from source to target.
    """

    for _ in range(max_iterations):
        # find closest points
        matched_source, matched_target = findClosestPoints(source, target)

        # align source to target via SVD
        T = alignSVD(matched_source, matched_target)

        # transform source point cloud
        ones = np.ones((source.shape[0], 1))
        homogeneous_source = np.hstack((source, ones)) # create Nx3 to apply transformation
        transformed_source = (T @ homogeneous_source.T).T # results in Nx3
        source = transformed_source[:, 0:2] # only use Nx2 part

        # check for convergence
        mean_error = np.mean(np.linalg.norm(matched_source - matched_target, axis=1))
        if mean_error < tolerance:
            break
    
    return T

def findClosestPoints(source, target):
    """
    Find the closest points in target for each point in source.

    Args:
        source: Nx2 array representing the source 2D point cloud.
        target: Nx2 array representing the target 2D point cloud.

    Returns:
        matched_source: Nx2 array representing the source points.
        matched_target: Nx2 array representing the closest target points.
    """

    # build kd-tree for target point cloud
    tree = kd(target)
    # query kd-tree for closest points
    distances, indices = tree.query(source)

    # construct matched point clouds
    matched_target = target[indices]
    matched_source = source

    return matched_source, matched_target


def alignSVD(matched_source, matched_target):
    """
    Align source point cloud to target point cloud via SVD.

    Args:
        source: Nx2 array representing the source 2D point cloud.
        target: Nx2 array representing the target 2D point cloud.

    Returns:
        T: 3x3 array representing the transformation matrix from source to target.
    """
    # normalize point clouds about their centroids
    source_centroid = compute_centroid(matched_source)
    target_centroid = compute_centroid(matched_target)
    normalized_source = matched_source - source_centroid
    normalized_target = matched_target - target_centroid

    # compute cross-covariance matrix M
    M = normalized_source * normalized_target.T

    # compute SVD of M
    U, S, Vt = np.linalg.svd(M)

    # compute rotation R and translation t from svd(M)
    R = Vt.T @ U.T
    t = target_centroid - R @ source_centroid

    # formatting transformation matrix T
    T = np.eye(3)
    T[0:2, 0:2] = R
    T[0:2, 2] = t

    return T

def compute_centroid(points):
    """
    Compute the centroid of a set of points.

    Args:
        points: nx2 array representing a 2D point cloud.

    Returns:
        centroid: 2D array representing the centroid of the point cloud.
    """
    points_sum = np.sum(points, axis=0)
    centroid = points_sum / points.shape[0]
    return centroid