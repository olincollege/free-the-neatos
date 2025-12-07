import numpy as np
from scipy.spatial import KDTree as kd

def transformation_from_icp(source, target, max_iterations=20, tolerance=1e-6):
    """
    Performs iterative closest point (ICP) to align the source scan to the 
    target scan in order to find the best rigid transformation between the 2.
    
    Args:
        source: Nx2 array representing the source 2D point cloud.
        target: Nx2 array representing the target 2D point cloud.
        max_iterations: maximum number of ICP iterations.
        tolerance: convergence tolerance.
    
    Returns:
        R: 2x2 array representing the rotation matrix
        t: 2D array representing the translation vector.
    """

    for _ in range(max_iterations):
        # find closest points
        S, T = findClosestPoints(source, target)

        # align source to target via SVD
        Tmat = alignSVD(S, T)

        # transform source point cloud
        ones = np.ones((source.shape[0], 1))
        homogeneous_S = np.hstack((source, ones)) # create Nx3 to apply transformation
        transformed_S = (Tmat @ homogeneous_S.T).T # results in Nx3
        transformed_S = transformed_S[:, 0:2] # only use Nx2 part

        # check for convergence
        mean_error = np.mean(np.linalg.norm(transformed_S - T, axis=1))
        if mean_error < tolerance:
            break

        # update source point cloud for next iteration
        source = transformed_S
    
    # extract rotation and translation from Tmat
    R = Tmat[0:2, 0:2]
    t = Tmat[0:2, 2]

    return R, t

def findClosestPoints(source, target):
    """
    Find the closest points in target for each point in source.

    Args:
        source: Nx2 array representing the source 2D point cloud.
        target: Nx2 array representing the target 2D point cloud.

    Returns:
        matched_source: Nx2 array representing the matched source points.
        matched_target: Nx2 array representing the corresponding closest 
                        target points.
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
        matched_source: Nx2 array representing the source 2D point cloud.
        matched_target: Nx2 array representing the target 2D point cloud.

    Returns:
        T: 3x3 array representing the transformation matrix from source to target.
    """
    
    # compute centroids of each point cloud
    source_centroid = np.mean(matched_source, axis=0)
    target_centroid = np.mean(matched_target, axis=0)

    # normalize via centering the point clouds
    S = matched_source - source_centroid
    T = matched_target - target_centroid

    # compute cross-covariance matrix M
    M = T.T @ S

    # compute SVD of M
    U, S, Vt = np.linalg.svd(M)

    # compute rotation R and translation t from svd(M)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: # ensure a proper rotation (det(R) = 1)
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = target_centroid - R @ source_centroid

    # formatting transformation matrix T
    Tmat = np.eye(3)
    Tmat[0:2, 0:2] = R
    Tmat[0:2, 2] = t

    return Tmat