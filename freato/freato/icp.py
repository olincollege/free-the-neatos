import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors

def tf_from_icp(source, target, T_init, max_iterations=20, tolerance=1e-6):
    """
    Performs iterative closest point (ICP) to align the source scan to the 
    target scan in order to find the best rigid transformation between the 2.
    
    Args:
        source: Nx2 array representing the source 2D point cloud.
        target: Nx2 array representing the target 2D point cloud.
        max_iterations: maximum number of ICP iterations.
        tolerance: convergence tolerance based on change in mean squared error.
    
    Returns:
        T: 3x3 array representing the transformation matrix from source to target.
    """

    # copy to not overrite caller data
    src = source.copy()
    tgt = target.copy()

    #target_tree = KDTree(tgt) # build kd-tree for target point cloud
    target_tree = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(tgt)
    
    Tmat = T_init  # initialize total transformation matrix

    mean_sq_error = 1.0e6  # initialize error as large number
    delta_err = 1.0e6    # change in error (used in stopping condition)

    # **FIX: Apply initial transformation to source**
    ones = np.ones((src.shape[0], 1))
    src_stacked = np.hstack((src, ones))
    src_tf = (Tmat @ src_stacked.T).T[:, 0:2]  # Apply T_init first!

    for _ in range(max_iterations):
        # find closest points
        src_matched, tgt_matched, idx = findCorrespondences(src_tf, tgt, target_tree)

        # align source to target via SVD
        Tmat_new = alignSVD(src_matched, tgt_matched)

        # accumulate transformation
        Tmat = Tmat_new @ Tmat

        # transform full source point cloud
        src_tf = (Tmat @ src_stacked.T).T[:, 0:2] # results in Nx2 array

        # calculate error using matched points only
        new_err = np.mean(np.sum((src_matched - tgt_matched)**2, axis=1))

        # Check convergence
        delta_err = abs(mean_sq_error - new_err)
        mean_sq_error = new_err
        
        if delta_err < tolerance:
            break


    return Tmat

def findCorrespondences(source, target, target_tree):
    """
    Find the closest points in target for each point in source.

    Args:
        source: Nx2 array representing the source 2D point cloud.
        target: Nx2 array representing the target 2D point cloud.

    Returns:
        matched_source: Nx2 array representing the matched source points.
        matched_target: Nx2 array representing the corresponding closest 
                        target points.
        idx: list of indices of the closest target points for each source point.
    """

    # Query for closest points
    dist, idx = target_tree.kneighbors(source)
    dist = dist.ravel()
    idx = idx.ravel()
    
    #dist, idx = target_tree.kneighbors(source)
    dist = dist.ravel()
    idx = idx.ravel()

    max_correspondence_dist = 0.3  # meters - tune this!
    
    # Filter by distance threshold
    valid_mask = dist < max_correspondence_dist
    
    # Remove duplicate matches
    target_to_source = {}
    for i in range(len(idx)):
        if not valid_mask[i]:
            continue
        target_idx = idx[i]
        if target_idx not in target_to_source or dist[i] < target_to_source[target_idx][1]:
            target_to_source[target_idx] = (i, dist[i])
    
    # Build correspondence lists
    valid_src_pts = []
    valid_tgt_pts = []
    for target_idx, (source_idx, _) in target_to_source.items():
        valid_src_pts.append(source[source_idx, :])
        valid_tgt_pts.append(target[target_idx, :])

    if len(valid_src_pts) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), []

    matched_src = np.array(valid_src_pts).reshape(-1, 2)
    matched_tgt = np.array(valid_tgt_pts).reshape(-1, 2)

    return matched_src, matched_tgt, list(target_to_source.keys())

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
    src_centroid = np.mean(matched_source, axis=0)
    tgt_centroid = np.mean(matched_target, axis=0)

    # normalize via centering the point clouds
    src_centered = matched_source - src_centroid
    tgt_centered = matched_target - tgt_centroid

    # compute cross-covariance matrix M
    M = np.dot(tgt_centered.T, src_centered)

    # compute SVD of M
    U, _, Vt = np.linalg.svd(M)

    # compute rotation R and translation t from svd(M)
    R = U @ Vt

    t = tgt_centroid - R @ src_centroid

    # formatting transformation matrix T
    Tmat = np.eye(3)
    Tmat[0:2, 0:2] = R
    Tmat[0:2, 2] = t

    return Tmat