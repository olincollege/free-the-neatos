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

    src_tf = src # initialize transformed source points as original source points for 1st iteration

    for _ in range(max_iterations):
        # find closest points
        src_matched, tgt_matched, idx = findCorrespondences(src_tf, tgt, target_tree)

        # align source to target via SVD
        Tmat_new = alignSVD(src_matched, tgt_matched)

        # accumulate transformation
        Tmat = Tmat_new @ Tmat

        # transform full source point cloud
        ones = np.ones((src.shape[0], 1))
        src_stacked = np.hstack((src, ones)) # create Nx3 to apply transformation
        src_tf = (Tmat @ src_stacked.T).T[:, 0:2] # results in Nx2 array

        # find mean squared error between corresponding matched transformed source points and target points
        new_err = 0
        for i, idx_val in enumerate(idx):
            if idx_val != -1:
                diff = src_tf[i,:] - tgt[idx_val,:]
                new_err += np.dot(diff, diff.T)

        new_err /= float(len(tgt_matched))

        # update error and calculate delta error
        delta_err = abs(mean_sq_error - new_err)
        if delta_err < tolerance:
            break
        mean_sq_error = new_err

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
    
    # query kd-tree for closest points
    #dist, idx = target_tree.query(source)
    dist, idx = target_tree.kneighbors(source)
    dist = dist.ravel()
    idx = idx.ravel()

    # remove duplicate matches
    unique = False
    while not unique:
        unique = True
        for i in range(len(idx)): # sort through each source point
            if idx[i] == -1:
                continue
            for j in range(i+1,len(idx)): # compare to every other source point
                if idx[i] == idx[j]: # if identical match exists...
                    if dist[i] < dist[j]: # keep match with closer distance
                        idx[j] = -1
                    else:
                        idx[i] = -1
                        break

    # build array of nearest neighbor target points and corresponding source points
    valid_src_pts = []
    valid_tgt_pts = []
    for i, idx_val in enumerate(idx):
        if idx_val != -1:
            valid_tgt_pts.append(target[idx_val, :])
            valid_src_pts.append(source[i, :])

    matched_src = np.array(valid_src_pts).reshape(-1,2)
    matched_tgt = np.array(valid_tgt_pts).reshape(-1,2)

    return matched_src, matched_tgt, idx

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