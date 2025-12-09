import numpy as np
import matplotlib.pyplot as plt
from icp import tf_from_icp as tf_icp
#from icp_test_func import Align2D

def test_icp(rotation, translation):
    """
    Test the ICP implementation by generating two point clouds with a known
    rigid transformation between them, then using ICP to estimate that transformation.
    
    Args:
        rotation: degrees of rotation to apply to the original point cloud.
        translation: 2D array representing the translation vector.
    """

    # generate original point cloud (scan A)
    scan_A = generate_scan_A()

    # generate transformed point cloud (scan B)
    scan_B = generate_scan_B(scan_A, rotation, translation)

    # Convert to homogeneous coordinates
    scan_A_homogeneous = np.hstack([scan_A, np.ones((scan_A.shape[0], 1))])
    scan_B_homogeneous = np.hstack([scan_B, np.ones((scan_B.shape[0], 1))])

    #aligner = Align2D(scan_B_homogeneous, scan_A_homogeneous, np.eye(3))
    #T = aligner.AlignICP(50, 1e-6)

    T = tf_icp(scan_B, scan_A, max_iterations=100, tolerance=1e-12)

    # print estimated results
    print("Estimated rotation:\n", T[0:2,0:2])
    print("Estimated translation:\n", T[0:2,2])

    # compensate for reference frame difference
    print("\nExpected rotation:\n", np.linalg.inv(np.array([
        [np.cos(np.deg2rad(rotation)), -np.sin(np.deg2rad(rotation))],
        [np.sin(np.deg2rad(rotation)),  np.cos(np.deg2rad(rotation))]
    ])))
    print("Expected translation:\n", -1*np.array(translation))

    plt.figure()
    plt.scatter(scan_A[:,0], scan_A[:,1], s=1, label='Scan A (target)', c='blue')
    plt.scatter(scan_B[:,0], scan_B[:,1], s=1, label='Scan B (source)', c='red')
    plt.axis('equal')
    plt.legend()
    plt.title("Original Scans")
    plt.show()

def generate_scan_A():
    """
    Generate a 2D point cloud representing a simple indoor environment
    """
    pts = []

    # rectangular room
    xs = np.linspace(-4, 4, 120)
    #pts += [[x, 4] for x in xs]     # top wall
    pts += [[x, -4] for x in xs]    # bottom wall
    ys = np.linspace(-4, 4, 120)
    pts += [[-4, y] for y in ys]    # left wall
    pts += [[4, y] for y in ys]     # right wall

    # circular pillar
    # theta = np.linspace(0, 2*np.pi, 80)
    # pts += [[1 + 1.0*np.cos(t), 1 + 1.0*np.sin(t)] for t in theta]

    # block
    xs = np.linspace(-1.5, -0.5, 40)
    ys = np.linspace(0.5, 1.5, 40)
    pts += [[xs[i], 0.5] for i in range(40)]
    #pts += [[xs[i], 1.5] for i in range(40)]
    #pts += [[-1.5, ys[i]] for i in range(40)]
    pts += [[-0.5, ys[i]] for i in range(40)]
    

    return np.array(pts)

def generate_scan_B(scan_A, rotation, translation):
    """
    Generate a transformed version of scan_A by applying a known rigid transformation
    """

    theta = np.deg2rad(rotation) # rotation
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    t = np.array(translation) # translation

    scan_B = (R @ scan_A.T).T + t
    return scan_B

if __name__ == "__main__":
    test_icp(-10, [0.0, -0.2])