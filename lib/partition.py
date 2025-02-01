import numpy as np
from scipy.spatial.transform import Rotation


def random_displacement(n_pts: int, r: float):
    """
    Random sample displacement vectors [n_pts,3] in a ball with radius r
    :param n_pts:
    :param r:
    :return:
    """
    # Use spherical coordinates
    u = np.random.rand(n_pts)
    v = np.random.rand(n_pts)

    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    R = r * np.cbrt(np.random.rand(n_pts))

    # Convert to cartesian coordinates
    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)
    sinPhi = np.sin(phi)
    cosPhi = np.cos(phi)

    x = R * sinPhi * cosTheta
    y = R * sinPhi * sinTheta
    z = R * cosPhi

    return np.stack([x, y, z], axis=-1)


def distance_aware_weighting(alpha: float, beta: float, dists: np.ndarray, dmax: float):
    """
    Distance aware weighting function
    w(x)=(1-\alpha)e^{-\beta\frac{\|p_x-c\|}{d_{max}}}+\alpha
    :param alpha:
    :param beta:
    :param dist:
    :param dmax:
    :return:
    """
    return (1 - alpha) * np.exp(-beta * (dists / dmax)) + alpha


def periodic_sample(
    points: np.ndarray,
    alpha: float,
    beta: float,
    T: float,
    k: int = 3,
    return_mask=False,
):
    """
    Periodic sampling: Real scans do not necessarily have regular sampling because sampling depends on the point of view.
    1. randomly sample a center c
    2. compute binary mask m_i = 1(|cos(\frac{2\pi}{T}||x_i-c||)| > cos(
    alpha
    phi) )
    :param points:
    :param alpha:
    :param T:
    :return:
    """

    # 1. randomly sample points and form a center c
    samples = points[np.random.randint(0, len(points) - 1, size=k)]
    c = np.mean(samples, axis=0)

    # 2. compute binary mask
    dists = points - c
    dists = np.linalg.norm(dists, axis=-1)
    dmax = np.max(dists)

    daw = distance_aware_weighting(alpha, beta, dists, dmax)

    mask = np.abs(np.cos(2 * np.pi / T * dists)) > np.cos(daw)

    # print(mask)

    if return_mask:
        return points[mask], mask
    else:
        return points[mask]


def partition_pcd(points: np.ndarray, a_ratio: float, overlap: float, return_idx=False):
    n_pts = len(points)

    overlap_len = int(n_pts * overlap)
    a_len = int(n_pts * a_ratio) + int(overlap_len / 2)
    b_len = n_pts - int(n_pts * a_ratio) + int(overlap_len / 2)

    # print(a_ratio,a_len, b_len,overlap_len,n_pts)

    seed_idx = np.random.randint(0, n_pts)

    dists = points - points[seed_idx]
    dists = np.linalg.norm(dists, axis=-1)

    sorted_idxes = np.argsort(dists)

    # print(dists[sorted_idxes])

    a_idx = sorted_idxes[:a_len]
    b_idx = sorted_idxes[-b_len:]

    if return_idx:
        return points[a_idx], points[b_idx], a_idx, b_idx
    else:
        return points[a_idx], points[b_idx]


def sample_random_transform(
    rot_mean: float, rot_sigma: float, trans_mean: float, trans_sigma: float
):
    """
    Sample random rigid transformation with rotation angle of x,y,z axis iid N(rot_mean,rot_sigma) and |translation| ~ N(trans_mean, trans_sigma)
    :param rot_mean:
    :param rot_sigma:
    :param trans_mean:
    :param trans_sigma:
    :return: rotation matrix and translation vector
    """

    # 1. Sample random rotation angles
    rot = np.random.normal(rot_mean, rot_sigma, 3)

    # x,y maybe should not be rotated for road scene
    rot[0] = 0.0
    rot[1] = 0.0

    rot_mat = Rotation.from_euler("xyz", rot).as_matrix()

    # 2. Sample random translation vector whose length ~ N(trans_mean, trans_sigma)
    length = np.random.normal(trans_mean, trans_sigma)
    rand_vec = np.random.normal(size=3)

    # z should not be altered for raod scene
    rand_vec[2] = 0.0

    rand_vec = rand_vec / np.linalg.norm(rand_vec) * length

    return rot_mat, rand_vec


def generate_pair(
    points: np.ndarray,
    a_ratio: float,
    overlap: float,
    r: float,
    alpha: float = 0.8,
    beta: float = 200,
    T: float = 2,
    k: int = 8,
    rot_mean=np.pi / 2,
    rot_sigma=np.pi,
    trans_mean=10,
    trans_sigma=5,
    partial_partition=False,
    add_displacement=True,
    enable_periodic_sample=True,
):
    if partial_partition:
        a, b = partition_pcd(points, a_ratio, overlap)
    else:
        # we just let a,b be the same. they are all a copy of points
        a = points.copy()
        b = points.copy()

    if add_displacement:
        a_displacement = random_displacement(len(a), r)
        b_displacement = random_displacement(len(b), r)

        a += a_displacement
        b += b_displacement

    if enable_periodic_sample:
        a = periodic_sample(a, alpha, beta, T, k)
        b = periodic_sample(b, alpha, beta, T, k)

    rot, trans = sample_random_transform(rot_mean, rot_sigma, trans_mean, trans_sigma)

    # apply transformation
    rotation = Rotation.from_matrix(rot)

    a = rotation.apply(a) + trans

    return a, b, rot, trans


def generate_pair_with_idx(
    points: np.ndarray,
    a_ratio: float,
    overlap: float,
    r: float,
    alpha: float = 0.8,
    beta: float = 200,
    T: float = 2,
    k: int = 8,
    rot_mean=np.pi / 2,
    rot_sigma=np.pi,
    trans_mean=10,
    trans_sigma=5,
    partial_partition=False,
    add_displacement=True,
    enable_periodic_sample=True,
):
    if partial_partition:
        a, b, a_idx, b_idx = partition_pcd(points, a_ratio, overlap, return_idx=True)
    else:
        # we just let a,b be the same. they are all a copy of points
        a = points.copy()
        b = points.copy()
        a_idx = np.arange(len(a))
        b_idx = np.arange(len(b))

    if add_displacement:
        a_displacement = random_displacement(len(a), r)
        b_displacement = random_displacement(len(b), r)

        a += a_displacement
        b += b_displacement

    if enable_periodic_sample:
        a, mask_a = periodic_sample(a, alpha, beta, T, k, return_mask=True)
        b, mask_b = periodic_sample(b, alpha, beta, T, k, return_mask=True)
        a_idx = a_idx[mask_a]
        b_idx = b_idx[mask_b]

    rot, trans = sample_random_transform(rot_mean, rot_sigma, trans_mean, trans_sigma)

    # apply transformation
    rotation = Rotation.from_matrix(rot)

    a = rotation.apply(a) + trans

    return a, b, a_idx, b_idx, rot, trans
