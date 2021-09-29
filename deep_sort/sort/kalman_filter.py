# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


# 卡方分布-马氏距离
# https://www.cnblogs.com/Yuanjing-Liu/p/9252844.html
# https://blog.csdn.net/bluesliuf/article/details/88862918
"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # 卡尔曼滤波的状态转移矩阵 匀速模型
        self._motion_mat = np.eye(2 * ndim, 2 * ndim) # shape(8, 8)

        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # 卡尔曼滤波更新矩阵
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.

        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        # measurement -> detection.to_xyah()
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement  # [4] # 构建位置信息
        mean_vel = np.zeros_like(mean_pos)  # [4] 刚出现的目标位置默认速度为0
        mean = np.r_[mean_pos, mean_vel]  # [8]
        # 初始化的mean = {}
        # P 估计误差协方差矩阵初始化
        # 协方差矩阵，其中的元素值越大则不确定性越大，可以任意初始化
        std = [
            2 * self._std_weight_position * measurement[3],  # x 
            2 * self._std_weight_position * measurement[3],  # y
            1e-2,                                            # a
            2 * self._std_weight_position * measurement[3],  # h
            10 * self._std_weight_velocity * measurement[3],  # vx
            10 * self._std_weight_velocity * measurement[3],  # vy
            1e-5,                                            # va
            10 * self._std_weight_velocity * measurement[3]]  # vh

        covariance = np.diag(np.square(std))  # 对角线 [8, 8] 取平方后转化为对角线

        return mean, covariance

    def predict(self, mean, covariance):
        # 相当于得到t时刻估计值
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """

        # Q 预测过程中噪声协方差
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]

        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]

        # np.r_ 按列连接两个矩阵
        # 初始化噪声矩阵Q
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) # shape(8, 8)

        # x' = Fx
        mean = np.dot(self._motion_mat, mean) # shape(8) 

        # P' = FPF^T+Q
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        # R 测量过程中噪声的协方差
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]

        # 初始化噪声矩阵R
        innovation_cov = np.diag(np.square(std))

        # 将均值向量映射到检测空间，即Hx'
        mean = np.dot(self._update_mat, mean) # shape(8, )

        # 将协方差矩阵映射到检测空间，即HP'H^T
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T)) # shape(4, 4)

        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        # 通过估计值和观测值估计最新结果
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """

        # 将均值和协方差映射到检测空间，得到 Hx' 和 S
        projected_mean, projected_cov = self.project(mean, covariance)

        # 矩阵分解
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)

        # 计算卡尔曼增益K
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T

        # z - Hx'
        innovation = measurement - projected_mean

        # x = x' + Ky
        new_mean = mean + np.dot(innovation, kalman_gain.T)

        # P = (I - KH)P'
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        # 计算状态分布和测量值之间的门控距离
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance) # mean.shape(4, ) covariance.shape(4, 4)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance) # shape(4, 4)
        d = measurements - mean # shape(N, 4)
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True) # shape(4, N)
        squared_maha = np.sum(z * z, axis=0) # 马氏距离的平方，平方符合卡方分布
        return squared_maha  # 马氏距离
