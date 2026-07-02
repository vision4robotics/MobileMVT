"""Uncertainty-aware Kalman tracker for multi-vessel tracking.

This module implements the motion-modeling core of the Uncertainty-Aware
Viewpoint Compensation (UAVC) mechanism described in the paper. Each track is
modeled by a constant-acceleration Kalman filter on the ground plane, with an
adaptive process-noise scheme: the process noise is inflated in proportion to a
running estimate of the residual energy, so that the filter relaxes its motion
assumption during abrupt, UAV-induced maneuvers and tightens it again once the
motion settles.

State vector (dim_x = 6):  [x, vx, ax, y, vy, ay]
Measurement vector (dim_z = 2):  [x, y]  (metric ground-plane location)
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from enum import Enum


class TrackStatus(Enum):
    """Lifecycle state of a track."""
    Tentative = 0   # newly created, not yet confirmed
    Confirmed = 1   # stably associated across frames
    Coasted = 2     # temporarily unmatched, propagated by motion only


class KalmanTracker:
    """Single-target constant-acceleration Kalman filter with adaptive noise.

    Args:
        y: Initial ground-plane measurement, shape (2, 1) or (2,).
        R: Measurement-noise covariance, shape (2, 2).
        wx, wy: Base process-noise spectral densities along the x / y axes.
        vmax: Maximum expected speed, used to initialize the velocity variance.
        w, h: Initial bounding-box width and height (kept for visualization).
        dt: Time step between frames (seconds). Defaults to 1/30.
        lambda_: Sensitivity of the adaptive process-noise scaling.
        ema_alpha: Smoothing factor of the residual-energy EMA in [0, 1).
    """

    count = 1

    def __init__(self, y, R, wx, wy, vmax, w, h, dt=1 / 30, lambda_=0.05, ema_alpha=0.9):
        self.kf = KalmanFilter(dim_x=6, dim_z=2)

        # State-transition matrix (constant-acceleration model, x and y blocks).
        self.kf.F = np.array([
            [1, dt, 0.5 * dt * dt, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt * dt],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ])

        # Measurement matrix: observe position (x, y) only.
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ])

        self.kf.R = R

        # Initial state covariance: zero position uncertainty, vmax-derived
        # velocity uncertainty (acceleration entries left at zero).
        self.kf.P = np.zeros((6, 6))
        np.fill_diagonal(self.kf.P, np.array([1, vmax ** 2 / 3.0, 1, vmax ** 2 / 3.0]))

        # Base process-noise covariance Q0 = G Q G^T, with G mapping the
        # per-axis noise onto the [position, velocity, acceleration] state.
        G = np.zeros((6, 2))
        G[0, 0] = 0.5 * dt * dt
        G[1, 0] = dt
        G[2, 0] = 1
        G[3, 1] = 0.5 * dt * dt
        G[4, 1] = dt
        G[5, 1] = 1
        Q0 = np.array([[wx, 0], [0, wy]])
        self.base_Q = np.dot(np.dot(G, Q0), G.T)
        self.kf.Q = self.base_Q.copy()

        # Initialize state from the first measurement; velocities/accelerations 0.
        self.kf.x[0] = y[0]
        self.kf.x[1] = 0
        self.kf.x[2] = 0
        self.kf.x[3] = y[1]
        self.kf.x[4] = 0
        self.kf.x[5] = 0

        # Track bookkeeping.
        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1
        self.w = w
        self.h = h
        self.status = TrackStatus.Tentative

        # Adaptive process-noise parameters and running residual-energy estimate.
        self.lambda_ = lambda_
        self.ema_alpha = ema_alpha
        self.r_squared_ema = 0

    def update(self, y, R):
        """Correct the state with a new measurement and adapt the process noise.

        The squared residual energy is smoothed by an EMA and used to inflate the
        base process noise, so the filter loosens its motion prior under large,
        maneuver-induced residuals.
        """
        residual = y - np.dot(self.kf.H, self.kf.x)
        r_norm_sq = np.linalg.norm(residual) ** 2
        self.r_squared_ema = self.ema_alpha * self.r_squared_ema + (1 - self.ema_alpha) * r_norm_sq
        alpha = 1 + self.lambda_ * self.r_squared_ema
        self.kf.Q = alpha * self.base_Q
        self.kf.update(y, R)

    def predict(self):
        """Advance the state by one time step and return the predicted (x, y)."""
        self.kf.predict()
        self.age += 1
        return np.dot(self.kf.H, self.kf.x)

    def get_state(self):
        """Return the full 6-D state vector."""
        return self.kf.x

    def distance(self, y, R):
        """Gated association cost: Mahalanobis distance plus log-determinant term.

        Combines the squared Mahalanobis distance between the measurement and the
        predicted position with the innovation-covariance log-determinant, giving
        a statistically consistent matching cost for the Hungarian assignment.
        """
        diff = y - np.dot(self.kf.H, self.kf.x)
        S = np.dot(self.kf.H, np.dot(self.kf.P, self.kf.H.T)) + R
        SI = np.linalg.inv(S)
        mahalanobis = np.dot(diff.T, np.dot(SI, diff))
        logdet = np.log(np.linalg.det(S))
        return mahalanobis[0, 0] + logdet
