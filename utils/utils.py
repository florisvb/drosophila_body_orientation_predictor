"""
utils.py — Drosophila Body Orientation Predictor Utilities

Consolidated utility module for the full prediction pipeline:
  - Data loading and merging (HDF5 → CSV)
  - Feature augmentation (groundspeed, airspeed, thrust, etc.)
  - Heading correction (naive unwrap and convex-optimization variants)
  - Trajectory filtering and smoothing
  - Time-delay embedding for the neural network
  - Model creation
  - Visualization
  - General data utilities
"""

# ============================================================
# Imports
# ============================================================
import copy
import datetime

import cvxpy
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynumdiff as pynd
import scipy
import scipy.stats
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

import fly_plot_lib_plot as fpl


# ============================================================
# Colormaps
# ============================================================

blue_cmap   = mcolors.LinearSegmentedColormap.from_list('blue_cmap',   ['blue',       'blue'])
red_cmap    = mcolors.LinearSegmentedColormap.from_list('red_cmap',    ['red',        'red'])
orange_cmap = mcolors.LinearSegmentedColormap.from_list('orange_cmap', ['darkorange', 'darkorange'])


# ============================================================
# Angle & Math Utilities
# ============================================================

def wrapToPi(rad):
    """Wrap angle(s) in radians to the range [-π, π]."""
    rad_wrap = copy.copy(rad)
    q = (rad_wrap < -np.pi) | (np.pi < rad_wrap)
    rad_wrap[q] = ((rad_wrap[q] + np.pi) % (2 * np.pi)) - np.pi
    return rad_wrap


def wrapTo2Pi(rad):
    """Wrap angle(s) in radians to the range [0, 2π)."""
    rad = copy.copy(rad)
    rad = rad % (2 * np.pi)
    return rad


def polar2cart(r, theta):
    """Convert polar coordinates (r, θ) to Cartesian (x, y)."""
    return r * np.cos(theta), r * np.sin(theta)


def cart2polar(x, y):
    """Convert Cartesian coordinates (x, y) to polar (r, θ)."""
    return np.sqrt(x**2 + y**2), np.arctan2(y, x)


def circular_distance(angle1, angle2):
    """
    Compute the shortest angular distance between two angles on a circle.

    Works element-wise for scalars, arrays, or pandas Series.

    Parameters
    ----------
    angle1, angle2 : float or array-like
        Angles in radians.

    Returns
    -------
    float or np.ndarray
        Shortest distance in radians, always in [0, π].
    """
    angle1 = np.asarray(angle1) % (2 * np.pi)
    angle2 = np.asarray(angle2) % (2 * np.pi)
    direct = np.abs(angle1 - angle2)
    return np.minimum(direct, 2 * np.pi - direct)


def log_scale_with_negatives(x, epsilon=2.0, inverse=False):
    """
    Transform values to log-scale while preserving the sign of negatives.

    Adds epsilon before log to prevent issues with values < 1. If inverse=True,
    returns 1/result.
    """
    x_negative_idx = x < 0
    y = x.copy()
    y[~x_negative_idx] = np.log(epsilon + y[~x_negative_idx])
    y[x_negative_idx] = -np.log(epsilon + -y[x_negative_idx])
    if inverse:
        y = 1 / y
    return y


# ============================================================
# Data Loading & Merging
# ============================================================

def correct_for_wind(trajec_df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct airvelocity_x for the constant wind offset.

    The raw HDF5 data has airvelocity_x in the lab frame; subtracting
    2 × wind_speed converts it to be wind-relative.

    Parameters
    ----------
    trajec_df : pd.DataFrame
        Raw trajectory DataFrame with 'airvelocity_x' and 'wind_speed' columns.

    Returns
    -------
    pd.DataFrame
        Copy with corrected 'airvelocity_x'.
    """
    corrected = trajec_df.copy()
    corrected["airvelocity_x"] = corrected["airvelocity_x"] - 2 * corrected["wind_speed"]
    return corrected


def remove_irrelevant_trajectory_data(trajec_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unused columns from the raw trajectory DataFrame and standardize names.

    Drops: course, frame, odor, odor_stimulus, position_z, velocity_z,
    groundspeed_xy, airspeed_xy.
    Renames: objid → trajec_objid, wind_speed → windspeed,
             wind_direction → windspeed_angle.
    """
    trimmed = trajec_df.copy().drop(
        ["course", "frame", "odor", "odor_stimulus", "position_z", "velocity_z",
         "groundspeed_xy", "airspeed_xy"], axis=1
    )
    trimmed = trimmed[['objid', 'timestamp', 'position_x', 'position_y',
                        'velocity_x', 'velocity_y', 'airvelocity_x', 'airvelocity_y',
                        "wind_speed", "wind_direction"]]
    return trimmed.rename(columns={
        'objid': 'trajec_objid',
        'wind_speed': 'windspeed',
        'wind_direction': 'windspeed_angle',
    })


def remove_irrelevant_body_data(body_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unused columns from the raw body-orientation DataFrame and standardize names.

    Drops: date, frame, longaxis_0, longaxis_1, position_x, position_y.
    Renames: angle → ellipse_short_angle.
    """
    trimmed = body_df.copy().drop(
        ["date", "frame", "longaxis_0", "longaxis_1", "position_x", "position_y"], axis=1
    )
    trimmed = trimmed[['body_objid', 'timestamp', 'eccentricity', 'angle']]
    return trimmed.rename(columns={'angle': 'ellipse_short_angle'})


def sync_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round timestamps to two decimal places (SS.ss) for DataFrame merging.

    Converts raw UNIX timestamps to within-second values so trajectory and
    body DataFrames can be merged on 'timestamp'.
    """
    df_synced = df.copy()
    df_synced['timestamp'] = df['timestamp'].apply(
        lambda x: int(datetime.datetime.utcfromtimestamp(x).strftime('%S')) +
                  float(datetime.datetime.utcfromtimestamp(x).strftime('%f')[0:2]) / 100
    )
    return df_synced


def join_all_body_and_trajec_df(
    synced_body: pd.DataFrame,
    synced_trajectory: pd.DataFrame,
    key_table: pd.DataFrame,
) -> list:
    """
    Merge body-orientation and trajectory DataFrames for every unique trajectory ID.

    Uses key_table to match each trajec_objid to its body_objid, then
    inner-joins on 'timestamp'.

    Returns
    -------
    list of pd.DataFrame
        One merged DataFrame per unique trajectory.
    """
    def _join_one(trajec_id):
        fly_trajectory = synced_trajectory[synced_trajectory['trajec_objid'] == trajec_id]
        body_id = key_table[key_table['trajec_objid'] == trajec_id]['body_objid'].values[0]
        fly_body = synced_body[synced_body['body_objid'] == body_id]
        return pd.merge(fly_trajectory, fly_body, on='timestamp', how='inner').drop(['body_objid'], axis=1)

    return [_join_one(tid) for tid in synced_trajectory['trajec_objid'].unique()]


def transform_timestamps_to_start_at_zero(fly_trajectory_and_body: pd.DataFrame) -> pd.DataFrame:
    """
    Shift timestamps so the trajectory starts at t=0, rounded to 2 decimal places.

    Mutates the input DataFrame in place and also returns it.
    """
    fly_trajectory_and_body["timestamp"] = (
        fly_trajectory_and_body["timestamp"] - fly_trajectory_and_body["timestamp"][0]
    )
    fly_trajectory_and_body["timestamp"] = fly_trajectory_and_body["timestamp"].round(2)
    return fly_trajectory_and_body


# ============================================================
# Data Augmentation
# ============================================================

def augment_fly_trajectory(
    fly_trajectory_and_body: pd.DataFrame,
    compute_heading_from_ellipses: bool = True,
) -> pd.DataFrame:
    """
    Augment a fly trajectory DataFrame with derived physical quantities.

    Computes and appends:
      - groundspeed, groundspeed_angle
      - airspeed, airspeed_angle
      - linear_acceleration, linear_acceleration_angle
      - thrust, thrust_angle
      - heading_angle (from ellipse_short_angle, if compute_heading_from_ellipses=True)

    Uses a Savitzky-Golay differentiator (params=[2, 10, 10]) for acceleration.

    Parameters
    ----------
    fly_trajectory_and_body : pd.DataFrame
        Must have velocity_x, velocity_y, airvelocity_x, airvelocity_y,
        timestamp columns. Also needs ellipse_short_angle if
        compute_heading_from_ellipses=True.
    compute_heading_from_ellipses : bool
        Set to False when heading_angle is already provided (e.g., David's data).

    Returns
    -------
    pd.DataFrame
        Copy with augmented columns added.
    """
    aug = fly_trajectory_and_body.copy()

    gnd_vx, gnd_vy = aug["velocity_x"], aug["velocity_y"]
    aug["groundspeed"] = np.sqrt(gnd_vx**2 + gnd_vy**2)
    aug["groundspeed_angle"] = np.arctan2(gnd_vy, gnd_vx)

    air_vx, air_vy = aug["airvelocity_x"], aug["airvelocity_y"]
    aug["airspeed"] = np.sqrt(air_vx**2 + air_vy**2)
    aug["airspeed_angle"] = np.arctan2(air_vy, air_vx)

    params = [2, 10, 10]
    dt = np.median(np.diff(aug.timestamp))
    _, accel_x = pynd.linear_model.savgoldiff(aug["velocity_x"], dt, params)
    _, accel_y = pynd.linear_model.savgoldiff(aug["velocity_y"], dt, params)
    aug["linear_acceleration"] = np.sqrt(accel_x**2 + accel_y**2)
    aug["linear_acceleration_angle"] = np.arctan2(accel_y, accel_x)

    if compute_heading_from_ellipses:
        def _heading_from_ellipse(angle):
            # ±π/2 rotation depending on ellipse angle sign.
            # angle==0 defaults to downward heading; corrected by heading correction later.
            angle = np.where(angle < 0, angle - np.pi / 2, angle)
            angle = np.where(angle > 0, angle + np.pi / 2, angle)
            angle = np.where(angle == 0, angle - np.pi / 2, angle)
            return angle
        aug["heading_angle"] = _heading_from_ellipse(aug["ellipse_short_angle"])

    mass = 0.25e-6
    dragcoeff = mass / 0.170
    thrust_x = mass * accel_x + dragcoeff * air_vx
    thrust_y = mass * accel_y + dragcoeff * air_vy
    aug["thrust"] = np.sqrt(thrust_x**2 + thrust_y**2)
    aug["thrust_angle"] = np.arctan2(thrust_y, thrust_x)

    return aug


class compute:
    """
    Static methods for computing individual derived kinematic quantities.

    These return arrays rather than augmented DataFrames — use
    augment_fly_trajectory() for the full augmentation pipeline.
    """

    @staticmethod
    def angular_velocity(fly_df: pd.DataFrame, name_of_heading_field: str, name_of_time_field: str):
        """Compute angular velocity via Savitzky-Golay differentiation."""
        params = [2, 10, 10]
        dt = np.median(np.diff(fly_df[name_of_time_field]))
        _, ang_vel = pynd.linear_model.savgoldiff(fly_df[name_of_heading_field], dt, params)
        return ang_vel

    @staticmethod
    def angular_acceleration(fly_df: pd.DataFrame, name_of_heading_field: str, name_of_time_field: str):
        """Compute angular acceleration via Savitzky-Golay differentiation of angular velocity."""
        params = [2, 10, 10]
        dt = np.median(np.diff(fly_df[name_of_time_field]))
        _, ang_accel = pynd.linear_model.savgoldiff(
            compute.angular_velocity(fly_df, name_of_heading_field, name_of_time_field), dt, params
        )
        return ang_accel

    @staticmethod
    def linear_acceleration(fly_trajectory_and_body: pd.DataFrame):
        """
        Compute linear acceleration (x, y) via Savitzky-Golay differentiation of velocity.

        Returns
        -------
        accel_x, accel_y : np.ndarray
        """
        params = [2, 10, 10]
        dt = np.median(np.diff(fly_trajectory_and_body.timestamp))
        _, accel_x = pynd.linear_model.savgoldiff(fly_trajectory_and_body.velocity_x.values, dt, params)
        _, accel_y = pynd.linear_model.savgoldiff(fly_trajectory_and_body.velocity_y.values, dt, params)
        return accel_x, accel_y

    @staticmethod
    def thrust(fly_trajectory_and_body: pd.DataFrame, mass: float = 0.25e-6):
        """
        Compute thrust force (x, y) from kinematics.

        thrust = mass × acceleration + drag_coefficient × airvelocity,
        where drag_coefficient = mass / 0.170.

        Returns
        -------
        thrust_x, thrust_y : np.ndarray
        """
        dragcoeff = mass / 0.170
        accel_x, accel_y = compute.linear_acceleration(fly_trajectory_and_body)
        thrust_x = mass * accel_x + dragcoeff * fly_trajectory_and_body.airvelocity_x.values
        thrust_y = mass * accel_y + dragcoeff * fly_trajectory_and_body.airvelocity_y.values
        return thrust_x, thrust_y

    @staticmethod
    def heading_angle_corrected(
        fly_trajectory_and_body: pd.DataFrame,
        name_of_heading_field: str,
        name_of_airspeed_field: str,
    ):
        """
        Correct 180° heading ambiguities using np.unwrap (no MOSEK required).

        Aligns heading with the reference direction, unwraps π-flips, then
        wraps to [-π, π].

        Parameters
        ----------
        name_of_heading_field : str
            Column name for the heading angle (radians).
        name_of_airspeed_field : str
            Column name for the reference direction (e.g., 'groundspeed_angle').

        Returns
        -------
        np.ndarray
            Corrected heading angles.
        """
        traj = fly_trajectory_and_body.copy()
        angle = traj[name_of_heading_field]
        reference = traj[name_of_airspeed_field]
        initial_window = 5
        circ_diff_start = circular_distance(
            scipy.stats.circmean(reference[0:initial_window], low=-np.pi, high=np.pi),
            scipy.stats.circmean(angle[0:initial_window], low=-np.pi, high=np.pi),
        )
        if circ_diff_start > 0.5 * np.pi:
            angle = angle + np.pi * np.sign(circ_diff_start)
        corrected = np.unwrap(angle, period=np.pi, discont=0.5 * np.pi)
        phi_mean = scipy.stats.circmean(wrapToPi(corrected), low=-np.pi, high=np.pi)
        psi_mean = scipy.stats.circmean(wrapToPi(reference), low=-np.pi, high=np.pi)
        if circular_distance(phi_mean, psi_mean) > 0.5 * np.pi:
            corrected = corrected + np.pi * np.sign(circular_distance(phi_mean, psi_mean))
        return wrapToPi(corrected)

    @staticmethod
    def heading_angle_convex_opt(fly_trajectory_and_body: pd.DataFrame):
        """
        Correct 180° heading ambiguities via convex optimization (requires MOSEK).

        Minimizes total variation of the corrected heading plus deviation from
        the thrust direction.

        Returns
        -------
        np.ndarray
            Corrected heading angles wrapped to [-π, π].
        """
        def _angle_diff(alpha, beta):
            return (alpha - beta + np.pi) % (2 * np.pi) - np.pi

        k = cvxpy.Variable(len(fly_trajectory_and_body), integer=True)
        heading_angle = fly_trajectory_and_body.heading_angle.values
        thrust_x, thrust_y = compute.thrust(fly_trajectory_and_body)
        thrust_angle = np.arctan2(thrust_y, thrust_x)
        avg_diff = np.average(_angle_diff(heading_angle, thrust_angle))
        L = cvxpy.tv(heading_angle + np.pi * k) + cvxpy.norm1(avg_diff + np.pi * k)
        cvxpy.Problem(cvxpy.Minimize(L), [-1 <= k, k <= 1]).solve(solver='MOSEK')
        return np.arctan2(np.sin(heading_angle + k.value * np.pi),
                          np.cos(heading_angle + k.value * np.pi))

    @staticmethod
    def heading_angle_from_ellipse(angle):
        """
        Derive heading angle from the short-axis ellipse angle (adds ±π/2).

        Parameters
        ----------
        angle : array-like
            Ellipse short-axis angles in radians.

        Returns
        -------
        np.ndarray
        """
        angle = np.where(angle < 0, angle - np.pi / 2, angle)
        angle = np.where(angle > 0, angle + np.pi / 2, angle)
        return angle


# ============================================================
# Heading Correction
# ============================================================

def naive_heading_correction(fly_trajectory_and_body: pd.DataFrame) -> pd.DataFrame:
    """
    Correct 180° heading ambiguities in-place using np.unwrap (no MOSEK required).

    Algorithm:
      1. Align the initial heading with the groundspeed direction.
      2. Use np.unwrap with period=π to detect and remove π-flips.
      3. Align mean corrected heading with mean groundspeed direction.
      4. Wrap result to [-π, π].

    Parameters
    ----------
    fly_trajectory_and_body : pd.DataFrame
        Must have 'heading_angle' and 'groundspeed_angle' columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'heading_angle' corrected (mutated in place).
    """
    angle = fly_trajectory_and_body["heading_angle"].copy()
    course = fly_trajectory_and_body["groundspeed_angle"]
    initial_window = 5

    circ_diff_start = circular_distance(
        scipy.stats.circmean(course[0:initial_window], low=-np.pi, high=np.pi),
        scipy.stats.circmean(angle[0:initial_window], low=-np.pi, high=np.pi),
    )
    if circ_diff_start > 0.5 * np.pi:
        angle = angle + np.pi * np.sign(circ_diff_start)

    corrected = np.unwrap(angle, period=np.pi, discont=0.5 * np.pi)
    phi_mean = scipy.stats.circmean(wrapToPi(corrected), low=-np.pi, high=np.pi)
    psi_mean = scipy.stats.circmean(wrapToPi(course), low=-np.pi, high=np.pi)
    circ_diff = circular_distance(phi_mean, psi_mean)
    if circ_diff > 0.5 * np.pi:
        corrected = corrected + np.pi * np.sign(circ_diff)

    fly_trajectory_and_body["heading_angle"] = wrapToPi(corrected)
    return fly_trajectory_and_body


def convex_opt_heading_correction(fly_trajectory_and_body: pd.DataFrame) -> pd.DataFrame:
    """
    Correct 180° heading ambiguities via convex optimization (requires MOSEK).

    Minimizes total variation of the corrected heading (smoothness term, weight=1)
    plus thrust-magnitude-weighted alignment to the thrust direction (weight=5).
    The heading is unwrapped before optimization so the TV term correctly measures
    angular change across the ±π boundary.

    Parameters
    ----------
    fly_trajectory_and_body : pd.DataFrame
        Must have 'heading_angle', 'thrust_angle', and 'thrust' columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'heading_angle' corrected (mutated in place).
    """
    def _angle_diff(alpha, beta):
        return (alpha - beta + np.pi) % (2 * np.pi) - np.pi

    k = cvxpy.Variable(len(fly_trajectory_and_body), integer=True)
    heading_angle = fly_trajectory_and_body.heading_angle.values
    heading_angle_unwrapped = np.unwrap(heading_angle)
    thrust_angle = fly_trajectory_and_body.thrust_angle.values
    thrust_mag = fly_trajectory_and_body.thrust.values
    thrust_mag_norm = thrust_mag / thrust_mag.mean()
    delta = _angle_diff(heading_angle, thrust_angle)

    L = 1 * cvxpy.tv(heading_angle_unwrapped + np.pi * k) + 5 * cvxpy.norm1(cvxpy.multiply(thrust_mag_norm, delta + np.pi * k))
    cvxpy.Problem(cvxpy.Minimize(L), [-1 <= k, k <= 1]).solve(solver='MOSEK')

    fly_trajectory_and_body["heading_angle"] = np.arctan2(
        np.sin(heading_angle_unwrapped + k.value * np.pi),
        np.cos(heading_angle_unwrapped + k.value * np.pi),
    )
    return fly_trajectory_and_body


# ============================================================
# Trajectory Filtering
# ============================================================

def calc_circular_difference(trajec: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'circ_dist_diff' column: circular distance between consecutive heading angles.

    Used to identify trajectories with large abrupt heading jumps (residual
    uncorrected 180° flips).

    Parameters
    ----------
    trajec : pd.DataFrame
        Must have a 'heading_angle' column.

    Returns
    -------
    pd.DataFrame
        Copy with 'circ_dist_diff' column added.
    """
    out = trajec.copy()
    out["circ_dist_diff"] = circular_distance(
        out["heading_angle"],
        out["heading_angle"].shift(1, fill_value=out["heading_angle"].iloc[0]),
    )
    return out


def pull_out_individual_trajectories(
    all_fly_data: pd.DataFrame,
    min_traj_length: int = 5,
) -> list:
    """
    Split a concatenated trajectory DataFrame into individual trajectory DataFrames.

    Trajectories are identified by their timestamp resetting to 0.00. Only
    trajectories with more than min_traj_length rows are returned.

    Parameters
    ----------
    all_fly_data : pd.DataFrame
        Concatenated data where each trajectory starts with timestamp == 0.00.
    min_traj_length : int
        Minimum number of timesteps required to include a trajectory.

    Returns
    -------
    list of pd.DataFrame
    """
    traj_start = np.where(all_fly_data.timestamp.values == 0.00)[0]
    n_traj = traj_start.shape[0]
    print(f"Detected {n_traj} trajectories.")

    traj_list = []
    for n in range(n_traj):
        traj_end = all_fly_data.shape[0] if n == (n_traj - 1) else traj_start[n + 1]
        traj = all_fly_data.iloc[traj_start[n]:traj_end, :]
        if traj.shape[0] > min_traj_length:
            traj_list.append(traj)

    print(f"{len(traj_list)} trajectories with more than {min_traj_length} timesteps.")
    return traj_list


# ============================================================
# Trajectory Smoothing
# ============================================================

def smooth_trajectory(trajectory: pd.DataFrame, savgol_params: list = None) -> pd.DataFrame:
    """
    Smooth 'heading_angle' using Savitzky-Golay differentiation.

    Internally unwraps the angle to a continuous signal before smoothing,
    then stores the result back in 'heading_angle'.

    Parameters
    ----------
    trajectory : pd.DataFrame
        Must have a 'heading_angle' column.
    savgol_params : list, optional
        [order, window, iter] for pynd.savgoldiff. Default: [1, 5, 5].

    Returns
    -------
    pd.DataFrame
        Copy with 'heading_angle' replaced by the smoothed version.
    """
    if savgol_params is None:
        savgol_params = [1, 5, 5]

    def _unwrap_angle(z, correction_window_for_2pi=100, n_range=2):
        """Unwrap angle z to a continuous signal, correcting 2π jumps."""
        smooth_zs = np.array(z[0:2])
        for i in range(2, len(z)):
            first_ix = max(0, i - correction_window_for_2pi)
            nbase = np.round((smooth_zs[-1] - z[i]) / (2 * np.pi))
            candidates = [n * 2 * np.pi + nbase * 2 * np.pi + z[i] for n in range(-n_range, n_range)]
            error = np.abs(candidates - np.mean(smooth_zs[first_ix:i]))
            smooth_zs = np.hstack((smooth_zs, [candidates[np.argmin(error)]]))
        return smooth_zs

    out = trajectory.copy()
    heading_angle = np.array(out["heading_angle"])
    smooth_heading_angle, _ = pynd.savgoldiff(_unwrap_angle(heading_angle), dt=0.01, params=savgol_params)
    out["heading_angle"] = smooth_heading_angle
    return out


# ============================================================
# Time-Delay Embedding
# ============================================================

def collect_offset_rows(
    df: pd.DataFrame,
    aug_column_names=None,
    keep_column_names=None,
    w: int = 1,
    direction: str = 'backward',
) -> pd.DataFrame:
    """
    Create a time-delay embedding by stacking offset rows as new columns.

    For each column in aug_column_names, creates w new columns named
    '<col>_0', '<col>_1', ..., '<col>_{w-1}' using a sliding window.
    The result has (n - w + 1) rows.

    Parameters
    ----------
    df : pd.DataFrame
    aug_column_names : list of str, optional
        Columns to augment. Defaults to all columns.
    keep_column_names : list of str, optional
        Columns to include unchanged (not augmented).
    w : int
        Window size.
    direction : str
        'backward' for lookback window, 'forward' for lookahead.

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with (n - w + 1) rows.
    """
    df = df.reset_index(drop=True)
    if aug_column_names is None:
        aug_column_names = df.columns

    new_column_names = {a: [f"{a}_{k}" for k in range(w)] for a in aug_column_names}
    n_row = df.shape[0]
    n_row_train = n_row - w + 1
    df_aug_dict = {}

    for a in aug_column_names:
        data = np.asmatrix(df.loc[:, [a]])
        mat = np.nan * np.ones((n_row_train, w))
        for i in range(w):
            if direction == 'backward':
                startI, endI = w - 1 - i, n_row - i
            elif direction == 'forward':
                startI, endI = i, n_row - w + 1 + i
            else:
                raise ValueError("direction must be 'forward' or 'backward'")
            mat[:, i] = np.squeeze(data[startI:endI, :])
        df_aug_dict[a] = pd.DataFrame(mat, columns=new_column_names[a])

    df_aug = pd.concat(list(df_aug_dict.values()), axis=1)

    if keep_column_names is not None:
        for c in keep_column_names:
            if direction == 'backward':
                startI, endI = w - 1, n_row
            elif direction == 'forward':
                startI, endI = 0, n_row - w
            else:
                raise ValueError("direction must be 'forward' or 'backward'")
            keep = df.loc[startI:endI, [c]].reset_index(drop=True)
            df_aug = pd.concat([df_aug, keep], axis=1)

    return df_aug


def augment_with_time_delay_embedding(fly_traj_list: list, wind_augment: bool = False, wrap_angles: bool = False, **kwargs) -> pd.DataFrame:
    """
    Apply time-delay embedding to a list of trajectory DataFrames.

    Calls collect_offset_rows on each trajectory and concatenates the results.

    Expected kwargs
    ---------------
    time_window : int
        Number of timesteps in the window.
    input_names : list of str
        Columns to use as model inputs (will be augmented).
    output_names : list of str
        Columns to keep as model outputs (not augmented).
    direction : str
        'backward' or 'forward'.

    Parameters
    ----------
    wind_augment : bool, optional
        If True, apply random wind-direction rotation to each trajectory's
        angle columns (heading_angle, groundspeed_angle, airspeed_angle,
        thrust_angle) before embedding. This simulates multiple wind
        directions and is used to train model_CEM_all-angle-rotate.keras.
        Default is False.
    wrap_angles : bool, optional
        If True, wrap all rotated angle columns back to [-pi, pi] after
        adding the rotation. Only applies when wind_augment=True. Without
        wrapping, rotated angles can range up to 3*pi, which is
        out-of-distribution relative to inference-time inputs. Default is False.

    Returns
    -------
    pd.DataFrame
        Concatenated augmented DataFrame rounded to 4 decimal places.
    """
    time_window = kwargs["time_window"]
    input_names = kwargs["input_names"]
    output_names = kwargs["output_names"]
    direction = kwargs["direction"]

    angle_cols = ['heading_angle', 'groundspeed_angle', 'airspeed_angle', 'thrust_angle']

    rotations = []
    if wind_augment:
        rng = np.random.default_rng(29464)
        rotations = list(rng.random((len(fly_traj_list),)) * 2 * np.pi)

    traj_augment_list = []
    for i, traj in enumerate(fly_traj_list):
        if wind_augment:
            traj = traj.copy()
            for col in angle_cols:
                traj[col] = traj[col] + rotations[i]
                if wrap_angles:
                    traj[col] = np.arctan2(np.sin(traj[col]), np.cos(traj[col]))
            traj['heading_angle_x'] = np.cos(traj['heading_angle'])
            traj['heading_angle_y'] = np.sin(traj['heading_angle'])

        traj_augment_list.append(
            collect_offset_rows(traj,
                                aug_column_names=input_names,
                                keep_column_names=output_names,
                                w=time_window,
                                direction=direction)
        )

    return np.round(pd.concat(traj_augment_list, ignore_index=True), 4)


def sliding_window(df, slide=None, w=None, n_window_limit=None, seed=None, aug_column_names=None):
    """
    Extract windowed samples from a DataFrame at fixed or random start points.

    Unlike collect_offset_rows (one row per timestep), sliding_window produces
    one row per window.

    Parameters
    ----------
    df : pd.DataFrame
    slide : int or None
        Step size between windows. If None, windows are placed randomly.
    w : int
        Window size (number of rows per window).
    n_window_limit : int or None
        Required when slide=None; number of random windows to draw.
    seed : int or None
        Random seed (used when slide=None).
    aug_column_names : list of str, optional
        Columns to include. Defaults to all columns.

    Returns
    -------
    df_aug_all : pd.DataFrame
    df_list : list of pd.DataFrame
        Raw window DataFrames (one per window).
    """
    df = df.reset_index(drop=True)
    if aug_column_names is None:
        aug_column_names = df.columns

    n_points = df.shape[0]
    n_possible = n_points - w + 1

    if slide is None:
        assert n_window_limit is not None, '"n_window_limit" must not be None when "slide"=None'
        assert n_window_limit < n_possible, '"n_window_limit" must be less than # rows minus window size'
        if n_window_limit > 0.9 * n_possible:
            print('You are using 90% of the possible start points, do you really need to be doing this randomly?')
        np.random.seed(seed=seed)
        window_start = np.unique(np.random.randint(0, high=n_possible, size=n_window_limit, dtype=int))
        n = 1
        while window_start.shape[0] < n_window_limit:
            window_start = np.unique(np.hstack((window_start, np.random.randint(0, high=n_possible, size=1, dtype=int))))
            n += 1
    else:
        window_start = np.arange(0, n_possible, slide, dtype=int)

    window_indices = np.zeros((window_start.shape[0], w), dtype=int)
    for r, ws in enumerate(window_start):
        window_indices[r, :] = np.arange(ws, ws + w, 1, dtype=int)

    df_list = [df.loc[window_indices[r, :], aug_column_names] for r in range(window_indices.shape[0])]
    new_column_names = {a: [f"{a}_{k}" for k in range(w)] for a in aug_column_names}

    df_aug_list = []
    for df_window in df_list:
        df_aug = []
        for cname in df_window.columns:
            var_aug = df_window.loc[:, [cname]].T
            var_aug.columns = new_column_names[cname]
            df_aug.append(var_aug)
        temp = pd.concat(df_aug, axis=1, ignore_index=False)
        df_aug = pd.DataFrame(np.concatenate(df_aug, axis=1), columns=temp.columns)
        df_aug.index = df_window.index[0:1]
        df_aug_list.append(df_aug)

    return pd.concat(df_aug_list, axis=0), df_list


# ============================================================
# Neural Network
# ============================================================

def create_model(n_input: int, n_output: int, neurons: int = 50, layers: int = 1):
    """
    Build a fully-connected Keras model for heading prediction.

    Architecture: Dense(relu) × layers → Dense(linear) → L2-normalisation output.
    The L2-normalisation layer hard-constrains outputs to the unit circle.
    Loss: mean_squared_error. Optimizer: Adam.

    Parameters
    ----------
    n_input : int
        Number of input features.
    n_output : int
        Number of output units (2 for heading_angle_x, heading_angle_y).
    neurons : int
        Neurons per hidden layer.
    layers : int
        Number of hidden layers.

    Returns
    -------
    keras.Sequential
    """
    from keras.models import Sequential
    from keras.layers import Dense, UnitNormalization

    model = Sequential()
    model.add(Dense(neurons, input_dim=n_input, activation='relu'))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(n_output, activation='linear'))
    model.add(UnitNormalization(axis=-1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model


# ============================================================
# Visualization
# ============================================================

def plot_trajectory(
    ax,
    fly_trajectory_and_body: pd.DataFrame,
    plot_ellipses: bool = True,
    every_nth: int = 4,
    L: float = 0.008,
    legend: bool = True,
    heading_color: str = 'blue',
    heading_size: float = 0.5,
    heading_alpha: float = 0.8,
    show_groundspeed: bool = False,
    arrow_scale: float = 1.0,
    arrow_width: float = 0.03,
):
    """
    Plot a fly trajectory with quiver arrows for heading, air velocity, and thrust.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    fly_trajectory_and_body : pd.DataFrame
        Must have: position_x, position_y, velocity_x, velocity_y,
        heading_angle, airvelocity_x, airvelocity_y, thrust_angle, thrust,
        trajec_objid, and (if plot_ellipses=True) eccentricity,
        ellipse_short_angle.
    plot_ellipses : bool
        If True, draw body ellipses at each sampled point.
    every_nth : int
        Downsample factor for quiver/ellipse plotting.
    L : float
        Major axis length for ellipses (plot units).
    legend : bool
    show_groundspeed : bool
        If True, draw groundspeed direction quiver arrows.
    """
    ax.set_aspect('equal')
    padding = 0.01
    ax.set_xlim(np.nanmin(fly_trajectory_and_body['position_x']) - padding,
                np.nanmax(fly_trajectory_and_body['position_x']) + padding)
    ax.set_ylim(np.nanmin(fly_trajectory_and_body['position_y']) - padding,
                np.nanmax(fly_trajectory_and_body['position_y']) + padding)
    ax.plot(fly_trajectory_and_body['position_x'].values,
            fly_trajectory_and_body['position_y'].values,
            color='red', label='Trajectory', zorder=1)

    if plot_ellipses:
        for ix in range(0, len(fly_trajectory_and_body), every_nth):
            ellipse = matplotlib.patches.Ellipse(
                [fly_trajectory_and_body['position_x'].iloc[ix],
                 fly_trajectory_and_body['position_y'].iloc[ix]],
                L * fly_trajectory_and_body["eccentricity"].iloc[ix],
                L,
                angle=fly_trajectory_and_body["ellipse_short_angle"].iloc[ix] * 180 / np.pi,
                color='black',
            )
            ax.add_artist(ellipse)

    pos_x = fly_trajectory_and_body['position_x'].iloc[::every_nth].values
    pos_y = fly_trajectory_and_body['position_y'].iloc[::every_nth].values
    heading = fly_trajectory_and_body["heading_angle"].iloc[::every_nth].values
    air_vx = fly_trajectory_and_body['airvelocity_x'].iloc[::every_nth].values
    air_vy = fly_trajectory_and_body['airvelocity_y'].iloc[::every_nth].values
    thrust_angle = fly_trajectory_and_body['thrust_angle'].iloc[::every_nth].values
    thrust_mag = fly_trajectory_and_body['thrust'].iloc[::every_nth].values
    gnd_vx = fly_trajectory_and_body['velocity_x'].iloc[::every_nth].values
    gnd_vy = fly_trajectory_and_body['velocity_y'].iloc[::every_nth].values
    gnd_speed = np.sqrt(gnd_vx**2 + gnd_vy**2)
    gnd_speed = np.where(gnd_speed == 0, 1, gnd_speed)

    plot_scale = 0.015 * arrow_scale
    thrust_scale = 10000 * arrow_scale

    # Heading — triangles via fpl.colorline_with_heading
    xymean = np.mean(np.abs(np.hstack((pos_x, pos_y))))
    xymean_scaled = 0.21 * xymean * heading_size
    size_radius = 0.01 * heading_size if xymean_scaled < 0.0001 else float(np.hstack((xymean_scaled, 1))[0])
    heading_cmap = mcolors.LinearSegmentedColormap.from_list('_heading', [heading_color, heading_color])
    color_vals = np.ones(len(pos_x))
    fpl.colorline_with_heading(
        ax, np.flip(pos_x), np.flip(pos_y), np.flip(color_vals), np.flip(heading),
        nskip=0, size_radius=size_radius, deg=False, colormap=heading_cmap,
        center_point_size=0.0001, colornorm=[0, 1], show_centers=False,
        size_angle=20, alpha=heading_alpha, edgecolor='none',
    )
    ax.quiver(pos_x, pos_y, air_vx * plot_scale, air_vy * plot_scale,
              color="blue", angles='xy', scale_units='xy', scale=1, width=arrow_width,
              label="Air velocity")
    ax.quiver(pos_x, pos_y,
              np.cos(thrust_angle) * thrust_mag * thrust_scale,
              np.sin(thrust_angle) * thrust_mag * thrust_scale,
              color="green", angles='xy', scale_units='xy', scale=1, width=arrow_width,
              label="Thrust")
    if show_groundspeed:
        ax.quiver(pos_x, pos_y,
                  (gnd_vx / gnd_speed) * plot_scale,
                  (gnd_vy / gnd_speed) * plot_scale,
                  color="red", angles='xy', scale_units='xy', scale=1, width=arrow_width,
                  label="Groundspeed dir.")

    ax.set_title(fly_trajectory_and_body["trajec_objid"].iloc[0])
    if legend:
        ax.legend()


def plot_trajectory_with_predicted_heading(
    trajectory: pd.DataFrame,
    axis,
    n_input: int,
    best_estimator,
    nskip: int = 0,
    arrow_size=None,
    include_id: bool = False,
    plt_show: bool = False,
    smooth: bool = False,
    **kwargs,
):
    """
    Plot a trajectory with predicted (blue) and true (red) heading arrows overlaid.

    Uses fpl.colorline_with_heading for visualization. Predicted heading is drawn
    under the true heading for comparison.

    Parameters
    ----------
    trajectory : pd.DataFrame
        Must have position_x, position_y, heading_angle, timestamp, and the
        input columns expected by augment_with_time_delay_embedding.
    axis : matplotlib.axes.Axes
    n_input : int
        Number of input features (used to slice the augmented DataFrame columns).
    best_estimator : keras model or sklearn estimator
        Trained model with a .predict() method returning [cos(heading), sin(heading)].
    nskip : int
        Number of arrows to skip between plotted arrows.
    arrow_size : float or list or None
        Size of heading arrows. None = auto-scale.
    include_id : bool
        If True, set axis title to trajectory's trajec_objid.
    plt_show : bool
        Unused; kept for API compatibility.
    smooth : bool
        If True, apply Gaussian smoothing (sigma=2) to predicted heading components.
    **kwargs : dict
        Passed to augment_with_time_delay_embedding (time_window, input_names,
        output_names, direction).
    """
    def _predict_heading(df):
        aug = augment_with_time_delay_embedding([df], **kwargs).iloc[:, 0:n_input]
        components = best_estimator.predict(aug)
        if smooth:
            components = gaussian_filter1d(best_estimator.predict(aug), sigma=2, axis=0)
        heading_pred = np.arctan2(components[:, 1], components[:, 0])
        n_prepend = len(df["position_x"]) - len(heading_pred)
        return np.concatenate([np.full(n_prepend, heading_pred[0]), heading_pred])

    def _plot_arrows(xpos, ypos, phi, color, colormap, edgecolor, alpha):
        color = np.array(color)
        xymean = np.mean(np.abs(np.hstack((xpos, ypos))))
        if arrow_size is None:
            xymean_scaled = 0.21 * xymean
            sz = np.array(0.01) if xymean_scaled < 0.0001 else np.hstack((xymean_scaled, 1))
            size_radius = sz[sz > 0][0]
        elif isinstance(arrow_size, list):
            sz = np.hstack((arrow_size[0] * xymean, 1))
            size_radius = sz[sz > 0][0]
        else:
            size_radius = arrow_size

        colornorm = [np.min(color), np.max(color)]
        fpl.colorline_with_heading(
            axis, np.flip(xpos), np.flip(ypos), np.flip(color), np.flip(phi),
            nskip=nskip, size_radius=size_radius, deg=False, colormap=colormap,
            center_point_size=0.0001, colornorm=colornorm, show_centers=False,
            size_angle=20, alpha=alpha, edgecolor=edgecolor,
        )
        axis.set_aspect('equal')
        min_size = 0.1
        xrange = max(xpos.max() - xpos.min(), min_size)
        yrange = max(ypos.max() - ypos.min(), min_size)
        axis.set_xlim(xpos.min() - 0.2 * xrange, xpos.max() + 0.2 * xrange)
        axis.set_ylim(ypos.min() - 0.2 * yrange, ypos.max() + 0.2 * yrange)
        if include_id:
            axis.set_title(trajectory['trajec_objid'].iloc[0])

    heading_predicted = _predict_heading(trajectory)
    xpos = trajectory.position_x.values
    ypos = trajectory.position_y.values
    timestamps = trajectory.timestamp.values

    _plot_arrows(xpos, ypos, heading_predicted, timestamps,
                 colormap=blue_cmap, edgecolor='none', alpha=0.7)
    _plot_arrows(xpos, ypos, trajectory.heading_angle.values, timestamps,
                 colormap=red_cmap, edgecolor='black', alpha=0.3)


def custom_density_plots(
    axes: list,
    training_and_testing_X_data: list,
    training_and_testing_y_data: list,
    best_estimator,
    cmap,
    titles: list = None,
) -> None:
    """
    Plot 2D log-scale density histograms of predicted vs. true heading angles.

    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
    training_and_testing_X_data : list of np.ndarray
        Input feature arrays for each dataset (e.g., [X_train, X_test]).
    training_and_testing_y_data : list of array-like
        True heading component arrays (shape [N, 2]) for each dataset.
    best_estimator : model with .predict()
    cmap : matplotlib colormap
    titles : list of str, optional
        Defaults to ["Training Data", "Testing Data"].
    """
    if titles is None:
        titles = ["Training Data", "Testing Data"]
    for axis, Xdata, ydata, title in zip(axes, training_and_testing_X_data,
                                          training_and_testing_y_data, titles):
        Y_predict = best_estimator.predict(Xdata, batch_size=4096)
        zeta_predict = (np.arctan2(Y_predict[:, 1], Y_predict[:, 0]) + 2 * np.pi) % (2 * np.pi)
        zeta_true = (np.arctan2(ydata.values[:, 1], ydata.values[:, 0]) + 2 * np.pi) % (2 * np.pi)

        h = axis.hist2d(zeta_true, zeta_predict, bins=(128, 128), density=True,
                        cmap=cmap, norm=mcolors.LogNorm(clip=True))
        plt.colorbar(h[3], ax=axis).set_label('Density (log scale)')
        axis.set_ylim(-0.1, 2 * np.pi + 0.1)
        axis.set_xlim(-0.1, 2 * np.pi + 0.1)
        axis.set_ylabel('Predicted Heading (rad)')
        axis.set_xlabel('True Heading (rad)')
        axis.set_title(title)


def make_color_map(color_list=None, color_proportions=None, N=256):
    """
    Build a matplotlib colormap from a list of named colors.

    Parameters
    ----------
    color_list : list of str, optional
        Defaults to a white → blue → yellow → red diverging scale.
    color_proportions : list of float, 'even', or None
        Positions of each color in [0, 1]. None = evenly spaced starting at 0.01.
        'even' = evenly spaced from 0 to 1.
    N : int
        Number of discrete colormap levels.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    if color_list is None:
        color_list = ['white', 'deepskyblue', 'mediumblue', 'yellow', 'orange', 'red', 'darkred']
    if color_proportions is None:
        v = np.hstack((0.0, np.linspace(0.01, 1, len(color_list) - 1)))
    elif color_proportions == 'even':
        v = np.linspace(0.0, 1, len(color_list))
    else:
        v = color_proportions
    return LinearSegmentedColormap.from_list('rg', list(zip(v, color_list)), N=N)


def plot_fly_inputs_stacked(df: pd.DataFrame, axes=None):
    """
    Plot the six kinematic input signals on stacked, time-aligned axes.

    Signals: groundspeed, groundspeed_angle, airspeed, airspeed_angle,
    thrust, thrust_angle.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'timestamp' and the six signal columns.
    axes : array-like of matplotlib.axes.Axes, optional
        If None, a new figure with 6 subplots is created.

    Returns
    -------
    axes : list of matplotlib.axes.Axes
    """
    input_cols = ['groundspeed', 'groundspeed_angle', 'airspeed',
                  'airspeed_angle', 'thrust', 'thrust_angle']
    angle_cols = {'groundspeed_angle', 'airspeed_angle', 'thrust_angle'}
    ylabels = {
        'groundspeed':       'GS (m/s)',
        'groundspeed_angle': 'GS angle',
        'airspeed':          'AS (m/s)',
        'airspeed_angle':    'AS angle',
        'thrust':            'Thrust',
        'thrust_angle':      'Thrust angle',
    }
    colors = {
        'groundspeed':       'red',
        'groundspeed_angle': 'red',
        'airspeed':          'turquoise',
        'airspeed_angle':    'turquoise',
        'thrust':            'green',
        'thrust_angle':      'green',
    }
    if axes is None:
        _, axes = plt.subplots(len(input_cols), 1, figsize=(10, 8), dpi=150, sharex=True)
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    for ax, col in zip(axes, input_cols):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        ax.plot(df['timestamp'], df[col], lw=0.8, color=colors.get(col, '#333333'))
        ax.set_ylabel(ylabels.get(col, col), fontsize=10, rotation=0,
                      ha='right', va='center', labelpad=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if col in angle_cols:
            ax.set_ylim(-np.pi, np.pi)
            ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_yticklabels([r'$-\pi$', '', r'$0$', '', r'$\pi$'])
        else:
            ymax = df[col].max()
            ax.set_ylim(0, ymax * 1.05)
            ax.set_yticks([0, ymax])
            ax.set_yticklabels([r'$0$', f'${ymax:.2g}$'])
    axes[-1].set_xlabel("Time (s)", fontsize=10)
    return axes


# ============================================================
# Data Utilities
# ============================================================

def random_segments_from_df(
    df: pd.DataFrame,
    n_segment: int = 1,
    segment_sizes: tuple = (10,),
    reset_index: bool = False,
):
    """
    Extract random non-overlapping segments from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    n_segment : int
        Number of segments to extract.
    segment_sizes : tuple of int
        Possible segment lengths; one is chosen randomly per segment.
    reset_index : bool
        If True, reset the index of each extracted segment.

    Returns
    -------
    segment_list : list of pd.DataFrame
    segment_start_list : list of int
        Starting row indices of each segment.
    """
    n_df_size = df.shape[0]
    n_segment_sizes = len(segment_sizes)
    segment_start_list = []
    segment_list = []

    for n in range(n_segment):
        np.random.seed(seed=n)
        segment_size = int(segment_sizes[
            np.squeeze(np.random.randint(0, high=n_segment_sizes, size=1, dtype=int))
        ])
        c = 1
        np.random.seed(seed=n)
        segment_start = int(np.squeeze(
            np.random.randint(0, high=n_df_size - segment_size, size=1, dtype=int)
        ))
        while segment_start in segment_start_list:
            np.random.seed(seed=n + c)
            segment_start = int(np.squeeze(
                np.random.randint(0, high=n_df_size - segment_size, size=1, dtype=int)
            ))
            c += 1
            if c > 100:
                print('Warning: reusing random start point after 100 iterations')
                break

        segment_start_list.append(segment_start)
        segment = df.iloc[segment_start:(segment_start + segment_size), :]
        if reset_index:
            segment = segment.reset_index(drop=True)
        segment_list.append(segment)

    return segment_list, segment_start_list


def list_of_dicts_to_dict_of_lists(list_of_dicts: list, keynames=None, make_array=None) -> dict:
    """
    Convert a list of same-keyed dicts to a dict of lists.

    Parameters
    ----------
    list_of_dicts : list of dict
    keynames : list of str, optional
        Keys to extract. Defaults to all keys in the first dict.
    make_array : any, optional
        If not None, convert each list to a numpy array via np.hstack.

    Returns
    -------
    dict
        {key: [val_from_dict_0, val_from_dict_1, ...]}
    """
    if keynames is None:
        keynames = list_of_dicts[0].keys()
    result = {k: [d[k] for d in list_of_dicts] for k in keynames}
    if make_array is not None:
        result = {k: np.hstack(v) for k, v in result.items()}
    return result
