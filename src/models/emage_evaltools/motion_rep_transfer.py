from pathlib import Path

import numpy as np
import smplx
import torch

from . import rotation_conversions as rc

_EVALTOOLS_DIR = Path(__file__).resolve().parent
_SMPLX_MODEL_DIR = _EVALTOOLS_DIR / "smplx_models"
_SMPLX_MODEL_FILE = _SMPLX_MODEL_DIR / "smplx" / "SMPLX_NEUTRAL_2020.npz"


def _create_smplx_model():
    if not _SMPLX_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"SMPL-X model not found at {_SMPLX_MODEL_FILE}. "
            "Download SMPLX_NEUTRAL_2020.npz from https://smpl-x.is.tue.mpg.de "
            "(registration required) as described in README.md, section 'External assets'."
        )
    return smplx.create(
        str(_SMPLX_MODEL_DIR),
        model_type="smplx",
        gender="NEUTRAL_2020",
        use_face_contour=False,
        num_betas=300,
        num_expression_coeffs=100,
        ext="npz",
        use_pca=False,
    ).eval()


smplx_model = _create_smplx_model()


def get_motion_rep_tensor(motion_tensor, pose_fps=30, device="cuda", betas=None):
    """Return position/velocity/rotation features from axis-angle motion (torch)."""
    global smplx_model
    smplx_model = smplx_model.to(device)
    bs, n, _ = motion_tensor.shape
    motion_tensor = motion_tensor.float().to(device)
    motion_tensor_reshaped = motion_tensor.reshape(bs * n, 165)
    if betas is None:
        betas = torch.zeros(n, 300, device=device)
    else:
        if isinstance(betas, np.ndarray):
            betas = torch.from_numpy(betas)
        betas = betas.to(device)
        if betas.ndim == 1:
            betas = betas.unsqueeze(0).repeat(n, 1)
    output = smplx_model(
        betas=torch.zeros(bs * n, 300, device=device),
        transl=torch.zeros(bs * n, 3, device=device),
        expression=torch.zeros(bs * n, 100, device=device),
        jaw_pose=torch.zeros(bs * n, 3, device=device),
        global_orient=torch.zeros(bs * n, 3, device=device),
        body_pose=motion_tensor_reshaped[:, 3 : 21 * 3 + 3],
        left_hand_pose=motion_tensor_reshaped[:, 25 * 3 : 40 * 3],
        right_hand_pose=motion_tensor_reshaped[:, 40 * 3 : 55 * 3],
        return_joints=True,
        leye_pose=torch.zeros(bs * n, 3, device=device),
        reye_pose=torch.zeros(bs * n, 3, device=device),
    )
    joints = output["joints"].reshape(bs, n, 127, 3)[:, :, :55, :]
    dt = 1 / pose_fps
    init_vel = (joints[:, 1:2] - joints[:, 0:1]) / dt
    middle_vel = (joints[:, 2:] - joints[:, :-2]) / (2 * dt)
    final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
    vel = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    position = joints
    rot_matrices = rc.axis_angle_to_matrix(motion_tensor.reshape(bs, n, 55, 3))
    rot6d = rc.matrix_to_rotation_6d(rot_matrices).reshape(bs, n, 55, 6)
    init_vel_ang = (motion_tensor[:, 1:2] - motion_tensor[:, 0:1]) / dt
    middle_vel_ang = (motion_tensor[:, 2:] - motion_tensor[:, :-2]) / (2 * dt)
    final_vel_ang = (motion_tensor[:, -1:] - motion_tensor[:, -2:-1]) / dt
    angular_velocity = torch.cat([init_vel_ang, middle_vel_ang, final_vel_ang], dim=1).reshape(bs, n, 55, 3)
    rep15d = torch.cat([position, vel, rot6d, angular_velocity], dim=3).reshape(bs, n, 55 * 15)
    return {
        "position": position,
        "velocity": vel,
        "rotation": rot6d,
        "axis_angle": motion_tensor,
        "angular_velocity": angular_velocity,
        "rep15d": rep15d,
    }


def get_motion_rep_numpy(poses_np, pose_fps=30, device="cuda", expressions=None, expression_only=False, betas=None):
    """Return position/velocity/rotation features from axis-angle motion (numpy)."""
    global smplx_model
    smplx_model = smplx_model.to(device)
    n = poses_np.shape[0]

    poses_ts = torch.from_numpy(poses_np).float().to(device).unsqueeze(0)
    poses_ts_reshaped = poses_ts.reshape(-1, 165)
    if betas is None:
        betas_ts = torch.zeros(n, 300, device=device)
    else:
        betas_ts = torch.from_numpy(betas).to(device)
        if betas_ts.ndim == 1:
            betas_ts = betas_ts.unsqueeze(0).repeat(n, 1)

    if expressions is not None and expression_only:
        expressions_ts = torch.from_numpy(expressions).float().to(device)
        output = smplx_model(
            betas=betas_ts,
            transl=torch.zeros(n, 3, device=device),
            expression=expressions_ts,
            jaw_pose=poses_ts_reshaped[:, 22 * 3 : 23 * 3],
            global_orient=torch.zeros(n, 3, device=device),
            body_pose=torch.zeros(n, 21 * 3, device=device),
            left_hand_pose=torch.zeros(n, 15 * 3, device=device),
            right_hand_pose=torch.zeros(n, 15 * 3, device=device),
            return_joints=True,
            leye_pose=torch.zeros(n, 3, device=device),
            reye_pose=torch.zeros(n, 3, device=device),
        )
        joints = output["vertices"].detach().cpu().numpy().reshape(n, -1)
        return {"vertices": joints}

    output = smplx_model(
        betas=betas_ts,
        transl=torch.zeros(n, 3, device=device),
        expression=torch.zeros(n, 100, device=device),
        jaw_pose=torch.zeros(n, 3, device=device),
        global_orient=torch.zeros(n, 3, device=device),
        body_pose=poses_ts_reshaped[:, 3 : 21 * 3 + 3],
        left_hand_pose=poses_ts_reshaped[:, 25 * 3 : 40 * 3],
        right_hand_pose=poses_ts_reshaped[:, 40 * 3 : 55 * 3],
        return_joints=True,
        leye_pose=torch.zeros(n, 3, device=device),
        reye_pose=torch.zeros(n, 3, device=device),
    )
    joints = output["joints"].detach().cpu().numpy().reshape(n, 127, 3)[:, :55, :]

    dt = 1 / pose_fps
    init_vel = (joints[1:2] - joints[0:1]) / dt
    middle_vel = (joints[2:] - joints[:-2]) / (2 * dt)
    final_vel = (joints[-1:] - joints[-2:-1]) / dt
    vel = np.concatenate([init_vel, middle_vel, final_vel], axis=0)

    position = joints

    poses_ts_reshaped_aa = poses_ts.reshape(1, n, 55, 3)
    rot_matrices = rc.axis_angle_to_matrix(poses_ts_reshaped_aa)[0]
    rot6d = rc.matrix_to_rotation_6d(rot_matrices).reshape(n, 55, 6).cpu().numpy()

    init_vel_ang = (poses_np[1:2] - poses_np[0:1]) / dt
    middle_vel_ang = (poses_np[2:] - poses_np[:-2]) / (2 * dt)
    final_vel_ang = (poses_np[-1:] - poses_np[-2:-1]) / dt
    angular_velocity = np.concatenate([init_vel_ang, middle_vel_ang, final_vel_ang], axis=0).reshape(n, 55, 3)

    rep15d = np.concatenate([position, vel, rot6d, angular_velocity], axis=2).reshape(n, 55 * 15)

    return {
        "position": position,
        "velocity": vel,
        "rotation": rot6d,
        "axis_angle": poses_np,
        "angular_velocity": angular_velocity,
        "rep15d": rep15d,
    }
