"""Rotation conversions for the 55 SMPL-X joints (axis-angle <-> 6D, quaternion slerp)."""
import numpy as np
import torch
import torch.nn.functional as F

NUM_JOINTS = 55
POSE_DIM = NUM_JOINTS * 3     # axis-angle
POSE_DIM_6D = NUM_JOINTS * 6  # 6D rotation


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / torch.clamp(angle, min=1e-8)
    x, y, z = axis.unbind(-1)
    ca = torch.cos(angle).squeeze(-1)
    sa = torch.sin(angle).squeeze(-1)
    one_minus_ca = 1.0 - ca
    m00 = ca + x * x * one_minus_ca
    m01 = x * y * one_minus_ca - z * sa
    m02 = x * z * one_minus_ca + y * sa
    m10 = y * x * one_minus_ca + z * sa
    m11 = ca + y * y * one_minus_ca
    m12 = y * z * one_minus_ca - x * sa
    m20 = z * x * one_minus_ca - y * sa
    m21 = z * y * one_minus_ca + x * sa
    m22 = ca + z * z * one_minus_ca
    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    batch_shape = matrix.shape[:-2]
    r = matrix.reshape(-1, 3, 3)
    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
    axis = torch.stack([r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]], dim=-1)
    axis = axis / torch.clamp(torch.norm(axis, dim=-1, keepdim=True), min=1e-8)
    small = angle < 1e-6
    if bool(small.any()):
        axis[small] = torch.tensor([1.0, 0.0, 0.0], device=matrix.device, dtype=matrix.dtype)
    return (axis * angle[:, None]).reshape(*batch_shape, 3)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    q_abs = torch.sqrt(torch.clamp(torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1), min=0.0))
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None]).clamp_min(0.1)
    pick = q_abs.argmax(dim=-1)
    quat = quat_candidates.gather(dim=-2, index=pick[..., None, None].expand(*pick.shape, 1, 4)).squeeze(-2)
    quat = F.normalize(quat, dim=-1)
    return torch.where(quat[..., :1] < 0.0, -quat, quat)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quaternion, dim=-1)
    r, i, j, k = q.unbind(dim=-1)
    two_s = 2.0 / (q * q).sum(dim=-1).clamp_min(1e-8)
    return torch.stack([
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def slerp_quaternion(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    lerp = F.normalize(q0 + alpha * (q1 - q0), dim=-1)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0).clamp_min(1e-8)
    theta = theta_0 * alpha
    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    slerp = F.normalize(s0 * q0 + s1 * q1, dim=-1)
    return torch.where(dot > 0.9995, lerp, slerp)


@torch.no_grad()
def aa_to_rot6d(poses_aa: np.ndarray, device: str = "cpu") -> np.ndarray:
    """[T, >=165] axis-angle -> [T, 330] 6D rotations."""
    arr = np.asarray(poses_aa, dtype=np.float32)
    t = torch.from_numpy(arr[:, :POSE_DIM].reshape(arr.shape[0], NUM_JOINTS, 3)).float().to(device)
    rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(t)).cpu().numpy()
    return rot6d.reshape(arr.shape[0], POSE_DIM_6D).astype(np.float32)


@torch.no_grad()
def rot6d_to_aa(body_6d: np.ndarray, device: str = "cpu") -> np.ndarray:
    """[T, >=330] 6D rotations -> [T, 165] axis-angle."""
    arr = np.asarray(body_6d, dtype=np.float32)
    t = torch.from_numpy(arr[:, :POSE_DIM_6D].reshape(arr.shape[0], NUM_JOINTS, 6)).float().to(device)
    aa = matrix_to_axis_angle(rotation_6d_to_matrix(t)).cpu().numpy()
    return aa.reshape(arr.shape[0], POSE_DIM).astype(np.float32)
