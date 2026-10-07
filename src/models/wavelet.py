import numpy as np
import pywt
import torch
import torch.nn as nn
from torch.nn import functional as F


def _center_padding(kernel: np.ndarray) -> tuple[int, int]:
    """Left / right padding that keeps the largest filter tap at the output position."""
    center_idx = int(np.argmax(np.abs(kernel)))
    return center_idx, len(kernel) - 1 - center_idx


def _pad_safe(x: torch.Tensor, pad: tuple[int, int], mode: str) -> torch.Tensor:
    length = x.shape[-1]
    if mode == "circular" and (pad[0] >= length or pad[1] >= length):
        # circular padding wraps around multiple times when pad >= length
        mode = "constant" if length <= 1 else "replicate"
    return F.pad(x, pad, mode=mode)


class MotionWaveletISWT(nn.Module):
    def __init__(self, in_channels: int, levels: int = 3, wavelet: str = "bior2.8"):
        super().__init__()
        self.levels = levels
        self.in_channels = in_channels

        wave = pywt.Wavelet(wavelet)
        rec_lo = np.array(wave.rec_lo)[::-1].copy()
        rec_hi = np.array(wave.rec_hi)[::-1].copy()

        self.pad_l_lo, self.pad_r_lo = _center_padding(rec_lo)
        self.pad_l_hi, self.pad_r_hi = _center_padding(rec_hi)

        w_l = torch.from_numpy(rec_lo).float().view(1, 1, -1)
        w_h = torch.from_numpy(rec_hi).float().view(1, 1, -1)

        self.register_buffer("w_l", w_l.repeat(in_channels, 1, 1))
        self.register_buffer("w_h", w_h.repeat(in_channels, 1, 1))

    def _to_block_order(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input is interleaved per-channel:
        [ch0(cA_L, cD_L, ..., cD_1), ch1(...), ...]
        Convert to block order expected by ISWT:
        [cA_L, cD_L, cD_{L-1}, ..., cD_1].
        """
        b, d, t = x.shape
        l1 = self.levels + 1
        c = self.in_channels
        if d != c * l1:
            raise RuntimeError(f"wavelet coeff dim {d} not equal to channels*(levels+1) ({c}*{l1})")
        return x.view(b, c, l1, t).permute(0, 2, 1, 3).contiguous().view(b, d, t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, Total_C, T)
        x = self._to_block_order(x)
        # Split channels: [cA3, cD3, cD2, cD1]
        chunks = torch.split(x, self.in_channels, dim=1)
        current_approx = chunks[0]
        details = chunks[1:]

        for i in range(self.levels):
            curr_level = self.levels - 1 - i
            dilation = 2 ** curr_level

            detail = details[i]

            pl_lo, pr_lo = self.pad_l_lo * dilation, self.pad_r_lo * dilation
            pl_hi, pr_hi = self.pad_l_hi * dilation, self.pad_r_hi * dilation

            padded_approx = _pad_safe(current_approx, (pl_lo, pr_lo), mode="circular")
            padded_detail = _pad_safe(detail, (pl_hi, pr_hi), mode="circular")

            rec_low = F.conv1d(padded_approx, self.w_l, dilation=dilation, groups=self.in_channels)
            rec_high = F.conv1d(padded_detail, self.w_h, dilation=dilation, groups=self.in_channels)

            # SWT reconstruction: (Low + High) / 2
            current_approx = (rec_low + rec_high) * 0.5

        return current_approx


class MotionWaveletSWT(nn.Module):
    """Stationary (undecimated) wavelet transform along time, the inverse of MotionWaveletISWT.

    (B, C, T) -> (B, C * (levels + 1), T) in block order [cA_L, cD_L, cD_{L-1}, ..., cD_1].
    """

    def __init__(self, in_channels: int, levels: int = 3, wavelet: str = "bior2.8"):
        super().__init__()
        self.levels = levels
        self.in_channels = in_channels

        wave = pywt.Wavelet(wavelet)
        dec_lo = np.array(wave.dec_lo)[::-1].copy()
        dec_hi = np.array(wave.dec_hi)[::-1].copy()

        self.pad_l_lo, self.pad_r_lo = _center_padding(dec_lo)
        self.pad_l_hi, self.pad_r_hi = _center_padding(dec_hi)

        self.register_buffer("w_l", torch.from_numpy(dec_lo).float().view(1, 1, -1).repeat(in_channels, 1, 1))
        self.register_buffer("w_h", torch.from_numpy(dec_hi).float().view(1, 1, -1).repeat(in_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current, outputs = x, []
        for i in range(self.levels):
            dilation = 2 ** i
            pad_lo = (self.pad_l_lo * dilation, self.pad_r_lo * dilation)
            pad_hi = (self.pad_l_hi * dilation, self.pad_r_hi * dilation)
            low = F.conv1d(_pad_safe(current, pad_lo, mode="circular"),
                           self.w_l, dilation=dilation, groups=self.in_channels)
            high = F.conv1d(_pad_safe(current, pad_hi, mode="circular"),
                            self.w_h, dilation=dilation, groups=self.in_channels)
            outputs.append(high)
            current = low
        outputs.append(current)
        return torch.cat(outputs[::-1], dim=1)
