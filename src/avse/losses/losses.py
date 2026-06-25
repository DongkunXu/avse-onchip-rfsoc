import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from asteroid.losses import pairwise_neg_sisdr, SingleSrcNegSTOI
from torch_pesq import PesqLoss


class MultiScaleTimeLoss(nn.Module):
    """多尺度时域损失（无FFT版本）"""
    def __init__(self, scales=[64, 256, 1024]):
        super().__init__()
        self.scales = scales

    def forward(self, pred, target):
        min_length = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_length]
        target = target[..., :min_length]

        total_loss = 0
        valid_scales = 0

        for scale in self.scales:
            actual_scale = min(scale, min_length // 4)
            if actual_scale < 4:
                continue
            if actual_scale % 2 == 0:
                actual_scale += 1

            kernel = torch.ones(1, 1, actual_scale, device=pred.device) / actual_scale
            padding = actual_scale // 2

            pred_smooth = F.conv1d(pred, kernel, padding=padding)[..., :min_length]
            target_smooth = F.conv1d(target, kernel, padding=padding)[..., :min_length]

            scale_loss = F.l1_loss(pred_smooth, target_smooth)
            total_loss += scale_loss
            valid_scales += 1

        return total_loss / max(valid_scales, 1)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT magnitude loss.

    Computes L1 loss between predicted and target log-magnitude spectrograms
    at multiple STFT resolutions.  This directly penalises checkerboard and
    other spectral artifacts that are invisible to pure time-domain losses.

    No learnable parameters — pure signal processing.

    Args:
        fft_sizes:   list of n_fft values (controls frequency resolution)
        hop_sizes:   list of hop lengths (controls time resolution)
        win_lengths: list of window lengths (controls spectral smoothness)
        eps:         small constant for log stability
    """

    def __init__(self,
                 fft_sizes=(512, 1024, 2048),
                 hop_sizes=(120, 240, 480),
                 win_lengths=(240, 600, 1200),
                 eps=1e-8):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.eps = eps

        # Pre-register Hann windows as buffers so they move with .to(device)
        for i, wl in enumerate(win_lengths):
            self.register_buffer(f'window_{i}', torch.hann_window(wl))

    def _stft(self, x, n_fft, hop, win_len, window):
        """Compute magnitude spectrogram for a [B, T] waveform."""
        B, T = x.shape
        # torch.stft expects [B, T] or [T]
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_len,
            window=window,
            return_complex=True,
            center=False,
        )
        return spec.abs()  # [B, F, frames]

    def forward(self, pred, target):
        # pred/target: [B, 1, T]
        pred_w = pred.squeeze(1)      # [B, T]
        target_w = target.squeeze(1)  # [B, T]

        total = 0.0
        for i, (n_fft, hop, wl) in enumerate(
                zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f'window_{i}')
            pred_mag = self._stft(pred_w, n_fft, hop, wl, window)
            tgt_mag = self._stft(target_w, n_fft, hop, wl, window)

            # Log-magnitude (more perceptually uniform)
            pred_log = torch.log(pred_mag + self.eps)
            tgt_log = torch.log(tgt_mag + self.eps)

            total += F.l1_loss(pred_log, tgt_log)

        return total / len(self.fft_sizes)


class ImprovedAVSELoss(nn.Module):
    """AVSE loss: time-domain + perceptual + multi-resolution spectral."""

    def __init__(self, config, sample_rate=16000):
        super().__init__()
        self.config = config

        self.l1_weight = getattr(config, 'l1_weight', 1.0)
        self.l2_weight = getattr(config, 'l2_weight', 0.5)
        self.stoi_weight = getattr(config, 'stoi_weight', 4.0)
        self.multiscale_weight = getattr(config, 'multiscale_weight', 1.0)
        self.si_sdr_weight = getattr(config, 'si_sdr_weight', 0.5)
        self.pesq_weight = getattr(config, 'pesq_weight', 3.0)
        self.stft_weight = getattr(config, 'stft_weight', 1.0)

        self.stoi_loss_func = SingleSrcNegSTOI(sample_rate=sample_rate)
        self.pesq_loss_func = PesqLoss(factor=1.0, sample_rate=sample_rate)
        self.multiscale_loss_func = MultiScaleTimeLoss()
        self.stft_loss_func = MultiResolutionSTFTLoss()

    def forward(self, pred, target):
        losses = {}

        min_length = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_length]
        target = target[..., :min_length]

        # Time-domain losses (mixed-precision safe)
        losses['l1_loss'] = F.l1_loss(pred, target)
        losses['l2_loss'] = F.mse_loss(pred, target)
        losses['multiscale_loss'] = self.multiscale_loss_func(pred, target)

        # STFT loss: directly penalises spectral artifacts
        losses['stft_loss'] = self.stft_loss_func(pred, target)

        # Perceptual losses — must run in float32
        with torch.cuda.amp.autocast(enabled=False):
            pred_fp32 = pred.float()
            target_fp32 = target.float()

            si_sdr_val = pairwise_neg_sisdr(pred_fp32, target_fp32)
            stoi_val = self.stoi_loss_func(pred_fp32.squeeze(1), target_fp32.squeeze(1))
            pesq_val = self.pesq_loss_func(target_fp32.squeeze(1), pred_fp32.squeeze(1))

            losses['si_sdr_loss'] = si_sdr_val.mean()
            losses['stoi_loss'] = stoi_val.mean()
            losses['pesq_loss'] = pesq_val.mean()

        losses['total_loss'] = (
            self.l1_weight       * losses['l1_loss'] +
            self.l2_weight       * losses['l2_loss'] +
            self.si_sdr_weight   * losses['si_sdr_loss'] +
            self.stoi_weight     * losses['stoi_loss'] +
            self.pesq_weight     * losses['pesq_loss'] +
            self.multiscale_weight * losses['multiscale_loss'] +
            self.stft_weight     * losses['stft_loss']
        )

        return losses


# Backward compatibility
AVSELoss = ImprovedAVSELoss
