import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """因果1D 深度可分离卷积 + 残差连接 (HW-friendly Tier B, 2026-04-27).

    将原来的 dense Conv1d(in,out,K) 替换为:
        DepthwiseConv1d(in, in, K, groups=in)  +  PointwiseConv1d(in, out, 1)
    每层 MAC: K·CIN·COUT  →  K·CIN + CIN·COUT,  HLS DSP 5-8× 削减.
    输入/输出 shape 完全不变, 残差路径不变.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.stride = stride

        # Depthwise: K-tap per-channel conv (no channel mixing). stride 在这一步.
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=0, dilation=dilation,
                                   groups=in_channels, bias=False)
        # Pointwise: 1×1 cross-channel mix.
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        padded_x = F.pad(x, (self.padding, 0))
        out = self.depthwise(padded_x)
        out = self.pointwise(out)
        out = self.bn(out)

        if residual.shape[-1] != out.shape[-1]:
            residual = residual[..., :out.shape[-1]]

        out = out + residual
        out = self.relu(out)
        return out


class _DWSepDecBlock(nn.Module):
    """Decoder stage: Upsample×2 + DepthwiseConv(k=3) + Pointwise(1×1) + BN + ReLU.

    HW-friendly Tier B (2026-04-27): replaces dense Conv1d(c_in, c_out, k=3).
    For the final stage (out_ch=1) keeps the original behavior — just a 3-tap
    conv with bias and no BN/ReLU, since DWSep with 1-channel output collapses
    to nearly a regular conv anyway (DW: 3 weights, PW: c_in weights).
    """
    def __init__(self, in_ch, out_ch, k=3, is_last=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.depthwise = nn.Conv1d(in_ch, in_ch, k, padding=k // 2,
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=is_last)
        self.bn = nn.BatchNorm1d(out_ch) if not is_last else nn.Identity()
        self.act = nn.ReLU(inplace=True) if not is_last else nn.Identity()

    def forward(self, x):
        x = self.up(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class TimedomainAudioEncoder(nn.Module):
    """Time-domain audio encoder (U-Net with concat skip connections).

    Encoder: 6x CausalConv1d (stride=2), channels 1->32->64->96->128->192->256
    Decoder: 6x (Upsample x2 + Conv1d k=3), channels mirror encoder.

    The decoder uses nearest-neighbour upsampling followed by a k=3 Conv1d
    instead of ConvTranspose1d. This avoids checkerboard artefacts, keeps
    output length exactly 2*T_in at every stage, and maps cleanly to HLS
    (no transposed-conv kernel needed).

    Skip connections (scheme B): at each decoder stage except the final one,
    the upsampled feature is concatenated with the encoder output at the same
    resolution and reduced by a 1x1 Conv + BN + ReLU back to the original
    channel count.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        kernel_sizes = config.model.kernel_sizes   # [25, 19, 15, 11, 7, 5]
        strides = config.model.strides             # [2,  2,  2,  2,  2, 2]
        channels = config.model.audio_channels     # 256

        channel_progression = [32, 64, 96, 128, 192, 256][:len(kernel_sizes)]

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.ModuleList()
        in_ch = 1
        for i, (ks, stride) in enumerate(zip(kernel_sizes, strides)):
            out_ch = channel_progression[i]
            self.encoder.append(CausalConv1d(in_ch, out_ch, ks, stride))
            in_ch = out_ch

        # ── Decoder ──────────────────────────────────────────────────────────
        # 2026-04-27 HW-friendly Tier B: Sequential(Upsample, dense Conv k=3, BN, ReLU)
        # 替换为 DWSep 版本 (Upsample → DW(k=3, groups=in) → PW(1×1) → BN → ReLU).
        # 每层 MAC: 3·CIN·COUT  →  3·CIN + CIN·COUT, HLS DSP 大幅削减.
        decoder_channels = list(reversed(channel_progression[:-1])) + [1]

        self.decoder = nn.ModuleList()
        in_ch = channels  # starts from fusion output (256)
        for out_ch in decoder_channels:
            is_last = (out_ch == 1)
            self.decoder.append(_DWSepDecBlock(in_ch, out_ch, k=3, is_last=is_last))
            in_ch = out_ch

        # ── Skip-connection fusion (concat + 1x1 Conv) ───────────────────────
        # Applied after decoder stages 0..N-2; final stage outputs ch=1 (no skip).
        self.skip_fusion = nn.ModuleList()
        for i in range(len(self.decoder) - 1):
            c = decoder_channels[i]
            self.skip_fusion.append(nn.Sequential(
                nn.Conv1d(c * 2, c, 1, bias=False),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            ))

    def encode(self, audio):
        """
        Args:
            audio: [B, 1, T]
        Returns:
            features: [B, 256, T/64]
            skips:    list of 6 encoder-stage outputs (shallow → deep)
        """
        x = audio
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        return x, skips

    def decode(self, features, skips):
        """
        Args:
            features: [B, 256, T']   (fusion output)
            skips:    list of encoder-stage outputs from encode()
        Returns:
            audio: [B, 1, T_out]
        """
        x = features
        num_stages = len(self.decoder)
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < num_stages - 1:
                # Skip at matching resolution: decoder stage i output aligns
                # with skips[num_stages - 2 - i] (5→s5, 4→s4, ..., 1→s1).
                skip = skips[num_stages - 2 - i]
                min_len = min(x.shape[-1], skip.shape[-1])
                x = x[..., :min_len]
                skip = skip[..., :min_len]
                x = torch.cat([x, skip], dim=1)
                x = self.skip_fusion[i](x)
        return x

    def forward(self, audio, encoder_only=False):
        encoded, skips = self.encode(audio)
        if encoder_only:
            return encoded
        return self.decode(encoded, skips)
