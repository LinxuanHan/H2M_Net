from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .degradation import low_frequency_projection


class PathologyGate(nn.Module):
    """129Xe 通气缺损保护门控。

    低信号区域可能是真实通气缺损，而不是需要由人类健康先验补全的纹理。
    因此这里用可微 sigmoid 阈值 + Gaussian 平滑生成软门控 M(x_LR)，
    只允许可靠肺实质区域接受较强的高频纹理 steering。
    """

    def __init__(self, threshold=0.15, temperature=0.05, kernel_size=3, sigma=1.0):
        super().__init__()
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.kernel_size = int(kernel_size)
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        g_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        g_2d = g_1d[:, None] * g_1d[None, :]
        g_2d = g_2d / g_2d.sum().clamp_min(1e-12)
        self.register_buffer("gaussian_kernel", g_2d.view(1, 1, kernel_size, kernel_size))

    def forward(self, x_lr):
        gate = torch.sigmoid((x_lr - self.threshold) / self.temperature)
        pad = self.kernel_size // 2
        gate_padded = F.pad(gate, (pad, pad, pad, pad), mode="reflect")
        kernel = self.gaussian_kernel.repeat(x_lr.shape[1], 1, 1, 1)
        return F.conv2d(gate_padded, kernel, groups=x_lr.shape[1]).clamp(0.0, 1.0)


def make_beta_schedule(timesteps: int, schedule: str = "linear") -> torch.Tensor:
    if schedule != "linear":
        raise ValueError(f"Unsupported beta schedule: {schedule}")
    return torch.linspace(1e-4, 2e-2, timesteps)


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000, beta_schedule: str = "linear"):
        super().__init__()
        betas = make_beta_schedule(timesteps, beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        self.timesteps = timesteps
        for name, value in {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
            "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod),
            "sqrt_recipm1_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod - 1),
        }.items():
            self.register_buffer(name, value)

    def extract(self, values: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self.extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ) * noise

    def training_loss(
        self,
        model: nn.Module,
        hr: torch.Tensor,
        lr_up: torch.Tensor,
        scale: torch.Tensor,
        cond_drop_prob: float = 0.1,
        x0_loss_weight: float = 0.0,
        gradient_loss_weight: float = 0.0,
        low_t_prob: float = 0.0,
        low_t_max_fraction: float = 0.35,
    ) -> torch.Tensor:
        batch = hr.shape[0]
        t = torch.randint(0, self.timesteps, (batch,), device=hr.device)
        if low_t_prob > 0:
            low_t_max = max(1, int(self.timesteps * low_t_max_fraction))
            low_t = torch.randint(0, low_t_max, (batch,), device=hr.device)
            use_low_t = torch.rand((batch,), device=hr.device) < low_t_prob
            t = torch.where(use_low_t, low_t, t)
        noise = torch.randn_like(hr)
        x_t = self.q_sample(hr, t, noise)
        cond_drop = torch.rand((), device=hr.device).item() < cond_drop_prob
        pred = model(x_t, t, lr_up, scale, cond_drop=cond_drop)
        loss = F.mse_loss(pred, noise)
        if x0_loss_weight > 0 or gradient_loss_weight > 0:
            pred_x0 = self.predict_start_from_noise(x_t, t, pred).clamp(0.0, 1.0)
            if x0_loss_weight > 0:
                loss = loss + x0_loss_weight * F.l1_loss(pred_x0, hr)
            if gradient_loss_weight > 0:
                pred_dx = pred_x0[..., :, 1:] - pred_x0[..., :, :-1]
                pred_dy = pred_x0[..., 1:, :] - pred_x0[..., :-1, :]
                hr_dx = hr[..., :, 1:] - hr[..., :, :-1]
                hr_dy = hr[..., 1:, :] - hr[..., :-1, :]
                loss = loss + gradient_loss_weight * (F.l1_loss(pred_dx, hr_dx) + F.l1_loss(pred_dy, hr_dy))
        return loss

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        lr_up: torch.Tensor,
        scale: torch.Tensor,
        sampling_steps: int = 50,
        guidance_scale: float = 1.5,
        data_consistency_weight: float = 0.0,
        init_mode: str = "lr_noise",
        noise_strength: float = 0.35,
        anchor_weight: float = 0.25,
        residual_scale: float = 0.7,
        residual_clip: float = 0.2,
    ) -> torch.Tensor:
        device = lr_up.device
        batch = lr_up.shape[0]
        if init_mode == "random":
            start_t = self.timesteps - 1
            x = torch.randn_like(lr_up)
        else:
            start_t = int((self.timesteps - 1) * float(noise_strength))
            start_t = max(1, min(self.timesteps - 1, start_t))
            t_start = torch.full((batch,), start_t, device=device, dtype=torch.long)
            x = self.q_sample(lr_up, t_start, torch.randn_like(lr_up))
        step_ids = torch.linspace(start_t, 0, sampling_steps, device=device).long()
        for index, t_value in enumerate(step_ids):
            t = torch.full((batch,), int(t_value.item()), device=device, dtype=torch.long)
            eps_cond = model(x, t, lr_up, scale, cond_drop=False)
            eps_uncond = model(x, t, lr_up, scale, cond_drop=True)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            x0 = self.predict_start_from_noise(x, t, eps).clamp(0.0, 1.0)
            x0 = low_frequency_projection(x0, lr_up, int(scale[0].item()), data_consistency_weight)
            if index == len(step_ids) - 1:
                x = x0
                continue
            next_t = torch.full((batch,), int(step_ids[index + 1].item()), device=device, dtype=torch.long)
            alpha_next = self.extract(self.alphas_cumprod, next_t, x.shape)
            x = torch.sqrt(alpha_next) * x0 + torch.sqrt(1.0 - alpha_next) * eps
        residual = (x - lr_up).clamp(-residual_clip, residual_clip)
        x = lr_up + residual_scale * residual
        if anchor_weight > 0:
            x = (1.0 - anchor_weight) * x + anchor_weight * lr_up
        return x.clamp(0.0, 1.0)

    @torch.no_grad()
    def ddim_steered_sample(
        self,
        model: nn.Module,
        lr_up: torch.Tensor,
        omega: float = 3.0,
        sampling_steps: int = 50,
        data_consistency_weight: float = 0.0,
        init_mode: str = "lr_noise",
        noise_strength: float = 0.35,
        gate_config: dict | None = None,
        degradation_scale: int = 2,
        delta: float = 1e-8,
        anchor_weight: float = 0.25,
        residual_scale: float = 0.7,
        residual_clip: float = 0.2,
        use_orthogonal: bool = True,
        use_pathology_gate: bool = True,
        use_residual_anchor: bool = True,
    ) -> torch.Tensor:
        """三路正交 latent steering 的 DDIM 推理。

        条件 ID 约定：
          0: Mouse_LR, 1: Human_LR, 2: Human_HR。
        三路预测一次 batched forward 完成：
          eps_M_LR = εθ(x_t, t, C_M_LR)
          eps_H_LR = εθ(x_t, t, C_H_LR)
          eps_H_HR = εθ(x_t, t, C_H_HR)

        高频纹理方向：
          V_detail = eps_H_HR - eps_H_LR
        跨物种拓扑方向：
          V_topo = eps_H_LR - eps_M_LR
        正交净化：
          V_detail_perp = V_detail - Proj_{V_topo}(V_detail)
        最终噪声：
          eps_hat = eps_M_LR + omega * M(x_LR) * V_detail_perp
        """

        device = lr_up.device
        batch = lr_up.shape[0]
        gate_kwargs = gate_config or {}
        pathology_gate = PathologyGate(**gate_kwargs).to(device)

        if init_mode == "random":
            start_t = self.timesteps - 1
            x = torch.randn_like(lr_up)
        else:
            start_t = int((self.timesteps - 1) * float(noise_strength))
            start_t = max(1, min(self.timesteps - 1, start_t))
            t_start = torch.full((batch,), start_t, device=device, dtype=torch.long)
            x = self.q_sample(lr_up, t_start, torch.randn_like(lr_up))

        gate = pathology_gate(lr_up) if use_pathology_gate else torch.ones_like(lr_up)
        step_ids = torch.linspace(start_t, 0, sampling_steps, device=device).long()
        for index, t_value in enumerate(step_ids):
            t = torch.full((batch,), int(t_value.item()), device=device, dtype=torch.long)

            x_t_cat = torch.cat([x, x, x], dim=0)
            lr_up_cat = torch.cat([lr_up, lr_up, lr_up], dim=0)
            t_cat = torch.cat([t, t, t], dim=0)
            cond_mouse_lr = torch.zeros((batch,), device=device, dtype=torch.long)
            cond_human_lr = torch.ones((batch,), device=device, dtype=torch.long)
            cond_human_hr = torch.full((batch,), 2, device=device, dtype=torch.long)
            cond_cat = torch.cat([cond_mouse_lr, cond_human_lr, cond_human_hr], dim=0)

            eps_all = model(x_t_cat, t_cat, lr_up_cat, cond_cat, cond_drop=False)
            eps_mouse_lr, eps_human_lr, eps_human_hr = eps_all.chunk(3, dim=0)

            v_detail = eps_human_hr - eps_human_lr
            v_topo = eps_human_lr - eps_mouse_lr
            dot = (v_detail * v_topo).flatten(1).sum(dim=1).view(batch, 1, 1, 1)
            denom = (v_topo * v_topo).flatten(1).sum(dim=1).view(batch, 1, 1, 1).clamp_min(delta)
            v_detail_ortho = v_detail - (dot / denom) * v_topo if use_orthogonal else v_detail

            eps = eps_mouse_lr + float(omega) * gate * v_detail_ortho
            x0 = self.predict_start_from_noise(x, t, eps).clamp(0.0, 1.0)
            x0 = low_frequency_projection(x0, lr_up, int(degradation_scale), data_consistency_weight)
            if index == len(step_ids) - 1:
                x = x0
                continue
            next_t = torch.full((batch,), int(step_ids[index + 1].item()), device=device, dtype=torch.long)
            alpha_next = self.extract(self.alphas_cumprod, next_t, x.shape)
            x = torch.sqrt(alpha_next) * x0 + torch.sqrt(1.0 - alpha_next) * eps

        if use_residual_anchor:
            residual = (x - lr_up).clamp(-residual_clip, residual_clip)
            x = lr_up + residual_scale * residual
            if anchor_weight > 0:
                x = (1.0 - anchor_weight) * x + anchor_weight * lr_up
        return x.clamp(0.0, 1.0)
