from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ElasticDeformer(nn.Module):
    """Smooth self-supervised elastic deformation for target-free H2M-Net training.

    The physical order is: deform the human HR image first, then degrade the
    deformed HR image to obtain a physically matched deformed LR-up condition.
    """

    def __init__(self, scale: int, alpha: float = 0.08, grid_size: int = 6, degradation_mode: str = "bicubic"):
        super().__init__()
        self.scale = int(scale)
        self.alpha = float(alpha)
        self.grid_size = int(grid_size)
        self.degradation_mode = degradation_mode

    @staticmethod
    def _base_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid.unsqueeze(0).repeat(batch, 1, 1, 1)

    def degrade(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        low_size = (max(1, height // self.scale), max(1, width // self.scale))
        down = F.interpolate(
            x,
            size=low_size,
            mode=self.degradation_mode,
            align_corners=False,
            antialias=True,
        )
        return F.interpolate(down, size=(height, width), mode=self.degradation_mode, align_corners=False).clamp(0.0, 1.0)

    def forward(self, x_hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x_hr.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got {tuple(x_hr.shape)}")
        batch, _, height, width = x_hr.shape
        coarse = torch.randn(batch, 2, self.grid_size, self.grid_size, device=x_hr.device, dtype=x_hr.dtype)
        displacement = F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=True)
        displacement = displacement.permute(0, 2, 3, 1) * self.alpha
        grid = self._base_grid(batch, height, width, x_hr.device, x_hr.dtype) + displacement
        x_hr_deformed = F.grid_sample(x_hr, grid, mode="bilinear", padding_mode="border", align_corners=True).clamp(0.0, 1.0)
        y_lr_deformed = self.degrade(x_hr_deformed)
        return x_hr_deformed, y_lr_deformed


class H2MSteeredDiffusion(nn.Module):
    """H2M-Net target-free elastic training and ORS/PSG/LRA inference wrapper."""

    def __init__(
        self,
        unet_model: nn.Module,
        scale: int,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        degradation_mode: str = "bicubic",
    ):
        super().__init__()
        self.unet = unet_model
        self.scale = int(scale)
        self.timesteps = int(timesteps)
        self.degradation_mode = degradation_mode
        betas = torch.linspace(beta_start, beta_end, self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        for name, value in {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
            "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod),
            "sqrt_recipm1_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod - 1.0),
        }.items():
            self.register_buffer(name, value)

    def extract(self, values: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def degrade(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        low_size = (max(1, height // self.scale), max(1, width // self.scale))
        down = F.interpolate(x, size=low_size, mode=self.degradation_mode, align_corners=False, antialias=True)
        return F.interpolate(down, size=(height, width), mode=self.degradation_mode, align_corners=False).clamp(0.0, 1.0)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self.extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ) * noise

    def _sample_timesteps(self, batch: int, device, low_t_prob: float = 0.0, low_t_max_fraction: float = 0.35) -> torch.Tensor:
        t = torch.randint(0, self.timesteps, (batch,), device=device)
        if low_t_prob > 0:
            low_t_max = max(1, int(self.timesteps * low_t_max_fraction))
            low_t = torch.randint(0, low_t_max, (batch,), device=device)
            choose_low = torch.rand(batch, device=device) < low_t_prob
            t = torch.where(choose_low, low_t, t)
        return t

    def _single_noise_loss(
        self,
        x0_target: torch.Tensor,
        condition_lr: torch.Tensor,
        condition_id: int,
        cond_drop_prob: float = 0.1,
        x0_loss_weight: float = 0.0,
        gradient_loss_weight: float = 0.0,
        low_t_prob: float = 0.0,
        low_t_max_fraction: float = 0.35,
    ) -> torch.Tensor:
        batch = x0_target.shape[0]
        t = self._sample_timesteps(batch, x0_target.device, low_t_prob, low_t_max_fraction)
        noise = torch.randn_like(x0_target)
        x_t = self.q_sample(x0_target, t, noise)
        c = torch.full((batch,), int(condition_id), device=x0_target.device, dtype=torch.long)
        cond_drop = torch.rand((), device=x0_target.device).item() < cond_drop_prob
        pred_noise = self.unet(x_t, t, condition_lr, c, cond_drop=cond_drop)
        loss = F.mse_loss(pred_noise, noise)
        if x0_loss_weight > 0 or gradient_loss_weight > 0:
            pred_x0 = self.predict_start_from_noise(x_t, t, pred_noise).clamp(0.0, 1.0)
            if x0_loss_weight > 0:
                loss = loss + x0_loss_weight * F.l1_loss(pred_x0, x0_target)
            if gradient_loss_weight > 0:
                pred_dx = pred_x0[..., :, 1:] - pred_x0[..., :, :-1]
                pred_dy = pred_x0[..., 1:, :] - pred_x0[..., :-1, :]
                target_dx = x0_target[..., :, 1:] - x0_target[..., :, :-1]
                target_dy = x0_target[..., 1:, :] - x0_target[..., :-1, :]
                loss = loss + gradient_loss_weight * (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy))
        return loss

    def compute_training_loss(
        self,
        x_hr: torch.Tensor,
        deformer: ElasticDeformer,
        lambda_follower: float = 1.0,
        lambda_corrector: float = 1.0,
        cond_drop_prob: float = 0.1,
        x0_loss_weight: float = 0.0,
        gradient_loss_weight: float = 0.0,
        low_t_prob: float = 0.0,
        low_t_max_fraction: float = 0.35,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        y_lr = self.degrade(x_hr)
        x_hr_deformed, y_lr_deformed = deformer(x_hr)
        loss_prior = self._single_noise_loss(
            x_hr, y_lr, 2, cond_drop_prob, x0_loss_weight, gradient_loss_weight, low_t_prob, low_t_max_fraction
        )
        loss_follower = self._single_noise_loss(
            x_hr_deformed, y_lr_deformed, 0, cond_drop_prob, x0_loss_weight, gradient_loss_weight, low_t_prob, low_t_max_fraction
        )
        loss_corrector = self._single_noise_loss(
            x_hr, y_lr_deformed, 1, cond_drop_prob, x0_loss_weight, gradient_loss_weight, low_t_prob, low_t_max_fraction
        )
        total = loss_prior + float(lambda_follower) * loss_follower + float(lambda_corrector) * loss_corrector
        logs = {
            "loss_prior": float(loss_prior.detach().cpu()),
            "loss_follower": float(loss_follower.detach().cpu()),
            "loss_corrector": float(loss_corrector.detach().cpu()),
        }
        return total, logs

    def pathology_soft_gate(self, y_mouse_lr_up: torch.Tensor, tau: float = 0.1, gamma: float = 0.02) -> torch.Tensor:
        gamma = max(float(gamma), 1e-6)
        return (1.0 - torch.sigmoid((y_mouse_lr_up - float(tau)) / gamma)).clamp(0.0, 1.0)

    def low_resolution_anchor(self, x: torch.Tensor, y_mouse_lr_up: torch.Tensor) -> torch.Tensor:
        return (x - self.degrade(x) + y_mouse_lr_up).clamp(0.0, 1.0)

    @torch.no_grad()
    def ddim_steered_step(
        self,
        x_t: torch.Tensor,
        y_mouse_lr_up: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        omega: float = 1.0,
        tau: float = 0.1,
        gamma: float = 0.02,
        delta: float = 1e-7,
        apply_lra: bool = True,
    ) -> torch.Tensor:
        batch = x_t.shape[0]
        x_cat = torch.cat([x_t, x_t, x_t], dim=0)
        y_cat = torch.cat([y_mouse_lr_up, y_mouse_lr_up, y_mouse_lr_up], dim=0)
        t_cat = torch.cat([t, t, t], dim=0)
        cond_cat = torch.cat(
            [
                torch.zeros(batch, device=x_t.device, dtype=torch.long),
                torch.ones(batch, device=x_t.device, dtype=torch.long),
                torch.full((batch,), 2, device=x_t.device, dtype=torch.long),
            ],
            dim=0,
        )
        eps_all = self.unet(x_cat, t_cat, y_cat, cond_cat, cond_drop=False)
        eps_m_lr, eps_h_lr, eps_h_hr = eps_all.chunk(3, dim=0)

        v_topo = eps_h_lr - eps_m_lr
        v_detail = eps_h_hr - eps_m_lr
        dot = (v_detail * v_topo).flatten(1).sum(dim=1).view(batch, 1, 1, 1)
        denom = (v_topo * v_topo).flatten(1).sum(dim=1).view(batch, 1, 1, 1) + float(delta)
        eps_steered = eps_m_lr + v_detail - (dot / denom) * v_topo

        gate = self.pathology_soft_gate(y_mouse_lr_up, tau=tau, gamma=gamma)
        eps_final = (1.0 - gate) * eps_steered + gate * eps_m_lr
        eps_final = eps_m_lr + float(omega) * (eps_final - eps_m_lr)

        x0 = self.predict_start_from_noise(x_t, t, eps_final).clamp(0.0, 1.0)
        alpha_next = self.extract(self.alphas_cumprod, t_next, x_t.shape)
        x_prev = torch.sqrt(alpha_next) * x0 + torch.sqrt(1.0 - alpha_next) * eps_final
        if apply_lra:
            x_prev = self.low_resolution_anchor(x_prev, y_mouse_lr_up)
        return x_prev.clamp(0.0, 1.0)

    @torch.no_grad()
    def sample(
        self,
        y_mouse_lr_up: torch.Tensor,
        sampling_steps: int = 50,
        noise_strength: float = 0.2,
        omega: float = 1.0,
        tau: float = 0.1,
        gamma: float = 0.02,
        init_mode: str = "lr_noise",
        apply_lra: bool = True,
    ) -> torch.Tensor:
        batch = y_mouse_lr_up.shape[0]
        if init_mode == "random":
            start_t = self.timesteps - 1
            x = torch.randn_like(y_mouse_lr_up)
        else:
            start_t = int((self.timesteps - 1) * float(noise_strength))
            start_t = max(1, min(self.timesteps - 1, start_t))
            t_start = torch.full((batch,), start_t, device=y_mouse_lr_up.device, dtype=torch.long)
            x = self.q_sample(y_mouse_lr_up, t_start, torch.randn_like(y_mouse_lr_up))
        step_ids = torch.linspace(start_t, 0, sampling_steps, device=y_mouse_lr_up.device).long()
        for index, t_value in enumerate(step_ids):
            t = torch.full((batch,), int(t_value.item()), device=y_mouse_lr_up.device, dtype=torch.long)
            if index == len(step_ids) - 1:
                t_next = torch.zeros_like(t)
            else:
                t_next = torch.full((batch,), int(step_ids[index + 1].item()), device=y_mouse_lr_up.device, dtype=torch.long)
            x = self.ddim_steered_step(x, y_mouse_lr_up, t, t_next, omega=omega, tau=tau, gamma=gamma, apply_lra=apply_lra)
        return x.clamp(0.0, 1.0)
