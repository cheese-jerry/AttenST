import torch
import torch.nn.functional as F

# Based on code from https://github.com/pix2pixzero/pix2pix-zero
def noise_regularization(
        e_t, noise_pred_optimal, lambda_kl, lambda_ac, num_reg_steps, num_ac_rolls, generator=None
):
    for _outer in range(num_reg_steps):
        if lambda_kl > 0:
            _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
            l_kld = patchify_latents_kl_divergence(_var, noise_pred_optimal)
            l_kld.backward()
            _grad = _var.grad.detach()
            _grad = torch.clip(_grad, -100, 100)
            e_t = e_t - lambda_kl * _grad
        if lambda_ac > 0:
            for _inner in range(num_ac_rolls):
                _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                l_ac = auto_corr_loss(_var, generator=generator)
                l_ac.backward()
                _grad = _var.grad.detach() / num_ac_rolls
                e_t = e_t - lambda_ac * _grad
        e_t = e_t.detach()

    return e_t


# Based on code from https://github.com/pix2pixzero/pix2pix-zero
def auto_corr_loss(
        x, random_shift=True, generator=None
):
    B, C, H, W = x.shape
    assert B == 1
    x = x.squeeze(0)
    reg_loss = 0.0
    for ch_idx in range(x.shape[0]):
        noise = x[ch_idx][None, None, :, :]
        while True:
            if random_shift:
                roll_amount = torch.randint(0, noise.shape[2] // 2, (1,), generator=generator).item()
            else:
                roll_amount = 1
            reg_loss += (
                                noise * torch.roll(noise, shifts=roll_amount, dims=2)
                        ).mean() ** 2
            reg_loss += (
                                noise * torch.roll(noise, shifts=roll_amount, dims=3)
                        ).mean() ** 2
            if noise.shape[2] <= 8:
                break
            noise = F.avg_pool2d(noise, kernel_size=2)
    return reg_loss


def patchify_latents_kl_divergence(x0, x1, patch_size=4, num_channels=4):
    def patchify_tensor(input_tensor):
        patches = (
            input_tensor.unfold(1, patch_size, patch_size)
            .unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
        )
        patches = patches.contiguous().view(-1, num_channels, patch_size, patch_size)
        return patches

    x0 = patchify_tensor(x0)
    x1 = patchify_tensor(x1)

    kl = latents_kl_divergence(x0, x1).sum()
    return kl


def latents_kl_divergence(x0, x1):
    EPSILON = 1e-6
    x0 = x0.view(x0.shape[0], x0.shape[1], -1)
    x1 = x1.view(x1.shape[0], x1.shape[1], -1)
    mu0 = x0.mean(dim=-1)
    mu1 = x1.mean(dim=-1)
    var0 = x0.var(dim=-1)
    var1 = x1.var(dim=-1)
    kl = (
            torch.log((var1 + EPSILON) / (var0 + EPSILON))
            + (var0 + (mu0 - mu1) ** 2) / (var1 + EPSILON)
            - 1
    )
    kl = torch.abs(kl).sum(dim=-1)
    return kl

def inversion_step(
        pipe,
        z_t: torch.tensor,
        t: torch.tensor,
        prompt_embeds,
        added_cond_kwargs,
        resample_steps: int = 100,
        first_step_max_timestep: int = 250,
        generator=None,
        callback_t=None,
) -> torch.tensor:
    extra_step_kwargs = {}
    avg_range = pipe.cfg.average_first_step_range if t.item() < first_step_max_timestep else pipe.cfg.average_step_range
    resample_steps = min(pipe.cfg.max_resample_steps_first_step,
                            resample_steps) if t.item() < first_step_max_timestep else resample_steps

    nosie_pred_avg = None
    noise_pred_optimal = None
    z_tp1_forward = pipe.scheduler.add_noise(pipe.z_0, pipe.noise, t.view((1))).detach()  
    approximated_z_tp1 = z_t.clone() 

    for i in range(resample_steps + 1):  # i = 1,2,3....k

        if callback_t is not None:
            callback_t(t)  # get current timestep
            #print("pass the t to the hook")

        with torch.no_grad():
            prompt_embeds_in = prompt_embeds 
            added_cond_kwargs_in = added_cond_kwargs
            noise_pred = unet_pass(pipe, approximated_z_tp1, t, prompt_embeds_in, added_cond_kwargs_in)

            # perform guidance
            if pipe.do_classifier_free_guidance: 
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Calculate average noise
            if i >= avg_range[0] and i < avg_range[1]:  
                j = i - avg_range[0]
                if nosie_pred_avg is None:
                    nosie_pred_avg = noise_pred.clone()
                else:
                    nosie_pred_avg = j * nosie_pred_avg / (j + 1) + noise_pred / (j + 1)

        if i >= avg_range[0] or (not pipe.cfg.average_latent_estimations and i > 0): 
            pass
            noise_pred = noise_regularization(noise_pred, noise_pred_optimal,
                                              lambda_kl=pipe.cfg.noise_regularization_lambda_kl,
                                              lambda_ac=pipe.cfg.noise_regularization_lambda_ac,
                                              num_reg_steps=pipe.cfg.noise_regularization_num_reg_steps,
                                              num_ac_rolls=pipe.cfg.noise_regularization_num_ac_rolls,
                                              generator=generator)

        approximated_z_tp1 = pipe.scheduler.inv_step(noise_pred, t, z_t, **extra_step_kwargs, return_dict=False)[
            0].detach()

    
    if pipe.cfg.average_latent_estimations and nosie_pred_avg is not None:
        nosie_pred_avg = noise_regularization(nosie_pred_avg, noise_pred_optimal,
                                              lambda_kl=pipe.cfg.noise_regularization_lambda_kl,
                                              lambda_ac=pipe.cfg.noise_regularization_lambda_ac,
                                              num_reg_steps=pipe.cfg.noise_regularization_num_reg_steps,
                                              num_ac_rolls=pipe.cfg.noise_regularization_num_ac_rolls,
                                              generator=generator)
        approximated_z_tp1 = pipe.scheduler.inv_step(nosie_pred_avg, t, z_t, **extra_step_kwargs, return_dict=False)[
            0].detach()

    return approximated_z_tp1


@torch.no_grad()
def unet_pass(pipe, z_t, t, prompt_embeds, added_cond_kwargs):
    latent_model_input = torch.cat([z_t] * 2) if pipe.do_classifier_free_guidance else z_t
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    return pipe.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=None,
        cross_attention_kwargs=pipe.cross_attention_kwargs,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]
