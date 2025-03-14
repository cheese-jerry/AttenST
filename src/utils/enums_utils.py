import torch
# from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline  hb comment

# load the local pipeline for inference
from src.pipes.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from src.pipes.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline

from src.eunms import Model_Type, Scheduler_Type
from src.schedulers.euler_scheduler import MyEulerAncestralDiscreteScheduler
from src.schedulers.lcm_scheduler import MyLCMScheduler
from src.schedulers.ddim_scheduler import MyDDIMScheduler
from src.pipes.sdxl_inversion_pipeline import SDXLDDIMPipeline
from src.pipes.sdxl_controlnet_inversion_pipeline import SDXLDDIMCONTROLNETPipeline
from src.pipes.sd_inversion_pipeline import SDDDIMPipeline


def scheduler_type_to_class(scheduler_type):
    if scheduler_type == Scheduler_Type.DDIM:
        return MyDDIMScheduler
    elif scheduler_type == Scheduler_Type.EULER:
        return MyEulerAncestralDiscreteScheduler
    elif scheduler_type == Scheduler_Type.LCM:
        return MyLCMScheduler
    else:
        raise ValueError("Unknown scheduler type")


def is_stochastic(scheduler_type):
    if scheduler_type == Scheduler_Type.DDIM:
        return False
    elif scheduler_type == Scheduler_Type.EULER:
        return True
    elif scheduler_type == Scheduler_Type.LCM:
        return True
    else:
        raise ValueError("Unknown scheduler type")


def model_type_to_class(model_type):
    if model_type == Model_Type.SDXL:
        print("load SDXL...")
        return StableDiffusionXLImg2ImgPipeline, SDXLDDIMPipeline
    elif model_type == Model_Type.SDXL_Turbo:
        return StableDiffusionXLImg2ImgPipeline, SDXLDDIMPipeline
    elif model_type == Model_Type.LCM_SDXL:
        return StableDiffusionXLImg2ImgPipeline, SDXLDDIMPipeline
    elif model_type == Model_Type.SD15:
        return StableDiffusionImg2ImgPipeline, SDDDIMPipeline
    elif model_type == Model_Type.SD14:
        return StableDiffusionImg2ImgPipeline, SDDDIMPipeline
    elif model_type == Model_Type.SD21:
        return StableDiffusionImg2ImgPipeline, SDDDIMPipeline
    elif model_type == Model_Type.SD21_Turbo:
        return StableDiffusionImg2ImgPipeline, SDDDIMPipeline
    elif model_type == Model_Type.SDXLCN:
        print("load SDXLControlNetImg2Img...")
        return StableDiffusionXLControlNetImg2ImgPipeline, SDXLDDIMCONTROLNETPipeline
    else:
        raise ValueError("Unknown model type")


def model_type_to_model_name(model_type):
    if model_type == Model_Type.SDXL:
        return "stabilityai/stable-diffusion-xl-base-1.0"
    elif model_type == Model_Type.SDXL_Turbo:
        return "stabilityai/sdxl-turbo"
    elif model_type == Model_Type.LCM_SDXL:
        return "stabilityai/stable-diffusion-xl-base-1.0"
    elif model_type == Model_Type.SD15:
        return "runwayml/stable-diffusion-v1-5"
    elif model_type == Model_Type.SD14:
        return "CompVis/stable-diffusion-v1-4"
    elif model_type == Model_Type.SD21:
        return "stabilityai/stable-diffusion-2-1"
    elif model_type == Model_Type.SD21_Turbo:
        return "stabilityai/sd-turbo"
    if model_type == Model_Type.SDXLCN:
        return "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        raise ValueError("Unknown model type")


def model_type_to_size(model_type):
    if model_type == Model_Type.SDXL:
        return (1024, 1024)
    elif model_type == Model_Type.SDXL_Turbo:
        return (512, 512)
    elif model_type == Model_Type.LCM_SDXL:
        return (768, 768)  # TODO: check
    elif model_type == Model_Type.SD15:
        return (512, 512)
    elif model_type == Model_Type.SD14:
        return (512, 512)
    elif model_type == Model_Type.SD21:
        return (512, 512)
    elif model_type == Model_Type.SD21_Turbo:
        return (512, 512)
    else:
        raise ValueError("Unknown model type")


def is_float16(model_type):
    if model_type == Model_Type.SDXL:
        return True
    elif model_type == Model_Type.SDXL_Turbo:
        return True
    elif model_type == Model_Type.LCM_SDXL:
        return True
    elif model_type == Model_Type.SD15:
        return False
    elif model_type == Model_Type.SD14:
        return False
    elif model_type == Model_Type.SD21:
        return False
    elif model_type == Model_Type.SD21_Turbo:
        return False
    elif model_type == Model_Type.SDXLCN:
        return True
    else:
        raise ValueError("Unknown model type")


def is_sd(model_type):
    if model_type == Model_Type.SDXL:
        return False
    elif model_type == Model_Type.SDXL_Turbo:
        return False
    elif model_type == Model_Type.LCM_SDXL:
        return False
    elif model_type == Model_Type.SDXLCN:
        return False
    elif model_type == Model_Type.SD15:
        return True
    elif model_type == Model_Type.SD14:
        return True  # hb edit  True->False
    elif model_type == Model_Type.SD21:
        return True
    elif model_type == Model_Type.SD21_Turbo:
        return True
    else:
        raise ValueError("Unknown model type")


def _get_pipes(model_type, device, controlnet=None, image_encoder=None): # ,clip_model=None
    model_name = model_type_to_model_name(model_type)
    pipeline_inf, pipeline_inv = model_type_to_class(model_type)

    if is_float16(model_type):
        if controlnet is not None:
            print("loading fp16 model with ControlNet.......")
            pipe_inference = pipeline_inf.from_pretrained(
                model_name,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                safety_checker=None,
            ).to(device)
        elif image_encoder is not None:
            print("loading fp16 model with image_encoder.......")
            pipe_inference = pipeline_inf.from_pretrained(
                model_name,
                image_encoder=image_encoder,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
            ).to(device)
        else:
            print("loading fp16 model without image_encoder.......")
            pipe_inference = pipeline_inf.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None
            ).to(device)
    else:
        print("loading fp32 model.......")
        pipe_inference = pipeline_inf.from_pretrained(
            model_name,
            use_safetensors=True,
            safety_checker=None
        ).to(device)

    # print(pipe_inference.components)
    pipe_inversion = pipeline_inv(**pipe_inference.components)

    return pipe_inversion, pipe_inference


def get_pipes(model_type, scheduler_type, device="cuda", controlnet=None, image_encoder=None):#,clip_model=None
    scheduler_class = scheduler_type_to_class(scheduler_type)
    if controlnet is None:
        pipe_inversion, pipe_inference = _get_pipes(model_type, device, None,image_encoder=image_encoder) # ,clip_model=clip_model
    else:
        pipe_inversion, pipe_inference = _get_pipes(model_type, device, controlnet)

    pipe_inference.scheduler = scheduler_class.from_config(pipe_inference.scheduler.config)
    pipe_inversion.scheduler = scheduler_class.from_config(pipe_inversion.scheduler.config)

    if is_sd(model_type):
        pipe_inference.scheduler.add_noise = lambda init_latents, noise, timestep: init_latents
        pipe_inversion.scheduler.add_noise = lambda init_latents, noise, timestep: init_latents

    if model_type == Model_Type.LCM_SDXL:
        adapter_id = "latent-consistency/lcm-lora-sdxl"
        pipe_inversion.load_lora_weights(adapter_id)
        pipe_inference.load_lora_weights(adapter_id)

    return pipe_inversion, pipe_inference
