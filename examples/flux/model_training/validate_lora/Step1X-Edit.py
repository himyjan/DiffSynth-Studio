import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from PIL import Image


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen2.5-VL-7B-Instruct"),
        ModelConfig(model_id="stepfun-ai/Step1X-Edit", origin_file_pattern="step1x-edit-i1258.safetensors"),
        ModelConfig(model_id="stepfun-ai/Step1X-Edit", origin_file_pattern="vae.safetensors"),
    ],
)
pipe.load_lora(pipe.dit, "models/train/Step1X-Edit_lora/epoch-4.safetensors", alpha=1)

image = pipe(
    prompt="Make the dog turn its head around.",
    step1x_reference_image=Image.open("data/example_image_dataset/2.jpg").resize((768, 768)),
    height=768, width=768, cfg_scale=6,
    seed=0
)
image.save("image_Step1X-Edit_lora.jpg")
