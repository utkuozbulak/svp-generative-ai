import pdb
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Lambda, ToPILImage

# Model comes are here
from cls_model import Unet
# Scheduler comes are here
from funct_scheduler import linear_beta_schedule, cosine_beta_schedule
# Forward and backward difussion functions are here
from funct_diffusion import get_params, generate_images

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def sample_images_from_diffusion(diffusion_steps, nth_step, reverse_transform):
    sampled_images = []
    total_steps = len(diffusion_steps)
    no_of_generations = len(diffusion_steps[0])

    # Get images at the nth step and the final one
    step_indices = [0, 200, 400, 450, 480, 499]

    # Run for each image
    for im_id in range(no_of_generations):
        sampled_images.append([])
        for idx in step_indices:
            selected_image = diffusion_steps[idx][im_id]
            # Take the final image from the N images at that step
            pil_image = reverse_transform(selected_image.cpu())
            sampled_images[-1].append(pil_image)

    return sampled_images


def save_images_side_by_side(images, save_path, gap=5, gap_color=(0, 0, 0)):
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths) + gap * (len(images) - 1)
    max_height = max(heights)

    stitched_image = Image.new('RGB', (total_width, max_height), color=gap_color)

    x_offset = 0
    for img in images:
        stitched_image.paste(img, (x_offset, 0))
        x_offset += img.width + gap

    stitched_image.save(save_path)


if __name__ == "__main__":
    selected_object = 'silicone_oil'

    rseed = 55
    torch.manual_seed(rseed)
    # Output saving parameters
    output_folder = Path("../../diffusion_out/" + selected_object)

    # Diffusion parameters
    timesteps = 500
    selected_beta_scheduler = linear_beta_schedule
    diffusion_param_dict = get_params(timesteps, selected_beta_scheduler)

    # Dataset/Dataloader parameters
    channels = 1
    image_size = 64
    image_cnt = 10

    # Load model
    model_folder = Path('../../models')
    model_name = 'silicone_oil_diffusion_model.pth'
    model_path = model_folder / model_name

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4, 8)
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Load weights
    model.eval()
    model.cuda()

    # Reverse transforms for image creation
    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: torch.clamp(t, 0, 255)),  # <-- Clamp here
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage()
    ])

    # Generate images
    generated_ims = generate_images(diffusion_param_dict,
                                    model,
                                    image_size=image_size,
                                    image_cnt=image_cnt,
                                    channels=channels)

    # Sample images to see denoising
    sample_every_nth_step = 100
    selected_ims = sample_images_from_diffusion(generated_ims,
                                                sample_every_nth_step,
                                                reverse_transform)
    # Create the output folder
    final_output = output_folder / 'final'
    stit_output = output_folder / 'stitched'
    # Create folders, both of them
    final_output.mkdir(parents=True, exist_ok=True)
    stit_output.mkdir(parents=True, exist_ok=True)

    for image_id in range(image_cnt):
        save_images_side_by_side(selected_ims[image_id], stit_output / f'{rseed}_stitched_{image_id}.png')
        selected_ims[image_id][-1].save(final_output/ f'{rseed}_final_{image_id}.png')
