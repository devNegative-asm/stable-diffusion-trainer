import accelerate
import pickle
import json
import os
import io
import torchvision
import torch
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from diffusers import AutoencoderKL
from connector.store import AspectBucket
device = torch.device('cuda')

parser = ArgumentParser(description="Stable Diffusion Dataset Encoder")
parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    help="The name of the model to encode latents with. Could be HuggingFace ID or a directory",
)
parser.add_argument(
    "--config",
    type=str,
    default="config.json",
    help="The json config which holds the BUCKETS struct.",
)
parser.add_argument(
    "--buckets",
    type=str,
    required=True,
    help="The pickle holding aspect bucket information.",
)
parser.add_argument(
    "--hf_token",
    type=str,
    default=None,
    required=False,
    help="A HuggingFace token is needed to download private models for training.",
)
args = parser.parse_args()

with open(os.environ['SD_TRAINER_CONFIG_FILE'], "r") as config_file:
    config = json.load(config_file)
all_files = [os.path.join(config["DATA_PATH"], filename) for filename in os.listdir(config["DATA_PATH"])]
all_image_files = [instance for instance in all_files if instance.endswith(".png") or instance.endswith(".jpg") or instance.endswith(".webp")]


with open(args.buckets, 'rb') as f:
    bucket: AspectBucket = pickle.load(f)

vae = AutoencoderKL.from_pretrained(
    args.model, subfolder="vae", use_auth_token=args.hf_token
)

vae.requires_grad_(False)
vae = vae.to(device, dtype=torch.float32)

extensions = [".jpg", ".png", ".webp"]

def vae_encode(image_filepath, latent_filepath, aspect_ratio):
    image = Image.open(image_filepath).convert('RGB')
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])
    ])

    image = image.resize(aspect_ratio, resample=Image.Resampling.LANCZOS)
    # transform image to tensor
    tensor = torch.stack([transforms(image)])
    # move tensor to gpu
    tensor = tensor.to(device, memory_format=torch.contiguous_format, dtype=torch.float32).float()
    # encode latent using the vae
    latent = vae.encode(tensor).latent_dist.sample()
    latent = latent * 0.18215
    latent = latent[0]
    torch.save(latent, latent_filepath)

    del latent
    del tensor

time = 0
for aspect_ratio, image_list in bucket.bucket_data.items():
    time += len(image_list)

pbar = tqdm(total=time)

for aspect_ratio, image_list in bucket.bucket_data.items():
    for image_index in image_list:
        image_id = bucket.store.image_files[image_index][1]
        image_file_name = None
        latent_name = os.path.join(config["DATA_PATH"], image_id + ".latent")
        for ext in extensions:
            image_file_name = os.path.join(config["DATA_PATH"], image_id + ext)
            if(os.path.isfile(image_file_name)):
                break
        else:
            raise Exception(f"Extension not found for file {os.path.join(config['DATA_PATH'], image_id)}")
        vae_encode(image_file_name, latent_name, aspect_ratio)
        pbar.update()