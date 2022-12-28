import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from src.utils import ImageFolder, VAEHandler, denormalize, preprocess_images


# batch size, setting a value bigger than 1 is meaningless
# because calculate loss value pers 1 image whatever
BATCH_SIZE = 1

# image size, after preprocessing
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# project root
# ROOT_DIR = os.path.join(os.environ['PATH'], "evaluate-images-to-feed-diffusion")
ROOT_DIR = Path("./")

# model directory
# MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_DIR = ROOT_DIR / "models"

# vae model directory
# VAE_DIR = os.path.join(MODEL_DIR, "waifu-diffusion-v1-4")
VAE_DIR = MODEL_DIR / "waifu-diffusion-v1-4"

# focal model directory (to be used to crop images nicely)
# FOCAL_MODEL_DIR = os.path.join(MODEL_DIR, "focal")
FOCAL_MODEL_DIR = MODEL_DIR / "focal"

# raw image directory
# IMAGE_SOURCE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_SOURCE_DIR = ROOT_DIR / "images"

# processed image directory
# IMAGE_PREPROCESSED_DIR = os.path.join(ROOT_DIR, "processed")
IMAGE_PREPROCESSED_DIR = ROOT_DIR / "processed"


# Preprocess Images
# Crop and convert images suitable for feeding model.

# If you do not leave focal_model_dir=None, focal model is automatically downloaded.

# Then, images are cropped in consideration of where the face / focal point is.
preprocess_images(
    IMAGE_SOURCE_DIR, 
    IMAGE_PREPROCESSED_DIR, 
    width=IMAGE_WIDTH, 
    height=IMAGE_HEIGHT, 
    focal_model_dir=FOCAL_MODEL_DIR,
)

# Load VAE
vae_waifu_1_4 = VAEHandler(VAE_DIR)

# Prepare Evaluation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 1)])
dataset = ImageFolder(IMAGE_PREPROCESSED_DIR, transform).make_iterator(batch_size=BATCH_SIZE, shuffle=False)

# Evaluate
# The return value res has:

# Normalized tensor of original images
# Latent z
# Reconstructed tensors from z
# Loss values of each images
res = vae_waifu_1_4.get_loss_results(dataset)

# Loss
# If there are some images whose loss value are quite high, model might not be able to learn the expressions of it well.
df = res.df.copy(deep=True)
df.sort_index().plot(x="idx", y="loss", xlabel="image_idx", ylabel="loss", figsize=(7, 5))
plt.show()

# Visualize Results
# Worst and Best
res.plot_most_and_least_lossy_images(n=5)

# All - descending order
for i in res.df.index:
    plt.imshow(denormalize(np.array([res.rec[i]]))[0])
    plt.title(f"loss: {res.loss[i]}")
    plt.show()
