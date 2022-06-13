import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import tqdm
import scipy.ndimage as nd
from utils import deprocess, preprocess, clip
from torchvision.transforms import ToPILImage


def dream(image, model: torch.nn.Module, iterations, lr):
    """Updates the image to maximize outputs for n iterations"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    image = torch.Tensor(image).requires_grad_(True).to(device)
    image.retain_grad()
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss: torch.Tensor = torch.linalg.norm(out)
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().detach().numpy()


def deep_dream(image: Image, model, iterations, lr, octave_scale, num_octaves):
    """Main deep dream method"""
    image = preprocess(image).unsqueeze(0).cpu().numpy()

    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(
            nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1)
        )

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(
                detail, np.array(octave_base.shape) / np.array(detail.shape), order=1
            )
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return dreamed_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Run deep dream. Either contemplate (train) or dream (see what comes when looking at given image)"
    )

    dream_parser = subparsers.add_parser("dream")
    train_parser = subparsers.add_parser("contemplate")

    dream_parser.add_argument(
        "--input-image",
        type=str,
        default="images/supermarket.jpg",
        help="path to input image",
    )
    dream_parser.add_argument(
        "--iterations",
        default=20,
        type=int,
        help="number of gradient ascent steps per octave",
    )
    dream_parser.add_argument(
        "--at-layer",
        default=15,
        type=int,
        help="layer at which we modify image to maximize outputs",
    )
    dream_parser.add_argument("--lr", default=0.01, help="learning rate")
    dream_parser.add_argument(
        "--octave-scale", default=1.4, type=float, help="image scale between octaves"
    )
    dream_parser.add_argument(
        "--num-octaves", default=10, type=int, help="number of octaves"
    )
    args = parser.parse_args()

    # Load image
    # image = Image.open(args.input_image)
    image = torch.rand(1500, 1000, 3).numpy()

    # Define the model
    network = models.vgg16(pretrained=True)
    layers = list(network.features.children())
    model = nn.Sequential(*layers[: (args.at_layer + 1)])
    if torch.cuda.is_available:
        model = model.cuda()
    print(model)

    # Extract deep dream image
    dreamed_image = deep_dream(
        image,
        model,
        iterations=args.iterations,
        lr=args.lr,
        octave_scale=args.octave_scale,
        num_octaves=args.num_octaves,
    )

    # Save and plot image
    os.makedirs("outputs", exist_ok=True)
    filename = args.input_image.split("/")[-1]
    plt.figure(figsize=(20, 20))
    res = deprocess(dreamed_image)
    plt.imshow(res)
    plt.imsave(f"outputs/output_{filename}", res)
    plt.show()
