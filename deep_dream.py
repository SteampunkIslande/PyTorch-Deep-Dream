import typing
import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import tqdm
import scipy.ndimage as nd
from utils import deprocess, preprocess, clip
from dataset import DreamDataset

import datetime


def dream(image, model: torch.nn.Module, iterations: int, lr: float, target: torch.Tensor):
    """Updates the image to maximize outputs for n iterations"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    image = torch.Tensor(image).requires_grad_(True).to(device)

    image.retain_grad()
    loss_function = torch.nn.BCELoss()
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


def deep_dream(
    image: Image,
    model: torch.nn.Module,
    iterations: int,
    lr: float,
    octave_scale: float,
    num_octaves: int,
    target: torch.Tensor,
):
    """Main deep dream method"""

    image = preprocess(image).cpu().numpy()

    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr, target)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return dreamed_image


def transfer_vgg16(
    pretrained: bool = True, label_count: int = 2
) -> typing.Tuple[torch.nn.Module, torch.nn.Module]:
    """Example of transfer_* function. Returns two models for use with transfer learning.
    Usage:
        >>> head,trainable = transfer_vgg16()
        >>> input_tensor = torch.nn.rand()
        >>> output = trainable(head(input_tensor))

    The idea is that you get two models, one being the pretrained backbone (or head) and the other being the trainable tail.

    So it's easy to pass trainable parameters only to the optimizer

    Important: Assumes input size is (1024,1024)
    """
    model = torchvision.models.vgg16(pretrained=pretrained)
    children = list(model.children())
    head = children[0][:28]
    trainable = torch.nn.Sequential(
        *children[0][28:],  # Train last layers of vgg16
        torch.nn.Flatten(),  # Flatten the result for Fully Connected layers
        torch.nn.Linear(in_features=524288, out_features=label_count),  # FC1
        # torch.nn.Linear(in_features=512, out_features=label_count),  # FC2
        torch.nn.Sigmoid(),
    )
    return head, trainable


def train(
    model_name: str,
    dataset_path: str,
    batch_size: int = 4,
    epochs: int = 100,
    learning_rate: float = 0.1,
    log_interval: int = 10,
    save_interval: int = 100,
    save_path: str = "checkpoints",
    from_checkpoint: str = None,
):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_names = {
        "vgg16": transfer_vgg16,
        # "alexnet": torchvision.models.alexnet,
        # "resnet18": torchvision.models.resnet18,
    }

    # Initialize dataset
    dataset = DreamDataset(dataset_path)

    if model_name not in model_names:
        print(f"{model_name} is not currently supported, but you can add it to model_names")
        return 1

    # Initialize model (with pretrained and trainable parts)
    label_count = dataset.label_count
    print(f"Initializing model with {label_count} labels")
    head, trainable = model_names[model_name](label_count=label_count)

    # Set trainable part in training mode
    trainable.train()
    # Initialize opitimizer and loss function
    optimizer = torch.optim.Adamax(trainable.parameters(), lr=learning_rate)

    start_epoch = 0

    if from_checkpoint:
        with open(from_checkpoint, "rb") as f:
            ckpt = torch.load(f, "cpu")

            assert (
                ckpt["label_count"] == label_count
            ), f"Trained a model on {ckpt['label_count']} labels, cannot train it on {label_count} labels"

            assert (
                ckpt["optimizer_name"] == optimizer.__class__.__name__
            ), f"Cannot initialize {optimizer.__class__.__name__} with {ckpt['optimizer_name']} state dict"
            optimizer.load_state_dict(ckpt["optimizer_dict"])

            assert (
                ckpt["model_name"] == model_name
            ), f"Cannot initialize {model_name} with {ckpt['model_name']} state dict"
            trainable.load_state_dict(ckpt["trainable_dict"])

            start_epoch = ckpt["epoch"] if "epoch" in ckpt else 0

    loss_function = torch.nn.BCELoss()

    train_loader = DataLoader(dataset, batch_size=batch_size)

    model = torch.nn.Sequential(head, trainable)

    model = model.to(device)

    print("Training started")

    with open("log", "a") as log_file:

        start_timer = datetime.datetime.today()

        for epoch in tqdm.tqdm(range(start_epoch, epochs)):

            for batch_idx, (image, target) in enumerate(train_loader):
                image, target = image.to(device), target.to(device)
                optimizer.zero_grad()
                output: torch.Tensor = model(image)

                loss: torch.Tensor = loss_function(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx == len(train_loader) and epoch % log_interval == 0:
                    log_file.write(f"{epoch}:{loss.item()}" + "\n")

            if epoch % save_interval == 0 and epoch > 0:
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                    assert os.path.isdir(
                        save_path
                    ), f"Could not create checkpoint save directory {save_path}"

                torch.save(
                    {
                        "optimizer_name": optimizer.__class__.__name__,
                        "optimizer_dict": optimizer.state_dict(destination=torch.device("cpu")),
                        "model_name": model_name,
                        "pretrained": True,
                        "trainable_dict": trainable.state_dict(destination=torch.device("cpu")),
                        "label_count": label_count,
                        "epoch": epoch,
                    },
                    os.path.join(save_path, f"ckpt-epoch{epoch}.tar"),
                )

        end_timer = datetime.datetime.today()

    h, m, s = str(end_timer - start_timer).split(":")
    print(f"Training complete ! (Took {h} hours, {m} minutes, {s} seconds)")
    # Just save the model as is
    torch.save(model.to(torch.device("cpu")), "final_model.pth")
    return 0


def test_dream(
    model_path: str,
    input_image: str,
    iterations: int,
    learning_rate: float,
    expected_labels: typing.Tuple[float],
    octave_scale: float = 1.4,
    num_octaves: int = 10,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(input_image)

    assert image.size == (
        1024,
        1024,
    ), "Currently, this program only supports images of shape 1024*1024"

    # Define the model
    with open(model_path, "rb") as f:
        base_model: torch.nn.Module = torch.load(f, map_location=device)
        head, tail = list(base_model.children())
        model = torch.nn.Sequential(*head, *tail[:4])
        model.eval()

    target = torch.tensor(expected_labels, dtype=torch.float32)
    target.unsqueeze_(0)
    target = target.to(device)
    target.requires_grad_(True)

    # Extract deep dream image
    dreamed_image = deep_dream(
        image,
        model,
        iterations=iterations,
        lr=learning_rate,
        octave_scale=octave_scale,
        num_octaves=num_octaves,
        target=target,
    )

    # Save and plot image
    os.makedirs("outputs", exist_ok=True)
    filename = input_image.split("/")[-1]
    plt.figure(figsize=(20, 20))
    res = deprocess(dreamed_image)
    plt.imshow(res)
    plt.imsave(f"outputs/output_{filename}", res)
    plt.show()
    return 0


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        help="Run deep dream. Either train or dream (see what comes when looking at given image)"
    )

    dream_parser = subparsers.add_parser("dream")
    train_parser = subparsers.add_parser("train")

    dream_parser.add_argument(
        "--input-image",
        type=str,
        default="images/supermarket.jpg",
        help="Path to input image",
    )
    dream_parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of gradient ascent steps per octave",
    )
    dream_parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    dream_parser.add_argument(
        "--octave-scale", type=float, default=1.4, help="Image scale between octaves"
    )
    dream_parser.add_argument("--num-octaves", default=10, type=int, help="Number of octaves")
    dream_parser.add_argument(
        "--model-path", help="Path to a pth file with complete model to use", required=True
    )
    dream_parser.add_argument(
        "--expected-labels",
        help="Name of the label to dream of (must be in the dataset folder)",
        type=eval,
        required=True,
    )

    dream_parser.set_defaults(func=test_dream)

    train_parser.add_argument(
        "--model-name",
        default="vgg16",
        help="Name of a pretrained model to use to train",
        required=True,
        choices=("vgg16",),
    )
    train_parser.add_argument(
        "--dataset-path",
        help="Path to a dataset folder containing one subfolder per label",
        default="dataset",
    )
    train_parser.add_argument("--batch-size", help="How many images per batch", type=int, default=4)
    train_parser.add_argument(
        "--epochs", help="How many epochs to train the model for", type=int, default=100
    )
    train_parser.add_argument(
        "--learning-rate", help="Learning rate for training", type=float, default=0.1
    )
    train_parser.add_argument(
        "--log-interval",
        help="How many epochs between each progress update",
        type=int,
        default=10,
    )
    train_parser.add_argument(
        "--save-interval", help="How many epochs between checkpoint saves", type=int, default=100
    )
    train_parser.add_argument(
        "--save-path", help="Folder to save checkpoints to", default="checkpoints"
    )
    train_parser.add_argument(
        "--from-checkpoint", help="Path to a checkpoint to resume training from", default=None
    )

    train_parser.set_defaults(func=train)

    args = vars(parser.parse_args())
    func = args["func"]
    del args["func"]
    return func(**args)


if __name__ == "__main__":
    exit(main())
