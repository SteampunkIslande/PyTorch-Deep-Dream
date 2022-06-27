# Pytorch - Deep Dream
A basic PyTorch implementation of Deep Dream.
All credits go to 

## Examples

Training a new model, assuming dataset has the following structure:

dataset
  |-Label1
  |---Image 1
  |---Image 2
  |-Label2
  |---Image 1
  |---Image 2

```bash
$ python deep_dream.py train --model-name vgg16 --dataset-path dataset
```

```
$ python3 deep_dream.py dream
```
