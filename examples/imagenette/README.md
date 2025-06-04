# Imagenette Image Classification

This example trains and benchmarks a vision model on the Imagenette classification dataset.

## Requirements

The requirement for this example, aside from attorch and its dependencies, is,

* ```torchvision==0.19.0```

## Training

To run this example, please run ```python -m examples.imagenette.main``` from the root directory. The arguments are as follows.
* ```--model```: Name of vision model to train. Options are ```resnet50```, ```resnet101```, ```resnet152```, ```convnext_atto```, ```convnext_femto```, ```convnext_pico```, ```convnext_nano```, ```convnext_tiny```, ```convnext_small```, ```convnext_base```, ```convnext_large```, ```convnext_xlarge```, `vit_tiny_patch16`, `vit_small_patch32`, `vit_small_patch16`, `vit_small_patch8`, `vit_base_patch32`, `vit_base_patch16`, `vit_base_patch8`, `vit_large_patch32`, `vit_large_patch16`, and `vit_large_patch14`.
* ```--epochs```: Number of epochs to train for.
* ```--batch_size```: Batch size.
* ```--center_crop_size```: Center crop size for validation.
* ```--image_size```: Input image size.
* ```--num_workers```: Number of workers for data loading.