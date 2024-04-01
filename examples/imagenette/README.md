# Imagenette Image Classification

This example trains and benchmarks a vision model on the Imagenette classification dataset.

## Requirements

The requirement for this example, aside from attorch and its dependencies, is,

* ```torchvision==0.17.0```

## Training

To run this example, please run ```python -m examples.imagenette.main``` from the root directory. The arguments are as follows.

* ```--model```: Name of vision model to train. Options are ```resnet50```, ```resnet101```, ```resnet152```, ```convnext_tiny```, ```convnext_small```, ```convnext_base```, ```convnext_large```, ```convnext_xlarge```.
* ```--epochs```: Number of epochs to train for.
* ```--batch_size```: Batch size.
* ```--center_crop_size```: Center crop size for validation.
* ```--image_size```: Input image size.
* ```--num_workers```: Number of workers for data loading.