# WikiText-2 Language Modelling

This example trains and benchmarks a language model on the WikiText-2 dataset.

## Requirements

The requirements for this example, aside from attorch and its dependencies, are,

* ```datasets==2.18.0```
* ```transformers==4.39.0```

## Training

To run this example, please run ```python -m examples.wikitext-2.main``` from the root directory. The arguments are as follows.

* ```--model```: Name of language model to train. The only option is ```gpt2```.
* ```--epochs```: Number of epochs to train for.
* ```--batch_size```: Batch size.
* ```--seq_len```: Sequence length.
* ```--num_workers```: Number of workers for data loading.
