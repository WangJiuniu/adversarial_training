# Adversarial Training
Adversarial training methods for text classification (sentiment analysing on IMDB dataset).

## Based on
1. Paper [Adversarial training methods for semi-supervised text classification, ICLR 2017, Miyato T., Dai A., Goodfellow I.
](https://arxiv.org/abs/1605.07725)
**Only adversarial training has been implemented.**
2. Github repository [Adversarial Training Methods](https://github.com/enry12/adversarial_training_methods). This is another implementation using tensorflow.

## Requirements
This repository has been tested under python 3.6 and Pytorch 0.4.1 with GPU.

## Usage
1. Download [preprocessed IMDB dataset](https://drive.google.com/open?id=1Ro1uAayY6CzHXiaYqwohzNP5M3qGeGrQ) for this repository (you can also find the URL in imdb/google_drive.txt). And then uncompressing these files into directory *imdb*. Of course, you can try to generate these files under the guidance of [Adversarial Training Methods](https://github.com/enry12/adversarial_training_methods).
2. Run the main function in at_pytorch/run.py:
```shell
$ cd ./at_pytorch
$ python3 run.py
```
## Results
The running result can be seen in file at_pytorch/standard_result.txt, and brief description is as following:

Method | Seq. Length | Epochs | Accuracy
:------: | :-----------: | :------: | :--------:
baseline | 400 | 10 | 0.854 
adversarial | 400 | 10 | 0.871  

We have not get the results reported by the original paper, but our result shows the effective of adversarial training.
