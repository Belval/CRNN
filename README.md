# CRNN

A TensorFlow implementation of https://github.com/bgshih/crnn

## But what is a CRNN?

It is a Convolutional Recurrent Neural Network that can be used as an OCR

## Requirements

- Tensorflow (tested with 1.8) `pip3 install tensorflow`
- Scipy `pip3 install scipy`

## What training data was used?

All training data (200 000 examples) were generated using my other project https://github.com/Belval/TextRecognitionDataGenerator

To do the same, simply clone that project and do `python3 run.py -c 200000 -w 1 -t 8`. `-t` should be your processor thread count.

## Pretrained model

Available in CRNN/save. Use `python3 run.py -ex ../data/test --test --restore` to test.

## Results

It works but is a suboptimal solution for OCR in its current form as it makes some mistakes. Do note that I used a bigger char vector than the paper.

For fun, here are a list of words with their prediction:

| Ground truth 	| Prediction 	| Image 	|
|--------------	|------------	|-------	|
| retardates 	| retardates 	| ![1](samples/1.jpg "1") 	|
| pay-roller 	| poy-roler 	| ![2](samples/2.jpg "2") 	|
| rhizopodist | rhizospodist |  ![3](samples/3.jpg "3")	|
| theriacas | trenagas |  ![4](samples/4.jpg "4")	|
| semantically | semanticaly |  ![5](samples/5.jpg "5")	|
| dualistic | duaistic |  ![6](samples/6.jpg "6")	|
| high-flying | highi-fling | ![7](samples/7.jpg "7") 	|
| grossify | grsity | ![8](samples/8.jpg "8") 	|
| scutular | scutular |  ![9](samples/9.jpg "9")	|
| crispened | crispened | ![10](samples/10.jpg "10") 	|
