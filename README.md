# music-genre-classifier-convNet-keras

</br>

## Overview

Recognizing music genre is a challenging task in the area of music info retrieval. 

In this case we made an idea of using cnn architecture to distinguish music genres. 

And created a end-to-end mel-spectrogram conv2d CNN approach to distinguish between different "Musics" with Keras implementation.

</br>

## Requirement

  * Tensorflow
  * Keras
  * matplotlib.pyplot
  * librosa
  * numpy
  * pandas

</br>

## Data

* `music_analysis.csv`: music file name followed by 8 genre classes.

* [fma_small_zip](https://os.unil.cloud.switch.ch/fma/fma_small.zip): 8,000 tracks of 30s, 8 balanced genres (GTZAN-like) (7.2 GiB)

</br>

The Music genre we are to distinguish are based on [FMA: A DATASET FOR MUSIC ANALYSIS](https://github.com/mdeff/fma)

</br>

Dataset consists of 8 different genres which are:

1. Hip-Hop

2. Pop

3. Folk

4. Experimental

5. Rock

6. International

7. Electronic

8. Instrumental

</br>

## Method

</br>

1. Note that all the mp3 music files are converted to wav. Here we use only 10s of music file out of 30s (you could try using 30 sec if you want to). Input sizes are (128, 431)

2. load the music files using librosa.load. Here is one example of 10 sec mel-spectrogram.

![3](https://user-images.githubusercontent.com/40786348/45913154-84e0c780-be68-11e8-822f-446b3d8334d0.PNG)

3. implement data augmentation (time stretch 2times, pitch shift), and we get around 32000 music files.

4. append all the dataset, randomly split it into 8:1:1 (train, dev, test).

5. train model 

 * 3 conv2d layers followed by average-pooling layer and lastly 2 fully-connected layers with dropout

 * adam optimiser, 64 mini batch_size

</br>

## Result

**-training accuracy**

![1](https://user-images.githubusercontent.com/40786348/46267570-ae88a580-c570-11e8-983f-e2dece209b18.PNG)

**-training loss**

![2](https://user-images.githubusercontent.com/40786348/46267575-b1839600-c570-11e8-81ee-bcb151f6e55c.PNG)

</br>

**-validation accuracy**

![3](https://user-images.githubusercontent.com/40786348/46267576-b34d5980-c570-11e8-8fea-da47b3df3ccc.PNG)

**-validation loss**

![4](https://user-images.githubusercontent.com/40786348/46267578-b5171d00-c570-11e8-90e8-525fb6bdf92c.PNG)

</br>

**achieved around 0.71 accuracy with test set**

</br>

## References

Justin Salamon and Juan Pablo Bello, [Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification](https://arxiv.org/pdf/1608.04363.pdf), 2016

