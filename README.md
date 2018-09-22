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

## Instruction

</br>

1. Note that all the mp3 music files are converted to wav. Here we use only 10s of music file out of 30s (you could try using 30 sec if you want to). Input sizes are (128, 431)

2. load the music files using librosa.load. Here is one example of 10 sec mel-spectrogram.

![3](https://user-images.githubusercontent.com/40786348/45913154-84e0c780-be68-11e8-822f-446b3d8334d0.PNG)

2. assign all the genre classes a numbers ranging from 0 to 7 (pandas is used).

3. implement data augmentation (time stretch 2times, pitch shift), and we get around 32000 music files.

4. append all the dataset, randomly split it into 8:1:1 (train, dev, test).

</br>

## Training Result

![1](https://user-images.githubusercontent.com/40786348/45912694-945c1280-be60-11e8-9669-dd25ef3787e5.PNG)

![2](https://user-images.githubusercontent.com/40786348/45912695-96be6c80-be60-11e8-8f31-31d8c8f22ac9.PNG)

</br>


## References

Justin Salamon and Juan Pablo Bello, [Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification](https://arxiv.org/pdf/1608.04363.pdf), 2016

