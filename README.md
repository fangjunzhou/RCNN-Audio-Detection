# RCNN-Audio-Detection

## Use `generate-data-set.ipynb` to Generate Data Set

The `generate-data-set.ipynb` notebook contains following steps:

### Read Data Set

- Store all the training data in DATA_PATH
- Use the same name for audio file and label file, for example:
  - Audio file name: "audio-1.wav"
  - Label file name: "audio-1.json"
- For more details about audio and label format, see [Audio-Spectrum-Labeling-Toolset](https://github.com/Fangjun-Zhou/Audio-Spectrum-Labeling-Toolset)

### Preprocess

- Read in audio data
- Convert the audio into spectrogram
- Slice the spectrogram into overlapping windows
- TIME_SCALE is the scale of time span for each window. For example, if TIME_SCALE is 1, then each window will be a square. The length of each window will be equal to the height of audioSpectrogram.
  - For species making longer sounds, increase TIME_SCALE to make sure their audio fits in the window.

### Spectrum Enhancement

The spectrogram is enhanced using following methods:

- Normalize the spectrogram to 0-1:
  - Subtract the entire spectrogram by the minimum value in the spectrogram so that the minimum value is 0.
  - Divide the entire spectrogram by the maximum value in the spectrogram so that the maximum value is 1.
- Enhance the spectrogram using $f(x) = 1 - (1-x)^{\text{ENHANCE-FACTOR}}$

### Sliding Time Window

After enhancement, a sliding window is applied for normalized spectrogram size. Neighboring windows will overlap for 1/2 of the window size.

The generated windows may look like this:

![image](https://user-images.githubusercontent.com/79500078/186280899-3a84a315-b834-40da-9929-56f60bce7f59.png)

### Selective Search Region Proposal

For each window, selective search algorithm will be applied to generate region proposal.

For each proposal, and ground truth label, the overlapping area ration will be calculated. Regions with good performance will be choosen as positive proposals.

![image](https://user-images.githubusercontent.com/79500078/186281216-316ed49f-ed69-4cbc-9538-08dd20b3b43b.png)

## Bounding Box Classification with `bounding-box-classifier.ipynb`

After data set is generated, `bounding-box-classifier.ipynb` can use the data set to train a CNN model to classify the region proposal.

For each proposal in the training set, a label (positive/negative) is given. And BCELoss is apply as the loss function in this situation.

# Performance

With 56 positive labels and 56 negative labels, the model can achieve 90% accuracy in bounding box classification in testing set:

![image](https://user-images.githubusercontent.com/79500078/186281626-6f80643e-c795-4eff-8a32-12c60fb6f35d.png)

Equal number of positive labels and negative labels is suggested to reduce bias.
