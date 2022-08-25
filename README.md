# RCNN-Audio-Detection

![picture 6](/images/2022-08-24-23-17-56-cover.png)

## Dependencies

This project uses the older version of pytorch, installed using pip.

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

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

![picture 3](/images/2022-08-24-22-41-27-sliding-windows.png)

### Selective Search Region Proposal

For each window, selective search algorithm will be applied to generate region proposal.

For each proposal, and ground truth label, the overlapping area ration will be calculated. Regions with good performance will be choosen as positive proposals.

![picture 4](/images/2022-08-24-22-42-06-positive-negative-search.png)

## Bounding Box Classification with `bounding-box-classifier.ipynb`

After data set is generated, `bounding-box-classifier.ipynb` can use the data set to train a CNN model to classify the region proposal, and suggest offset.

For each proposal in the training set, a label (positive/negative) is given. And BCELoss is apply as the loss function in this situation.

Also, bounding box offset is also trained so the neural network can suggest the offset for each proposal.

### RCNN Structure

![picture 7](/images/2022-08-24-23-23-48-rcnn-structure.png)

The entire RCNN takes 2 inputs:

- The normalized spectrogram image in 1x64x64 format
- Meta data in 1x3 format:
  - Start frequency of the window
  - End frequency of the window
  - Time span of the window

The neural network computes the following outputs:

- If the window is positive, indicates the proposal is good or bad.
- The suggested offset for the window.

For example, following output means the neural network thinks the proposal is bad.

And it suggests to move the window anchor 2.76 pixels left, and 0.16 pixels down, expand the window width by 4.97 pixels, and expand the window height by 1 pixel for better performance.

```
(array([[0.36625865]], dtype=float32),
 array([[-2.7618032 , -0.16448733,  4.9751573 ,  1.0600426 ]],
       dtype=float32))
```

### Performance

With 110 seconds of audio, the data generator can generate about 1000 positive samples, and 3000 negative samples.

Learning 100 epochs will reach 80% accuracy on the test set.

![picture 8](/images/2022-08-24-23-28-59-machine-learning-performance.png)

Although the classification accuracy is only 80%, the final performance can be good since the neural network can suggest the offset for each proposal.

## Detect Audio

The `audio-detection.ipynb` notebook contains following steps:

- Load audio and RCNN model.
- Selective search proposals.
- Use RCNN model to filter the proposals.
- Apply the suggested offset to the proposals for final detection.

![picture 9](/images/2022-08-24-23-32-53-audio-detection-demo.png)
