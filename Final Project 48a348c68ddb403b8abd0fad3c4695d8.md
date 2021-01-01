# Final Project

## Initial Stuff [Common]

- Librosa

    Library for music and sound analysis

- Librosa.Load

    return y (audio time series) , sr (sampling rate of y)

## Audio Stuff

- Mell Spectrogram

    A mel spectrogram is a spectrogram where the frequencies are converted to the mel scale.

    - Mel Scale

        Studies have shown that humans do not perceive frequencies on a linear scale. We are better at detecting differences in lower frequencies than higher frequencies

        In 1937, Stevens, Volkmann, and Newmann proposed a unit of pitch such that equal distances in pitch sounded equally distant to the listener. This is called the mel scale. We perform a mathematical operation on frequencies to convert them to the mel scale.

    - Formula

        ![Final%20Project%2048a348c68ddb403b8abd0fad3c4695d8/Untitled.png](Final%20Project%2048a348c68ddb403b8abd0fad3c4695d8/Untitled.png)

        Where f is the frequency .

        There is no specific Mel Formula

        ![Final%20Project%2048a348c68ddb403b8abd0fad3c4695d8/Untitled%201.png](Final%20Project%2048a348c68ddb403b8abd0fad3c4695d8/Untitled%201.png)

    ```python
    librosa.feature.melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, power=2.0, **kwargs)
    ```

    Where: `y`: audio time series

    `sr` : Sampling rate of y

    `S` : Spectrogram

    `n_fft` : Length of the FFT window

    `hop_length` : Number of samples between successive frame

    `power` : Exponent for the magnitude melspectrogram. e.g., 1 for energy, 2 for power, etc

- Chroma Feature
    - The chroma feature is a descriptor, which represents the tonal content of a musical audio signal in a condensed form. Therefore chroma features can be considered as important prerequisite for high-level semantic analysis, like chord recognition or harmonic similarity estimation. A better quality of the extracted chroma feature enables much better results in these high-level tasks. Short Time Fourier Transforms and Constant Q Transforms are used for chroma feature extraction.
    - chroma features are based on the twelve pitch spelling attributes C, C♯, D, …, B as used in Western music notation, where each chroma vector indicates how the energy in a signal's frame is distributed across the twelve chroma bands.
    - `{C, C♯, D, D♯, E , F, F♯, G, G♯, A, A♯, B}`

        ![Final%20Project%2048a348c68ddb403b8abd0fad3c4695d8/Untitled%202.png](Final%20Project%2048a348c68ddb403b8abd0fad3c4695d8/Untitled%202.png)

    - Reference

        [](https://www.researchgate.net/publication/330796993_Chroma_Feature_Extraction)

        [chroma](https://musicinformationretrieval.com/chroma.html)

    - Chroma STFT: Short Time Fourier Transforms

        This transform is a compromise between a time- and a frequency-based representation by determining the sinusoidal frequency and phase content of local sections of a signal as it changes over time.

         In this way, the STFT does not only tell which frequencies are “contained” in the signal but also atwhich points of times or, to be more precise, in which time intervals these frequencies appear.

        - The Fourier transform, which is used to convert a time-dependent signal to afrequency-dependent signal, is one of the most important mathematical toolsin audio signal processing
        - Applying the Fourier transform to local sections of an audio signal, one obtains the short-time Fourier transform (STFT)

        ```python
        librosa.feature.chroma_stft(y=None, sr=22050, S=None, norm=inf, n_fft=2048, hop_length=512, tuning=None, **kwargs)
        ```

        `y` : audio time sereis

        `sr` : sampling rate of y

        `S`: power spectrogram

        `norm`: Column-wise normalization.

        `n_fft`: FFT window size

        `tuning`: Deviation from A440 tuning in fractional bins (cents). If None, it is automatically estimated.

        `hop_length` : hop length if provided y, sr instead of S

        - Hop Length

            The number of samples between successive frames, e.g., the columns of a spectrogram. This is denoted as a positive integer hop_length .

        - Returns: Normalized energy for each chroma bin at each frame.
    - Chroma Cens: Chroma energy normalized statistic
        - The main idea of CENS features is that taking statistics over relatively large windows smooths out local deviations in tempo, articulation, and execution of note groups such as trills or arpeggios
        - To this end, we fix a number ℓ∈N that determines the length of a smoothing window (e.g., a Hann window) and then consider local averages (weighted by the window function) of each of the twelve components of the sequence (Q(x1),…,Q(xN)). This again results in a sequence of 12-dimensional vectors with nonnegative entries. In the last step, this sequence is downsampled by a factor of d, and the resulting vectors are normalized with respect to the Euclidean norm (ℓ2-norm). The two steps, quantization and smoothing, can be thought of computing weighted statistics of the energy distribution over a window of ℓ consecutive vectors. Therefore, we call the resulting features CENSℓd (chroma energy normalized statistics).

        ```python
        librosa.feature.chroma_cens(y=None, sr=22050, C=None, hop_length=512, fmin=None, tuning=None, n_chroma=12, n_octaves=7, bins_per_octave=None, cqt_mode=’full’, window=None, norm=2, win_len_smooth=41)
        ```

        `y`:audio time series

        `sr`:sampling rate of y

        `C`: a pre-computed constant-Q spectrogram

        `hop_length`:number of samples between successive chroma frames

        `fmin`: minimum frequency to analyze in the CQT. Default: ‘C1’ ~= 32.7 Hz.

        `norm`: Column-wise normalization of the chromagram.

        `tuning`: Column-wise normalization of the chromagram.

        - Returns

            The output cens-chromagram

    - Chroma Cq : Constant-Q Transform

        Unlike the Fourier transform, but similar to the mel scale, the constant-Q transform (Wikipedia) uses a logarithmically spaced frequency axis.

        ```python
        librosa.feature.chroma_cqt(y=None, sr=22050, C=None, hop_length=512, fmin=None, norm=inf, threshold=0.0, tuning=None, n_chroma=12, n_octaves=7, window=None, bins_per_octave=None, cqt_mode=’full’)
        ```

        `y`:audio time series

        `sr`:sampling rate of y

        `C`: a pre-computed constant-Q spectrogram

        `hop_length`:number of samples between successive chroma frames

        `fmin`: minimum frequency to analyze in the CQT. Default: ‘C1’ ~= 32.7 Hz.

        `norm`: Column-wise normalization of the chromagram.

        `tuning`: Column-wise normalization of the chromagram.

        - Return

            **chromagram** : np.ndarray [shape=(n_chroma, t)]The output chromagram

    - numpy.vstack

        Stack arrays in sequence vertically (row wise).

        This is equivalent to concatenation along the first axis after 1-D arrays
        of shape *(N,)* have been reshaped to *(1,N)*. Rebuilds arrays divided by
        `[vsplit](https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html#numpy.vsplit)`.

        This function makes most sense for arrays with up to 3 dimensions.

- Mfcc

    Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC.

    They are derived from a type of cepstral representation of the audio clip (a nonlinear "spectrum-of-a-spectrum"). 

    The difference between the cepstrum and the mel-frequency cepstrum is that in the MFC, the frequency bands are equally spaced on the mel scale, which approximates the human auditory system's response more closely than the linearly-spaced frequency bands used in the normal cepstrum. 

    This frequency warping can allow for better representation of sound, for example, in audio compression

---

### CNN Stuff

- Relu layer

    Remove non linearity in network

- Softmax Function

    So normally, what would happen, is the dog and the cat neurons would have any kind of real values.

    They don't have to add up to one. But then, we would apply the SoftMax function, which is written up over there at the top, and that would bring these values to be between zero and one

    and it would make them add up to one.

    ![Final%20Project%2048a348c68ddb403b8abd0fad3c4695d8/Untitled%203.png](Final%20Project%2048a348c68ddb403b8abd0fad3c4695d8/Untitled%203.png)

    The softmax function takes as input a vector z of K real numbers, and normalizes it into a probability distribution consisting of K probabilities proportional to the exponentials of the input numbers.

- same_pad: max pool with 2x2 kernel, stride 2 and SAME paddin
- "SAME" = with zero padding

---

- Sigmoid

    Smooth gradient, preventing “jumps” in output values.
    Output values bound between 0 and 1, normalizing the output of each neuron.
    Clear predictions—For X above 2 or below -2, tends to bring the Y value (the prediction) to the edge of the curve, very close to 1 or 0. This enables clear predictions.

- TanH

    Zero centered—making it easier to model inputs that have strongly negative, neutral, and strongly positive values.
    Otherwise like the Sigmoid function.

- Softmax

    Able to handle multiple classes only one class in other activation functions—normalizes the outputs for each class between 0 and 1, and divides by their sum, giving the probability of the input value being in a specific class.
    Useful for output neurons—typically Softmax is used only for the output layer, for neural networks that need to classify inputs into multiple categories.

- epochs

    One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE. Since one epoch is too big to feed to the computer at once we divide it in several smaller batches.

Accuracy: 73%

- noise removal steps
    1. An FFT is calculated over the noise audio clip
    2. Statistics are calculated over FFT of the the noise (in frequency)
    3. A threshold is calculated based upon the statistics of the noise (and the desired sensitivity of the algorithm)
    4. An FFT is calculated over the signal
    5. A mask is determined by comparing the signal FFT to the threshold
    6. The mask is smoothed with a filter over frequency and time
    7. The mask is appled to the FFT of the signal, and is inverted