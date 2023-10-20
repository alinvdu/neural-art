## Data Configuration
1. What kind of stimuli are used in order to associate brain state with certain images?
    Categories of stimuli:
        1. Audio (Music, Sound Effects).
        2. Visuals (We present to the participant an image that we will hope that will trigger a visual and emotional impact.)
        3. Smell (Smell of chocolate, lemon, rotten food).
        4. Tactile (Tingling, pain).
        5. Taste (Strawberry, bananas, medicine).
    All the above stimuli will get a certain kind of "ground truth" in terms of output image.
2. All the above labels are kept together with the data for reference processes but the actual data used is the EEG data that's elicited based on this stimuli.
3. The data is pre-processed using Neural Signal Processing algorithms and then outputed in table format that the model can read, together with the ground truth.
    x. Cleaning the data:
        x.1. ICA for cleaning data.
        x.2. Band-Pass Filtering (anything outside of 100 hz).
        x.3. Notch Filtering (eliminate power line interference 50hz to 60hz).
        x.4. Laplacian Space Filtering.
    a. ERPs.
    b. Time-Frequency.
        b.1. Baseline normalization.
        b.2. Downsampling.
        b.3. Power.
        b.4. Inter Trial Phase Clustering (angle consistency between trials).
        b.5. Cross-Trial Regression (correlation matrix between trials - simple regression matrix calculations).
        b.6. Instantaneous Frequency.
    e. Synchronicity.
        e.1. Connectivity Hubs.
        e.2. Granger Causality (between several components).

* Considerations:
    # Stimuli Association:
        * Make sure the stimuli are presented in a consistent and controlled manner (duration, intensity, method should be controlled for).
        * Ensure the association between images and stimuli is clearly defined (eg. what images represents the smell of chocolate? is not left to interpretation).
    # Data Collection:
        * Standardized time window (-2 to +2 seconds).
    # Data Preprocessing:
        * Visuals inspection of channels to remove excessive noise, interpolating bad channels, and after ICA manual inspection of components.
        * Consider baseline correction for ERP.
        * Specify time-window for extracting ERP post stimulus.
    # Time Frequency:
        * Specify time frequency decomposition method (eg. wavelet convolution).
        * Consider ranges of interest in terms of frequencies.

## Feedback Loops:
    * After the art is shown to participants collect data about the association with the image.