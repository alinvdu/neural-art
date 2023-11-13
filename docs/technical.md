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
        x.1. High pass filter.
        x.2. Baseline normalization (with or without eyes open) - should I take baseline before each visualization?
        x.3. ICA for cleaning data.
        x.4. Manual Inspect of Time-Frequency plot for all channels for individual removal of ICA components.
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
    # Feedback Loops:
        * After the art is shown to participants collect data about the association with the image.

## EEG Data
    * Calibration phase:
        - 10000ms look at cursor.
        - 10000ms keep eyes closed.
    * Stimuli Phase:
        - 8000ms preparation -> get accustomed with the image.
        - 5000ms preparation -> letting user know that recording is starting.
        - 8000ms recording - eyes open.
        - 3000ms preparation -> letting user know that recording ended.
        - 4000ms preparation -> letting user know to imagine with eyes closed.
        - 8000ms recording - eyes closed - imagination.
    * First experiment with the first 10 images can be found here: https://player.emotivpro.com/remote/MzgjXc2NjA=Yqb
    * Emotiv will output 2 kinds of files to be used:
        1. intervalMarker.
            Structure: latency, duration, type, marker_value, key, timestamp, marker_id, timestamp.
            Columns we are interested in: duration, type and marker_value.
            Values we are interested in for marker: recording, recording_eyes_closed.
            Values we are interested in for type: phase_Meditating_cube, phase_Image_Cube, phase_Sitting_on_the_chair_in_nature, phase_The_valley_of_white, phase_The_cute_rat, phase_Abstract_forms_of_beauty, phase_The_Japanese_creature, phase_The_babel_fish, phase_Simulated_garden, phase_Parallel_Universes_Road.
        2. md.csv document with the containing neural serial data.
## Processing the EEG data
    # Stage 1: Processing the invervalMarker andd md.csv data in Python.
        Process these 2 csv files and output 1 csv files / each recording with the appropriate fields.
