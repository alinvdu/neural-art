% MATLAB script that imports the pre-processed EEG data, applies time 
% shifting with statistical and interpolative padding, adds a bit of noise
% and then performs time-frequency analysis on both the original and
% augmented data.
% Define the file path for pre-processed EEG data

% Define the trial and directories
trial = 'trial7';
inputDir = [trial '/cleaned/'];  % Directory where the original .csv files are stored
outputDir = [trial '/augmented/'];  % Directory where the augmented .csv files will be saved

% Check if output directory exists, if not, create it
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% List all .csv files in the input directory
files = dir([inputDir '*.csv']);

% Parameters for augmentation
shiftAmount = 0.8;  % 800 ms
samplingRate = 128;  % EEG data sampling rate in Hz
shiftSamples = round(shiftAmount * samplingRate);
windowSize = 20;     % Window size for smoothing the transition
noiseLevel = 20.5;   % Adjust this based on the scale of your EEG data

% Loop through each file
for idx = 1:length(files)
    fileName = files(idx).name;
    fileToProcess = [inputDir fileName];
    exportedDataFile = [outputDir fileName];

    % Import pre-processed EEG data
    preprocessedData = readmatrix(fileToProcess)';
    % Parameters for time shifting and padding
    shiftAmount = 0.8; % 800 ms
    samplingRate = 128; % EEG data sampling rate in Hz
    shiftSamples = round(shiftAmount * samplingRate);
    
    % Calculate statistical properties
    freqBands = [0.5 4; 4 8; 8 12; 12 30]; % Delta, Theta, Alpha, Beta bands
    amplitudeFactors = [0.5, 0.3, 0.2, 0.1]; % Relative amplitude of each band
    
    % Calculate statistical properties
    signalMean = mean(preprocessedData, 2);
    signalStd = std(preprocessedData, 0, 2);
    
    % Generate synthetic EEG-like signal
    syntheticSignalLeft = zeros(size(preprocessedData, 1), shiftSamples);
    syntheticSignalRight = zeros(size(preprocessedData, 1), shiftSamples);
    timeAxis = (1:shiftSamples) / samplingRate;
    
    for bandIdx = 1:size(freqBands, 1)
        freqs = freqBands(bandIdx, 1):freqBands(bandIdx, 2);
        for freq = freqs
            % Generate a sine wave for each frequency
            sineWave = sin(2 * pi * freq * timeAxis);
    
            % For each channel, randomly choose a phase shift and create a wave with that phase
            for ch = 1:size(preprocessedData, 1)
                phaseShift = randi(length(timeAxis));  % Random phase shift for the sine wave
                sineWaveShifted = circshift(sineWave, [0, phaseShift]);
    
                % Add the sine wave to the synthetic signal
                syntheticSignalLeft(ch, :) = syntheticSignalLeft(ch, :) + ...
                    amplitudeFactors(bandIdx) * signalStd(ch) * sineWaveShifted;
                syntheticSignalRight(ch, :) = syntheticSignalRight(ch, :) + ...
                    amplitudeFactors(bandIdx) * signalStd(ch) * sineWaveShifted;
            end
        end
    end
    
    % Add the mean of the original signal to the synthetic signal
    syntheticSignalLeft = bsxfun(@plus, syntheticSignalLeft, signalMean);
    syntheticSignalRight = bsxfun(@plus, syntheticSignalRight, signalMean);
    
    % Apply a window function for smooth transition
    windowSize = 20; % Number of samples over which to apply the window function
    windowFunc = hann(windowSize*2)';
    
    % Apply the window function to the left side
    leftTransition = syntheticSignalLeft(:, end-windowSize+1:end) .* windowFunc(1:windowSize) + ...
                     preprocessedData(:, 1:windowSize) .* windowFunc(windowSize+1:end);
    
    % Apply the window function to the right side
    rightTransition = syntheticSignalRight(:, 1:windowSize) .* windowFunc(windowSize+1:end) + ...
                      preprocessedData(:, end-windowSize+1:end) .* windowFunc(1:windowSize);
    
    % Calculate the standard deviation of the original data
    stdDev = std(preprocessedData, 0, 2);
    
    % % Adjust the noise level to be a fraction of the standard deviation
    % noiseFraction = 0.1;  % for example, 5% of the standard deviation
    % randomNoise = bsxfun(@times, noiseFraction * stdDev, randn(size(preprocessedData)));
    % 
    % % Add the noise to the original data
    % augmentedNoisedData = preprocessedData + randomNoise;
    % Generate pink noise - inversely proportional to the frequency
    
    
    % Number of samples and sampling rate
    N = size(preprocessedData, 2);
    fs = samplingRate;
    
    % Generate pink noise
    % The fft of pink noise is proportional to 1/f, so generate noise in frequency domain
    f = linspace(0, fs/2, floor(N/2)+1);  % Frequency vector from 0 to Nyquist frequency
    f(1) = 1;  % Avoid division by zero at the DC component
    pinkPSD = 1 ./ f;  % Power spectrum density of pink noise (1/f)
    
    % Create a two-sided symmetric PSD
    if mod(N,2)
        pinkPSD = [pinkPSD, fliplr(pinkPSD(2:end-1))];
    else
        pinkPSD = [pinkPSD, fliplr(pinkPSD(2:end))];
    end
    
    % Random phase for the pink noise
    randomPhase = exp(1i * 2 * pi * rand(size(pinkPSD)));
    
    % Construct the pink noise in frequency domain
    pinkNoiseFreqDomain = sqrt(pinkPSD) .* randomPhase;
    
    % Transform back to time domain
    pinkNoise = ifft(pinkNoiseFreqDomain, 'symmetric');
    
    % Scale the pink noise to match the desired noise level
    noiseLevel = 20.5; % Adjust this based on the scale of your EEG data
    pinkNoise = pinkNoise * std(preprocessedData(:)) * noiseLevel;
    
    % Repeat the pink noise for each channel
    pinkNoise = repmat(pinkNoise, size(preprocessedData, 1), 1);
    
    % Add the pink noise to the EEG data
    augmentedNoisedData = preprocessedData + pinkNoise(:, 1:end-1);
    
    % Combine the padded and original signals with transitions
    augmentedData = [syntheticSignalLeft(:, 1:end-windowSize), leftTransition, ...
                     augmentedNoisedData, rightTransition, ...
                     syntheticSignalRight(:, windowSize+1:end)];
    
    originalLength = 1024;  % Original length of the EEG data
    augmentedLength = size(augmentedData, 2);  % Length of the augmented data
    
    % Resample the augmented data to match the original length
    resampledData = resample(augmentedData', originalLength, augmentedLength)';
    % Export augmented data
    writematrix(resampledData', exportedDataFile);
end