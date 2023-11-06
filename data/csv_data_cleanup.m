% How to use this file:
% The purpose of this file is to cleanup EEG data. It loads a file
% containing recordings for a specific stimuli for Epoch X. It also loads a
% baseline from before stimuli. It does:
%     - High power filter.
%     - Baseline normalization.
%     - ICA and presents figures for each component.
%     - Commented code to remove specific components based on manual
%     inspection for each component.
% When running make sure the current folder is inside an experiment.
% The output will be a cleaned eeg file with .csv format.

% Define the file path (you should replace this with your actual file path)

phase = 'phase_Stimuli_Structural_DNA_recording';
trial = 'trial4/';
filename = [trial phase '.csv'];

% Set import options for metadata
opts_meta = delimitedTextImportOptions('NumVariables', 4);  
opts_meta.VariableNamesLine = 1;  % Line where variable names are defined
opts_meta.DataLines = [2 2];  % Line range of the actual metadata values
opts_meta.VariableNamingRule = 'preserve';  % Preserve original headers
opts_meta = setvaropts(opts_meta, [1, 2, 3, 4], 'WhitespaceRule', 'preserve');
opts_meta = setvaropts(opts_meta, [1, 2, 3, 4], 'EmptyFieldRule', 'auto');

% Read the metadata
metadata = readtable(filename, opts_meta);

% Convert the metadata table to a cell array if needed
metadata = table2cell(metadata);

%% This part is for Emotiv EEG files if ou want to refine your data please load the cleaned state.
% Set import options for EEG data
opts_eeg = delimitedTextImportOptions('NumVariables', 15);
opts_eeg.VariableNamesLine = 3;  % Line where variable names are defined
opts_eeg.DataLines = [4 1027];  % Lines of the actual EEG data
opts_eeg.VariableNamingRule = 'preserve';  % Preserve original headers
opts_eeg = setvaropts(opts_eeg, 1:15, 'WhitespaceRule', 'preserve');
opts_eeg = setvaropts(opts_eeg, 1:15, 'EmptyFieldRule', 'auto');

% Read the EEG data
eegData = readtable(filename, opts_eeg);
eegChannels = eegData.Properties.VariableNames(2:end);

% Convert the EEG data table to an array
eegData = table2array(eegData);

time = cellfun(@str2double, eegData(:, 1));

eegData = eegData(:, 2:end);

% Ensure eegData is a numeric array before performing fft
eegData = cellfun(@str2double, eegData);

%% Load cleaned EEG Data - run this only if you want to adjust the clean data with ICA
% cleanedFilename = [trial '/cleaned/' phase '_cleaned.csv'];
% eegData = readmatrix(cleanedFilename);

%% Finish EEG Data
% Optionally, separate timestamps and EEG channels
t_min = min(time);
t_max = max(time);
a = 0; % desired minimum value
b = 8; % desired maximum value

t_normalized = (a + (time - t_min) * (b - a) / (t_max - t_min))';

%import baseline
baseline = 'trial1/baseline_eyesopen.csv';
%baseline = 'trial1/baseline_eyesclosed.csv';

% Set import options for EEG data
opts_baseline = delimitedTextImportOptions('NumVariables', 15);
opts_baseline.DataLines = [1 1024];  % Lines of the actual EEG data
opts_baseline.VariableNamingRule = 'preserve';  % Preserve original headers
opts_baseline = setvaropts(opts_baseline, 1:15, 'WhitespaceRule', 'preserve');
opts_baseline = setvaropts(opts_baseline, 1:15, 'EmptyFieldRule', 'auto');

% Read the EEG data
baselineDataTable = table2cell(readtable(baseline, opts_baseline));
baselineData = cellfun(@str2double, baselineDataTable(:, 2:end));

srate = 128;
npnts = length(time);

numChan = size(eegChannels);
numChan = numChan(2);

% create a time frequency map for visualization
min_freq = 0;
max_freq = 80;
num_freq = 100;
frex = linspace(min_freq, max_freq, num_freq);

% wavelet
fwhm = linspace(.1, .1, num_freq);
wave_time =-2:1/srate:2;
half_wave = (length(wave_time)-1)/2;

% FFT params
nWave = length(wave_time);
nData = npnts;
nConv = nWave + nData - 1;

% high pass filtering first
fc = 0.5;
n = 4;

% Design the high-pass filter
[b, a] = butter(n, fc/(srate/2), 'high');

% Apply the filter - and normalization - not needed for clean data.
filteredData = filtfilt(b, a, eegData);
filteredBaselineData = filtfilt(b, a, baselineData);
baselinedData = bsxfun(@minus, filteredData, filteredBaselineData);
%baselinedData = eegData;

%% You can re-run this part after removing component
tf = zeros(numChan, num_freq, npnts);
tf_baselined = zeros(numChan, num_freq, npnts);
range_cycles = [ 4 10 ];
s = logspace(log10(range_cycles(1)), log10(range_cycles(end)), num_freq) ./ (2*pi*frex);

for fig = 1:ceil(numChan/4)
    figure(fig), clf; % Clear the figure once before creating subplots
end

% convolution over frequencies for each channel
for ch=1:numChan
    % FFT
    %dataX = fft(eegData(:, ch)', nConv); % use this for cleaned data
    dataXBaselined = fft(baselinedData(:, ch)', nConv);
    for fi=1:length(frex)
        % create a wavelet and get its FFT
        %wavelet = exp(2*1i*pi*frex(fi).*wave_time) .* exp(-4*log(2)*wave_time.^2 / fwhm(fi).^2);
        wavelet = exp(2*1i*pi*frex(fi).*wave_time) .* exp(-wave_time.^2./(2*s(fi)^2));
        waveletX = fft(wavelet, nConv);
    
        % scale
        waveletX = waveletX ./ max(waveletX);
    
%         % convolution
%         as = ifft(waveletX .* dataX);
%     
%         % cut wings
%         as = as(half_wave+1:end-half_wave);
%     
%         % compute power and average over trials
%         tf(ch, fi, :) = 2 * abs(as).^2;
    
        % repeat everything for baseline
        as_baseline = ifft(waveletX .* dataXBaselined);
    
        % cut wings
        as_baseline = as_baseline(half_wave+1:end-half_wave);
    
        % compute power and average over trials
        tf_baselined(ch, fi, :) = 2 * abs(as_baseline).^2;
    end

    % Determine which figure to plot on
    figNum = ceil(ch / 4);
    figure(figNum);

    % Determine the position of the subplot
    subPlotNum = mod(ch-1, 4) + 1;
    subplot(2, 2, subPlotNum);

    contourf(t_normalized, frex, squeeze(tf_baselined(ch, :, :)), 40, "linecolor", "none");
    colormap jet
    
    % figure meta data
    title("Time & frequency plot baseline norm " + eegChannels{ch});
    xlabel("Time (ms)");
    ylabel("Frequency (Hz)");
end

number_of_components = 14;

% ICA
[icasig, mixingMatrix, separatingMatrix] = fastica(baselinedData', 'numOfIC', number_of_components);

figure(5), clf;
% Plot time courses

foundComp = size(icasig);
foundComp = foundComp(1);

for i = 1:foundComp
    subplot(foundComp, 1, i);
    plot(icasig(i, :));
    title(['Component ' num2str(i)]);
end

% Number of components
nComponents = size(mixingMatrix, 2);

% Set the number of rows and columns for subplot
nRows = ceil(sqrt(nComponents));
nCols = ceil(nComponents / nRows);

% Create a figure
figure(6), clf;

chanlocsX = [-0.3, -0.6, -0.5, -0.5, -0.9, -0.8, -0.6, 0.6, 0.8, 0.9, 0.5, 0.5, 0.6, 0.3];
chanlocsY = [0.6, 0.5, 0.8, 0.3, 0, -0.3, -0.6, -0.6, -0.3, 0, 0.3, 0.8, 0.5, 0.6];

% Loop over the components
for i = 1:nComponents
    subplot(nRows, nCols, i);
    
    % Get the map for the current component from the mixingMatrix
    map = mixingMatrix(:, i);
    
    % Interpolate the map on a fine grid
    [gridX, gridY] = meshgrid(linspace(-1.1, 1.1, 100), linspace(-1.1, 1.1, 100));
    interpMap = griddata(chanlocsX, chanlocsY, map, gridX, gridY, 'v4');

    % Plot the interpolated map
    contourf(gridX, gridY, interpMap, 40, 'LineColor', 'none');
    hold on;

    % Parameters for the head outline
    headRadius = 1.0; % Radius of the head circle
    noseLength = 0.1; % Length of the nose
    earWidth = 0.1; % Width of the ears
    earHeight = 0.2; % Height of the ears
    
    % Draw the head circle
    theta = 0:0.01:2*pi;
    headX = headRadius * cos(theta);
    headY = headRadius * sin(theta);
    plot(headX, headY, 'k', 'LineWidth', 2);
    
    % Draw the nose
    noseX = [0, noseLength, 0];
    noseY = [headRadius, headRadius + noseLength, headRadius];
    plot(noseX, noseY, 'k', 'LineWidth', 2);
    
    % Draw the ears
    leftEarX = [-headRadius, -headRadius - earWidth, -headRadius];
    leftEarY = [earHeight, 0, -earHeight];
    plot(leftEarX, leftEarY, 'k', 'LineWidth', 2);
    
    rightEarX = [headRadius, headRadius + earWidth, headRadius];
    rightEarY = [earHeight, 0, -earHeight];
    plot(rightEarX, rightEarY, 'k', 'LineWidth', 2);
    
    % Plot the channel locations
    plot(chanlocsX, chanlocsY, 'k.', 'MarkerSize', 8);
    
    % Add title and remove axes for clarity
    title(['Component ' num2str(i)]);
    axis equal off;
end

% Add a colorbar
colormap(jet);

% remove component
componentToRemove = 4;
artifactSignal = mixingMatrix(:, componentToRemove) * icasig(componentToRemove, :);
baselinedData = baselinedData - artifactSignal';

exportfilename = [trial '/cleaned/' phase '_cleaned.csv'];

% Export the data to a CSV file
writematrix(baselinedData, exportfilename);