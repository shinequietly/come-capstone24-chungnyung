% 8PSK Constellation after Equalization
% This script shows the constellation of 8PSK after applying equalization.

% Parameters
M = 8; % Modulation order (8PSK)
numSymbols = 1e4; % Reduced number of symbols for plotting
SNRdB = 30; % Higher SNR value to reduce noise effects

% Generate random data
dataIn = randi([0 M-1], numSymbols, 1);

% 8PSK Modulation
txSignal = pskmod(dataIn, M);

% Apply Rayleigh Fading
h = sqrt(0.5) * (randn(size(txSignal)) + 1j * randn(size(txSignal)));
rxSignalRayleigh = h .* txSignal; % Apply the fading

% Add AWGN noise with higher SNR
rxSignalRayleighAWGN = awgn(rxSignalRayleigh, SNRdB, 'measured');

% Apply Equalizer (compensate for the channel)
equalizedSignal = rxSignalRayleighAWGN ./ h;

% Plot constellation diagram after equalization
figure;
scatterplot(equalizedSignal, 1, 0, 'y.'); % Set color to yellow
title('8PSK Constellation after Equalization');
grid on;
xlim([-2.5 2.5]); % Set x-axis limits
ylim([-2.5 2.5]); % Set y-axis limits

% Find the scatter plot line object and set MarkerSize
hMarkers = findall(gcf, 'Type', 'Line');
set(hMarkers, 'MarkerSize', 20); % Increase marker size
