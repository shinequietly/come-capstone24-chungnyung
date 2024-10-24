% QPSK Constellation after Passing through AWGN
% This script shows the traditional QPSK constellation after passing through an AWGN channel.

% Parameters
M = 4; % Modulation order (QPSK)
numSymbols = 1e4; % Number of symbols for plotting
SNRdB = 10; % SNR for AWGN

% Generate random data
dataIn = randi([0 M-1], numSymbols, 1);

% QPSK Modulation
txSignal = pskmod(dataIn, M, pi/4); % QPSK with phase offset of pi/4 for traditional QPSK

% Add AWGN noise
rxSignalAWGN = awgn(txSignal, SNRdB, 'measured');

% Plot constellation diagram after AWGN
figure;
scatterplot(rxSignalAWGN, 1, 0, 'y.'); % Set color to yellow
title('QPSK');
grid on;
xlim([-2 2]); % Set x-axis limits
ylim([-2 2]); % Set y-axis limits

% Find the scatter plot line object and set MarkerSize
hMarkers = findall(gcf, 'Type', 'Line');
set(hMarkers, 'MarkerSize', 20); % Increase marker size
