% Parameters
k = 4;    % number of input bits
M = 2^k;  % number of possible input symbols
n = 7;    % number of channel uses
EbNo = 3; % Eb/No in dB

% Convert Eb/No to channel Eb/No values using the code rate
R = k/n;
EbNoChannel = EbNo + 10*log10(R);

% Define custom channel function
function y = customChannel(x, EbNo)
    % Example custom channel: Combination of Rayleigh fading and AWGN
    % Define Rayleigh fading parameters
    rayleighChan = comm.RayleighChannel( ...
        'SampleRate', 1, ...
        'PathDelays', [0 1.5e-5], ...
        'AveragePathGains', [0 -3], ...
        'NormalizePathGains', true);

    % Pass signal through Rayleigh channel
    y = rayleighChan(x);

    % Add AWGN noise
    y = awgn(y, EbNo, 'measured');
end

% Create Autoencoder Layers
wirelessAutoencoder = [
  featureInputLayer(M,"Name","One-hot input","Normalization","none")
  
  fullyConnectedLayer(M,"Name","fc_1")
  reluLayer("Name","relu_1")
  
  fullyConnectedLayer(n,"Name","fc_2")
  
  helperAEWNormalizationLayer("Method", "Energy", "Name", "wnorm")
  
  fullyConnectedLayer(M,"Name","fc_3")
  reluLayer("Name","relu_2")
  
  fullyConnectedLayer(M,"Name","fc_4")
  softmaxLayer("Name","softmax")
  
  classificationLayer("Name","classoutput")];

% Custom encoding function
function encodedSymbols = customEncode(inputBits, customSymbols)
    encodedSymbols = customSymbols(inputBits + 1);
end

% Custom decoding function
function decodedBits = customDecode(receivedSymbols, customSymbols)
    [~, decodedBits] = min(abs(receivedSymbols - customSymbols.'), [], 2);
    decodedBits = decodedBits - 1;
end

% Train Autoencoder
n = 2;                      % number of channel uses
k = 2;                      % number of input bits
EbNo = 3;                   % dB
normalization = "Energy";   % Normalization "Energy" | "Average power"

[txNet, rxNet, infoTemp, wirelessAutoEncoder] = helperAEWTrainWirelessAutoencoder(n, k, normalization, EbNo);
infoTemp.n = n;
infoTemp.k = k;
infoTemp.EbNo = EbNo;
infoTemp.Normalization = normalization;
info = infoTemp;

% Plot Training Performance
figure
helperAEWPlotTrainingPerformance(info(1))

% Plot Autoencoder
figure
tiledlayout(2, 2)
nexttile([2 1])
plot(wirelessAutoEncoder(1))
title('Autoencoder')
nexttile
plot(txNet(1))
title('Encoder/Tx')
nexttile
plot(rxNet(1))
title('Decoder/Rx')

% Simulation Parameters
simParams.EbNoVec = 0:0.5:8;
simParams.MinNumErrors = 10;
simParams.MaxNumFrames = 300;
simParams.NumSymbolsPerFrame = 10000;
simParams.SignalPower = 1;
EbNoChannelVec = simParams.EbNoVec + 10*log10(R);

% Initialize Constellation Diagrams
txConst = comm.ConstellationDiagram('ShowReferenceConstellation', false, 'ShowLegend', true, 'ChannelNames', {'Tx Constellation'});
rxConst = comm.ConstellationDiagram('ShowReferenceConstellation', false, 'ShowLegend', true, 'ChannelNames', {'Rx Constellation'});

% Block Error Rate (BLER) Calculation
BLER = zeros(size(EbNoChannelVec));
for trainingEbNoIdx = 1:length(EbNoChannelVec)
    EbNo = EbNoChannelVec(trainingEbNoIdx);

    numBlockErrors = 0;
    frameCnt = 0;
    while (numBlockErrors < simParams.MinNumErrors) && (frameCnt < simParams.MaxNumFrames)
        d = randi([0 M-1], simParams.NumSymbolsPerFrame, 1);    % Random information bits
        x = customEncode(d, customQAMSymbols);                 % Custom Encoder
        txConst(x)
        y = customChannel(x, EbNo);                            % Custom Channel
        rxConst(y)
        dHat = customDecode(y, customQAMSymbols);              % Custom Decoder

        numBlockErrors = numBlockErrors + sum(d ~= dHat);
        frameCnt = frameCnt + 1;
    end
    BLER(trainingEbNoIdx) = numBlockErrors / (frameCnt * simParams.NumSymbolsPerFrame);
end

% Plot BLER vs EbNo
figure;
semilogy(simParams.EbNoVec, BLER, '-o');
xlabel('Eb/No (dB)');
ylabel('BLER');
title('BLER vs. Eb/No');
grid on;
