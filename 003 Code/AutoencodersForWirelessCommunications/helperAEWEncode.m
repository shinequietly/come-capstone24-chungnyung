function xComplex = helperAEWEncode(d,txNet)
%helperAEWEncode Autoencoder encoder network
%   X = helperAEWEncode(D,TX) encodes the data symbols, D, using the
%   encoder network, TX, and returns complex symbols, X. D must be and
%   integer between 0 and M-1, where M = 2^k and k is the number of input
%   bits per block for the encoder.
%
%   See also AutoencoderForWirelessCommunicationsExample, 
%   helperAEWDecode, helperAEWTrainWirelessAutoencoder.

%   Copyright 2020-2022 The MathWorks, Inc.

M = txNet.Layers(1).InputSize;
numSymbolsPerFrame = length(d);
inputSymbols = zeros(numSymbolsPerFrame,M);
inputSymbols(sub2ind([numSymbolsPerFrame, M], ...
  (1:numSymbolsPerFrame)',d+1)) = 1;

x = predict(txNet, inputSymbols, ...
  MiniBatchSize=numSymbolsPerFrame,ExecutionEnvironment="cpu");
xComplex = reshape(x',2,[])';
xComplex = xComplex(:,1) + 1i* xComplex(:,2);
end