function dHat = helperAEWDecode(y,rxNet)
%helperAEWDecode Autoencoder decoder network
%   D = helperAEWDecode(Y,RX) decodes the received complex symbols, Y,
%   using the decoder network, RX, and returns received symbol estimates,
%   D. D is an integer between 0 and M-1, where M = 2^k and k is the
%   number of input bits per block for the encoder.
%
%   The inference is run on CPU since the network and the data are too
%   small. The overhead of transferring this data to the GPU is more than
%   the speed gain of the GPU.
%
%   See also AutoencoderForWirelessCommunicationsExample, 
%   helperAEWEncode, helperAEWTrainWirelessAutoencoder.

%   Copyright 2020-2022 The MathWorks, Inc.

n = rxNet.Layers(1).InputSize;
yNet = [real(y) imag(y)];
yNet = reshape(yNet',n,[])';

xHatCat = classify(rxNet,yNet, ...
  MiniBatchSize=length(yNet), ...   % Process whole frame at once
  ExecutionEnvironment="cpu");      % Run on CPU
dHat = double(xHatCat) - 1;
end