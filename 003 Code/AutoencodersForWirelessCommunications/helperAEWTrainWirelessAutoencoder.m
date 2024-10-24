function [txNet,rxNet,info,trainedNet] = ...
  helperAEWTrainWirelessAutoencoder(n,k,normalization,EbNo,varargin)
%helperAEWTrainWirelessAutoencoder Train wireless autoencoder
%   [TX,RX,INFO,AE] = helperAEWTrainWirelessAutoencoder(N,K,NORM,EbNo)
%   trains an autoencoder, AE, with (N,K), where K is the number of input
%   bits and N is the number of channel uses. The autoencoder employs NORM
%   normalization. NORM must be one of 'Energy' and 'Average power'. The
%   channel is an AWGN channel with Eb/No set to EbNo. TX and Rx are the
%   encoder and decoder parts of the autoencoder that can be used in the
%   helperAEWEncoder and helperAEWDecoder functions, respectively. INFO is
%   the training information that can be used to check the convergence
%   behavior of the training process.
%
%   [TX,RX,INFO,AE] = helperAEWTrainWirelessAutoencoder(...,TP) provides
%   training parameters as follows:
%     TP.Plots: Plots to display during network training defined as one of
%               'none' (default) or 'training-progress'.
%     TP.Verbose: Indicator to display training progress information
%               defined as 1 (true) (default) or 0 (false).
%     TP.MaxEpochs: Maximum number of epochs defined as a positive integer.
%               The default is 15.
%     TP.InitialLearnRate: Initial learning rate as a floating point number
%               between 0 and 1. The default is 0.01;
%     TP.LearnRateSchedule: Learning rate schedule defined as one of
%               'piecewise' (default) or 'none'.
%     TP.LearnRateDropPeriod: Number of epochs for dropping the learning 
%               rate as a positive integer. The default is 10.
%     TP.LearnRateDropFactor: Factor for dropping the learning rate,
%               defined as a scalar between 0 and 1. The default is 0.1.
%     TP.MiniBatchSize: Size of the mini-batch to use for each training
%               iteration, defined as a positive integer. The default is 
%               20*M.
%
%   See also AutoencoderForWirelessCommunicationsExample, helperAEWEncode,
%   helperAEWDecode, helperAEWNormalizationLayer, helperAEWAWGNLayer.

%   Copyright 2020-2022 The MathWorks, Inc.

% Derived parameters
M = 2^k;
R = k/n;

if nargin > 4
  trainParams = varargin{1};
else


% 기본 훈련 옵션을 설정
% 검증 정확도(validation accuracy)가 80% 이상 95% 미만이 되도록 충분히 높은 최대 에포크 수를 설정
% 확률적 경사 하강법(SGD)은 수렴(convergence)을 달성하기 위해 충분한 심볼을 포함한 대표적인 미니배치가 필요
% 
% 따라서, 심볼 수(M)에 따라 미니배치 크기를 늘려야 합니다.
% 메모리가 부족하지 않도록 가능한 큰 미니배치 크기를 선택
% 초기 학습률(learning rate)을 설정할 때, 학습이 발산하지 않으면서도 검증 정확도가 합리적인 시간 내에 수렴할 수 있도록 충분히 높게 설정
% 미니배치 크기를 늘리면 초기 학습률도 함께 증가시켜야 할 수 있다는 점을 유의
% 학습률을 5 에포크마다 10배씩 줄입니다.
% 네트워크와 데이터 크기가 작기 때문에 훈련은 CPU에서 실행
% 데이터를 GPU로 전송하는 오버헤드가 GPU의 속도 이득보다 큼.


  trainParams.MaxEpochs = 10;
  trainParams.MiniBatchSize = 100*M;
  trainParams.InitialLearnRate = 0.08;
  trainParams.LearnRateSchedule = 'piecewise';
  trainParams.LearnRateDropPeriod = 5;
  trainParams.LearnRateDropFactor = 0.1;
  trainParams.ExecutionEnvironment = 'cpu'; %GPU 사용시 parallel toolbox 필요
  trainParams.Plots = 'none';
  trainParams.Verbose = false;
end

% 최대 에포크 수 (MaxEpochs)
% 의미: 훈련 데이터셋을 네트워크가 몇 번 반복 학습할지 결정. 한 에포크는 전체 데이터셋을 한 번 훈련하는 것
% 증가 효과: 더 많은 에포크는 모델이 데이터에 대해 더 많이 학습할 기회를 제공. 그러나 너무 많으면
% 오버피팅(overfitting) 위험.
% 감소 효과: 적은 에포크 수는 모델이 충분히 학습하지 못할 수 있어, 성능이 낮을 수 있음. 모델이 학습 부족으로 인해 훈련
% 데이터에 잘 맞지 않을 수 있음.

% 미니 배치 크기 (MiniBatchSize)
% 의미: 훈련 시 한 번에 처리되는 데이터 샘플의 수를 결정. 작은 미니 배치 크기는 메모리 사용량을 줄이고, 더 자주 파라미터를 업데이트.
% 증가 효과: 큰 미니 배치 크기는 학습이 더 안정적일 수 있으며, 병렬 처리에 유리. 하지만 메모리 사용량이 증가, 학습 속도가 느려질 수 있음.
% 감소 효과: 작은 미니 배치 크기는 메모리 사용량을 줄이고, 훈련 과정에서 더 많은 변동을 허용할 수 있음. 하지만 학습이 더
% 불안정할 수 있으며, 수렴 속도가 느려질 수 있음.

% 초기 학습률 (InitialLearnRate)
% 의미: 훈련 시작 시 파라미터 업데이트의 크기를 결정. 학습률은 경량화를 위해 가중치를 얼마나 빠르게 조정할지 정의.
% 증가 효과: 높은 학습률은 빠르게 수렴할 수 있지만, 너무 높으면 학습이 불안정해져 발산할 수 있음.
% 감소 효과: 낮은 학습률은 더 안정적으로 학습할 수 있지만, 수렴 속도가 느려질 수 있음.

% 학습률 스케줄링 (LearnRateSchedule)
% 의미: 학습률이 훈련 과정에서 어떻게 변화할지 정의. 'piecewise'는 학습률을 특정 에포크마다 감소시키는 것을 의미.
% 증가 효과: 학습률을 점진적으로 감소시키면, 모델이 수렴하는 과정에서 더 세밀한 조정이 가능함.
% 감소 효과: 일정한 학습률을 유지하면, 초기에는 빠르게 수렴하지만, 이후에는 학습이 불안정할 수 있음.

% 학습률 감소 주기 (LearnRateDropPeriod)
% 의미: 학습률을 감소시키는 주기를 정의. 
% 증가 효과: 주기를 늘리면 학습률 감소가 늦어지므로 학습이 더 오래 유지되며, 더 안정적인 학습.
% 감소 효과: 주기를 짧게 설정하면 학습률이 더 자주 감소하여, 모델이 조기 수렴할 수 있지만 최적화 실패가능성 증가.




EbNoChannel = EbNo + 10*log10(R); 

numTrainSymbols = 2500 * M;
numValidationSymbols = 100 * M;




% 오토인코더 네트워크를 정의
% 입력은 길이가 M인 원-핫(one-hot) 벡터
% 인코더(encoder)는 두 개의 완전 연결층(fully connected layer)으로 구성
% 첫 번째 완전 연결층은 M개의 입력과 M개의 출력을 가지며, 그 뒤에 ReLU 레이어가 위치
% 두 번째 완전 연결층은 M개의 입력과 n개의 출력을 가지며, 그 뒤에 정규화(normalization) 레이어가 위치
% available methods are energy and average power normalization.
% 정규화 레이어는 인코더 출력에 제약을 가하며, 사용할 수 있는 방법으로는 에너지 정규화(energy normalization)와 평균 전력 정규화(average power normalization)가 있음.
% 인코더 레이어들 뒤에는 AWGN(AWGN channel) 채널 레이어가 위치함.
% 두 개의 출력값이 복소수 심볼에 매핑되기 때문에, BitsPerSymbol 값을 2로 설정
% 정규화 레이어가 단위 전력(unity power)을 가진 신호를 출력하므로, 신호 전력을 1로 설정
% 채널의 출력은 디코더(decoder) 레이어들로 전달
% 두 번째 완전 연결층은 M개의 입력과 M개의 출력을 가지며, 그 뒤에 소프트맥스(softmax) 레이어가 위치
% 디코더의 출력은 0부터 M-1까지의 범위에서 가장 높은 확률을 가진 전송 심볼로 선택


wirelessAutoEncoder = [
  featureInputLayer(M,"Name","One-hot input","Normalization","none")
  
  fullyConnectedLayer(M,"Name","fc_1")
  reluLayer("Name","relu_1")
  
  fullyConnectedLayer(n,"Name","fc_2")
  
  helperAEWNormalizationLayer("Method", normalization)
  
  helperAEWAWGNLayer("NoiseMethod","EbNo", ...
    "EbNo",EbNoChannel, ...
    "BitsPerSymbol",2, ...
    "SignalPower",1)
  
  fullyConnectedLayer(M,"Name","fc_3")
  reluLayer("Name","relu_2")
  
  fullyConnectedLayer(M,"Name","fc_4")
  softmaxLayer("Name","softmax")
  
  classificationLayer("Name","classoutput")];

% Generate random training data. Create one-hot input vectors and labels. 
d = randi([0 M-1],numTrainSymbols,1);
trainSymbols = zeros(numTrainSymbols,M);
trainSymbols(sub2ind([numTrainSymbols, M], ...
  (1:numTrainSymbols)',d+1)) = 1;
trainLabels = categorical(d);

% Generate random validation data. Create one-hot input vectors and labels. 
d = randi([0 M-1],numValidationSymbols,1);
validationSymbols = zeros(numValidationSymbols,M);
validationSymbols(sub2ind([numValidationSymbols, M], ...
  (1:numValidationSymbols)',d+1)) = 1;
validationLabels = categorical(d);

% Set training options
options = trainingOptions('adam', ...
  'InitialLearnRate',trainParams.InitialLearnRate, ...
  'MaxEpochs',trainParams.MaxEpochs, ...
  'MiniBatchSize',trainParams.MiniBatchSize, ...
  'Shuffle','every-epoch', ...
  'ValidationData',{validationSymbols,validationLabels}, ...
  'LearnRateSchedule', trainParams.LearnRateSchedule, ...
  'LearnRateDropPeriod', trainParams.LearnRateDropPeriod, ...
  'LearnRateDropFactor', trainParams.LearnRateDropFactor, ...
  'ExecutionEnvironment', trainParams.ExecutionEnvironment, ...
  'Plots', trainParams.Plots, ...
  'Verbose', trainParams.Verbose);

maxIter = 10;
done = false;
iterCnt = 0;
while ~done
  % Train the autoencoder network
  [trainedNet,info] = trainNetwork(trainSymbols,trainLabels,wirelessAutoEncoder,options);

  iterCnt = iterCnt + 1;
  % Check if the network converged or maximum iteration reached
  if (info.FinalValidationAccuracy > 80) || (iterCnt >= maxIter)
    done = true;
  end
end

% Separate the network into encoder and decoder parts. Encoder starts with
% the input layer and ends after the normalization layer.
for idxNorm = 1:length(trainedNet.Layers)
  if isa(trainedNet.Layers(idxNorm), 'helperAEWNormalizationLayer')
    break
  end
end
lgraph = addLayers(layerGraph(trainedNet.Layers(1:idxNorm)), ...
  regressionLayer('Name', 'txout'));
lgraph = connectLayers(lgraph,'wnorm','txout');
txNet = assembleNetwork(lgraph);

% Decoder starts after the channel layer and ends with the classification
% layer. Add a feature input layer at the beginning. 
for idxChan = idxNorm:length(trainedNet.Layers)
  if isa(trainedNet.Layers(idxChan), 'helperAEWAWGNLayer')
    break
  end
end
firstLayerName = trainedNet.Layers(idxChan+1).Name;
n = trainedNet.Layers(idxChan+1).InputSize;
lgraph = addLayers(layerGraph(featureInputLayer(n,'Name','rxin')), ...
  trainedNet.Layers(idxChan+1:end));
lgraph = connectLayers(lgraph,'rxin',firstLayerName);
rxNet = assembleNetwork(lgraph);
