function BLER = helperAEWAutoencoderBLER(txNet,rxNet,simParams)
%helperAEWAutoencoderBLER Autoencoder block error rate (BLER) simulation
%   BLER = helperAEWAutoencoderBLER(TX,RX,P) simulates the BLER performance
%   of the autoencoder with trained encoder network, TX, and trained
%   decoder network, RX over an AWGN channel. P is the simulation
%   parameters structure and must have the following fields:
%
%   EbNoVec            - Eb/No values to simulate
%   MinNumErrors       - Minimum number of errors to simulate
%   NumSymbolsPerFrame - Number of symbols to simulated per frame
%   MaxNumFrames       - Maximum number of symbols to simulate
%   SignalPower        - Expected transmitted signal power
%
%   See also AutoencoderForWirelessCommunicationsExample, helperAEWEncode,
%   helperAEWDecode.

%   Copyright 2020-2022 The MathWorks, Inc.

M = txNet.Layers(1).InputSize;  % 입력 심볼의 수 (M)
k = log2(M);  % 입력 비트 수 계산 (k)
n = rxNet.Layers(1).InputSize;  % 채널 사용 수 (n)
EbNoVec = simParams.EbNoVec;  % 시뮬레이션할 Eb/No 값들

R = k/n;  % 코드율 계산 (R)
numSymbolsPerFrame = simParams.NumSymbolsPerFrame;  % 프레임당 심볼 수
BLER = zeros(size(EbNoVec));  % Eb/No 값들에 대한 BLER 결과를 저장할 배열

% 각 Eb/No 값에 대해 BLER 시뮬레이션을 수행
for EbNoIdx = 1:length(EbNoVec)
  % 현재 Eb/No 값에 코드율을 반영
  EbNo = EbNoVec(EbNoIdx) + 10*log10(R);
  
  % AWGN 채널 설정
  chan = comm.AWGNChannel("BitsPerSymbol",2, ...
    "EbNo", EbNo, "SamplesPerSymbol", 1, ...
    "SignalPower", simParams.SignalPower);

  numBlockErrors = 0;  % 블록 오류 수 초기화
  frameCnt = 0;  % 프레임 카운터 초기화
  
  % 최소 오류 수에 도달하거나 최대 프레임 수에 도달할 때까지 프레임 전송 반복
  while (numBlockErrors < simParams.MinNumErrors) ...
      && (frameCnt < simParams.MaxNumFrames)
      
    d = randi([0 M-1],numSymbolsPerFrame,1);  % 랜덤 심볼 생성

    xComplex = helperAEWEncode(d,txNet);  % 인코더를 통해 심볼을 인코딩
    
    yComplex = chan(xComplex);  % AWGN 채널을 통해 신호 전송
    
    dHat = helperAEWDecode(yComplex,rxNet);  % 디코더를 통해 수신 심볼 복조

    % 원래 심볼과 복조된 심볼을 비교하여 오류 수 집계
    numBlockErrors = numBlockErrors + sum(d ~= dHat);
    frameCnt = frameCnt + 1;
  end
  
  % 현재 Eb/No 값에 대한 BLER 계산
  BLER(EbNoIdx) = numBlockErrors / (frameCnt*numSymbolsPerFrame);
end
