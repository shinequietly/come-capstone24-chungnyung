classdef helperAEWAWGNLayer < nnet.layer.Layer  
  properties
    NoiseMethod {mustBeMember(NoiseMethod, {'EbNo', 'EsNo', 'SNR'})} = 'EsNo'
    EbNo = 10
    EsNo = 10
    SNR = 10
    BitsPerSymbol = 1
    SignalPower = 1
  end
  
  properties (SetAccess = private, GetAccess = private)
    LocalSNR
  end
  
  methods
    function layer = helperAEWAWGNLayer(varargin)
      p = inputParser;
      addParameter(p, 'NoiseMethod', 'EsNo')
      addParameter(p, 'EbNo', 10)
      addParameter(p, 'EsNo', 10)
      addParameter(p, 'SNR', 10)
      addParameter(p, 'BitsPerSymbol', 1)
      addParameter(p, 'SignalPower', 1)
      addParameter(p, 'Name', 'awgn')
      addParameter(p, 'Description', '')
      
      parse(p, varargin{:})
      layer.NoiseMethod = p.Results.NoiseMethod;
      layer.EbNo = p.Results.EbNo;
      layer.EsNo = p.Results.EsNo;
      layer.SNR = p.Results.SNR;
      layer.BitsPerSymbol = p.Results.BitsPerSymbol;
      layer.SignalPower = p.Results.SignalPower;
      layer.Name = p.Results.Name;

      if isempty(p.Results.Description)
        switch p.Results.NoiseMethod
          case 'EbNo'
            value = layer.EbNo;
          case 'EsNo'
            value = layer.EsNo;
          case 'SNR'
            value = layer.SNR;
        end
        layer.Description = "AWGN channel with " + p.Results.NoiseMethod ...
          + " = " + num2str(value);
      else
        layer.Description = p.Results.Description;
      end
      layer.Type = 'AWGN Channel';

      samplesPerSymbol = 1;
      if strcmp(layer.NoiseMethod, 'EbNo')
        EsNo = layer.EbNo + 10 * log10(layer.BitsPerSymbol);
        layer.LocalSNR = EsNo - 10 * log10(samplesPerSymbol);
      elseif strcmp(layer.NoiseMethod, 'EsNo')
        EsNo = layer.EsNo;
        layer.LocalSNR = EsNo - 10 * log10(samplesPerSymbol);
      else
        layer.LocalSNR = layer.SNR;
      end
    end
    
    %% 여기를 수정해야함. 복소수를 쓰면안되는 모양인데 기본예제에서도 실수부 허수부를 분리하여 계산.

    
    
    

function z = predict(layer, x)
    % 입력 신호의 총 원소 개수를 계산
    numElem = numel(x);

    % 입력 신호가 홀수 개의 원소를 가진 경우, 제로 패딩을 추가
    if mod(numElem, 2)
        append = true;
        x = [x; zeros(1, size(x, 2))];
    else
        append = false;
    end

    % 입력 신호를 복소수 벡터로 가정
    % x가 2D 배열일 경우, 실수부와 허수부를 분리 <- 이부분이 필요한가?
    if size(x, 1) == 2
        realPart = x(1, :);
        imagPart = x(2, :);
        x = realPart + 1j * imagPart;
    else
        % x가 1D 배열인 경우, 그대로 사용
        x = x(:).'; % 변환하여 열 벡터로 변경
    end

    % Rayleigh fading 적용
    Fading = sqrt(0.5) * (randn(size(x)) + 1j * randn(size(x))); % 채널 페이딩
    Faded_signal = Fading .* x; % 페이딩된 신호

    % AWGN 추가
    noisePower = 10^(-layer.LocalSNR / 10); % 잡음 전력
    noiseStdDev = sqrt(noisePower / 2); % 잡음의 표준편차 (실수부와 허수부)
    AWGN_signal = noiseStdDev * (randn(size(Faded_signal)) + 1j * randn(size(Faded_signal))); % 가우시안 잡음
    noisySignal = Faded_signal + AWGN_signal;

    % 이퀄라이저 적용 (채널의 역수로 나누기)
    equalized_signal = noisySignal ./ Fading;

    % 2행 배열로 변환하여 반환
    z = [real(equalized_signal); imag(equalized_signal)];

    % 제로 패딩을 추가한 경우, 패딩된 원소를 제거
    if append
        z = z(1:end-1);
    end

    % 결과를 출력합니다.
    disp('Output z:');
    disp(z);
end





    
 %복소수 분리

% function z = predict(layer, x)
%     % 입력 신호의 총 원소 개수를 계산
%     numElem = numel(x);
% 
%     % 입력 신호가 홀수 개의 원소를 가진 경우, 제로 패딩을 추가.
%     if mod(numElem, 2)
%         append = true;
%         x = [x; zeros(1, size(x, 2))];
%     else
%         append = false;
%     end
% 
%     % 입력 신호를 2행으로 변환하여 실수부와 허수부로 분리
%     x2 = reshape(x, 2, []);
%     realPart = x2(1, :); % 실수부
%     imagPart = x2(2, :); % 허수부
% 
%     % Rayleigh fading을 적용
%     fadingReal = randn(size(realPart)); % 실수부의 Rayleigh fading
%     fadingImag = randn(size(imagPart)); % 허수부의 Rayleigh fading
%     fading = (fadingReal + 1j * fadingImag) / sqrt(2); % Rayleigh fading 벡터 생성
% 
%     fadedReal = realPart .* real(fading) - imagPart .* imag(fading);
%     fadedImag = realPart .* imag(fading) + imagPart .* real(fading);
% 
%     % AWGN을 추가
%     noisePower = 10^(-layer.LocalSNR/10); % 잡음 전력
%     noiseStdDev = sqrt(noisePower / 2); % 잡음의 표준편차 (실수부와 허수부)
%     noiseReal = noiseStdDev * randn(size(fadedReal)); % 실수부에 대한 잡음
%     noiseImag = noiseStdDev * randn(size(fadedImag)); % 허수부에 대한 잡음
% 
%     % 신호에 잡음을 추가합니다.
%     noisyReal = fadedReal + noiseReal;
%     noisyImag = fadedImag + noiseImag;
% 
%     % 실수부와 허수부를 결합하여 최종 복소수 신호를 생성
%     z2 = noisyReal + 1j * noisyImag;
% 
%     % 원래의 형태로 변환.
%     z = reshape(z2, 1,[]);
%   % z = reshape(z2, size(x));
%     % 제로 패딩을 추가한 경우, 패딩된 원소를 제거
%     if append
%         z = z(1:end-1, 1);
%     end
% 
%     % 결과를 출력합니다.
%     disp('Output z:');
%     disp(z);
% end



%Fading 적용
% 
% function z = predict(layer, x)
%     % 입력 x의 총 원소 개수를 계산
%     numElem = numel(x);
% 
%     % 입력 x가 홀수 개의 원소를 가진 경우, 제로 패딩을 추가
%     if mod(numElem, 2)
%         append = true;
%         x = [x; zeros(1, size(x, 2))];
%     else
%         append = false;
%     end
% 
%     % 복소수 신호를 실수부와 허수부로 나누기 위해 2행으로 변환
%     x2 = reshape(x, 2, []);
% 
%     % AWGN을 직접 구현
%     noisePower = 10^(-layer.LocalSNR/10); % 잡음 전력
%     noiseStdDev = sqrt(noisePower / 2); % 잡음의 표준편차
%     noise = noiseStdDev * (randn(size(x2)) + 1j * randn(size(x2))); % 복소수 잡음 생성
% 
%     % 신호에 잡음 추가
%     z2 = x2 + noise;
% 
%     % 원래의 형태로 변환
%     z = reshape(z2, size(x));
% 
%     % 제로 패딩을 추가한 경우, 패딩된 원소를 제거
%     if append
%         z = z(1:end-1, 1);
%     end
% end








% 원래 코드

  %   function z = predict(layer, x)
  % numElem = numel(x);
  % if mod(numElem, 2)
  %   append = true;
  %   x = [x; zeros(1, size(x, 2))];
  % else
  %   append = false;
  % end
  % 
  % x2 = reshape(x, 2, []);
  % z2 = awgn(x2, layer.LocalSNR, (layer.SignalPower) - 10 * log10(2));
  % z = reshape(z2, size(x));
  % 
  % if append
  %   z = z(1:end-1, 1);
  % end
  % 
  % %결과를 출력합니다.
  %   disp('Size of z:');
  %   disp(size(z));
  %   disp('Output z:');
  %   disp(z);
  % 
  % 
  %   end

    %%
    function dLdX = backward(layer, X, Z, dLdZ, memory)
      dLdX = dLdZ;
    end
    
    function sl = saveobj(layer)
      sl.NoiseMethod = layer.NoiseMethod;
      sl.EbNo = layer.EbNo;
      sl.EsNo = layer.EsNo;
      sl.SNR = layer.SNR;
      sl.BitsPerSymbol = layer.BitsPerSymbol;
      sl.SignalPower = sl.SignalPower;
      sl.LocalSNR = layer.LocalSNR;
    end
    
    function layer = reload(layer, sl)
      layer.NoiseMethod = sl.NoiseMethod;
      layer.EbNo = sl.EbNo;
      layer.EsNo = sl.EsNo;
      layer.SNR = sl.SNR;
      layer.BitsPerSymbol = sl.BitsPerSymbol;
      layer.SignalPower = sl.SignalPower;
      layer.LocalSNR = sl.LocalSNR;
    end
  end
  
  methods (Static)
    function layer = loadobj(sl)
      if isstruct(sl)
        layer = helperAEWAWGNLayer;
      else
        layer = sl;
      end
      layer = reload(layer, sl);
    end
  end
end
