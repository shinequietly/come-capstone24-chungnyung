classdef helperAEWNormalizationLayer < nnet.layer.Layer
%helperAEWNormalizationLayer Wireless symbol normalization layer
%   layer = helperAEWNormalizationLayer creates a wireless symbol
%   normalization layer. 
%
%   layer = helperAEWNormalizationLayer('PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%       'Name'   - A name for the layer. The default is ''.
%       'Method' - Normalization method as one of 'Energy' and 
%                  'Average power'. The default is 'Energy'.
%
%   Example:
%       % Create a normalization layer for energy normalization.
%
%       layer = helperAEWNormalizationLayer('Method','Energy');
%
%   See also AutoencoderForWirelessCommunicationsExample, helperAEWEncode,
%   helperAEWDecode, helperAEWAWGNLayer.

%   Copyright 2020 The MathWorks, Inc.


% 정규화 방법 ('Energy' 또는 'Average power')
  properties
    Method {mustBeMember(Method,{'Energy','Average power'})} = 'Energy'
  end
  
  methods
    function layer = helperAEWNormalizationLayer(varargin)
       % 생성자: 사용자 입력을 기반으로 계층의 속성을 초기화
      p = inputParser;
      addParameter(p,'Method','Energy')
      addParameter(p,'Name','wnorm')
      addParameter(p,'Description','')
      
      parse(p,varargin{:})
      
      layer.Method = p.Results.Method;
      layer.Name = p.Results.Name;
      if isempty(p.Results.Description)
        layer.Description = p.Results.Method + " normalization layer";
      else
        layer.Description = p.Results.Description;
      end
      
      layer.Type = 'Wireless Normalization';   % 레이어 유형을 설정
    end
    
    function z = predict(layer, x)
        % 입력 데이터를 정규화하여 출력을 생성
      %
      % Inputs:
      %         layer  - 레이어 객체
      %         x      - 입력 샘플 (심볼)
      % Outputs:
      %         z      - 정규화된 샘플



      n = size(x,1);
        
      if strcmp(layer.Method, 'Energy')
             z = x ./ sqrt(sum(x.^2)/(n/2));
        % 정규화 방법이 'Energy'인 경우:
        % 심볼의 에너지를 일정하게 유지하기 위해 정규화
        % 입력 데이터를 각 심볼의 에너지의 제곱근으로 나눔
        % 이때 n/2는 I/Q 성분으로 인해 2로 나눔.
     


      
      elseif strcmp(layer.Method, 'Average power')
        z = x ./ sqrt(mean(x.^2,'all')*2);
        % 정규화 방법이 'Average power'인 경우:
        % 심볼의 평균 전력을 일정하게 유지하기 위해 정규화
        % 입력 데이터를 평균 전력의 제곱근으로 나눔.
        % 여기서 `mean(x.^2,'all')`은 모든 요소에 대해 제곱한 후 평균을 구함.
        % *2는 I/Q 성분을 반영하기 위해 곱해줌.

      end
    end
  end
end