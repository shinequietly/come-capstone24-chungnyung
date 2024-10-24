% 파라미터 설정
N = 1000;                % 각 신호의 비트 수
M_QPSK = 4;              % QPSK 변조 차수
M_BPSK = 2;              % BPSK 변조 차수
fc_QPSK = 2e3;           % QPSK 반송파 주파수 (2 kHz)
fc_BPSK = 5e3;           % BPSK 반송파 주파수 (5 kHz)
fs = 15e3;               % 샘플링 주파수 (15 kHz)
t = (0:N-1)/fs;          % 시간 벡터

% 무작위 비트 생성
bits_QPSK = randi([0 1], N, 1);  % QPSK용 비트 생성
bits_BPSK = randi([0 1], N, 1);  % BPSK용 비트 생성

% QPSK 변조
symbols_QPSK = pskmod(bits_QPSK, M_QPSK, pi/M_QPSK);

% BPSK 변조
symbols_BPSK = pskmod(bits_BPSK, M_BPSK);

% 반송파 생성 (QPSK와 BPSK 각각 다른 주파수)
carrier_QPSK = cos(2 * pi * fc_QPSK * t');  % QPSK 반송파 (2 kHz)
carrier_BPSK = cos(2 * pi * fc_BPSK * t');  % BPSK 반송파 (5 kHz)

% FDM 적용: QPSK 및 BPSK 신호를 각각 다른 주파수 대역으로 변조
tx_signal_QPSK = real(symbols_QPSK) .* carrier_QPSK;
tx_signal_BPSK = real(symbols_BPSK) .* carrier_BPSK;

% 결합된 신호 전송 (FDM)
combined_signal = tx_signal_QPSK + tx_signal_BPSK;

% 저역 통과 필터 설계 (Moving Average Filter)
N_filter = 50;  % 필터 차수 (Moving Average Filter 차수)
lpf = ones(1, N_filter) / N_filter;  % 단순 이동 평균 필터 설계

% 수신기에서 신호 복원

% QPSK 신호 복조: 반송파 곱셈
received_QPSK = combined_signal .* cos(2 * pi * fc_QPSK * t');
% BPSK 신호 복조: 반송파 곱셈
received_BPSK = combined_signal .* cos(2 * pi * fc_BPSK * t');

% 저역 통과 필터 적용 (Moving Average Filter)
filtered_QPSK = filter(lpf, 1, received_QPSK);  % 필터 적용
filtered_BPSK = filter(lpf, 1, received_BPSK);  % 필터 적용

% QPSK 신호 복조 (I 채널)
demod_bits_QPSK = pskdemod(real(filtered_QPSK), M_QPSK, pi/M_QPSK);

% BPSK 신호 복조 (Q 채널)
demod_bits_BPSK = pskdemod(real(filtered_BPSK), M_BPSK);

% 비트 오류율(BER) 계산
num_errors_QPSK = sum(bits_QPSK ~= demod_bits_QPSK);  % QPSK 오류 비트 수
num_errors_BPSK = sum(bits_BPSK ~= demod_bits_BPSK);  % BPSK 오류 비트 수

BER_QPSK = num_errors_QPSK / N;  % QPSK 비트 오류율
BER_BPSK = num_errors_BPSK / N;  % BPSK 비트 오류율

% 결과 출력
fprintf('QPSK Bit Error Rate: %f\n', BER_QPSK);
fprintf('BPSK Bit Error Rate: %f\n', BER_BPSK);
