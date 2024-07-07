% データ数
num_i = 100; % 必要に応じて変更してください

% 90%のデータを0から30の範囲で生成
num_0_30 = round(0.9 * num_i);
x_0_30 = 30 * rand(num_0_30, 1);

% 10%のデータを30から100の範囲で生成
num_30_100 = num_i - num_0_30;
x_30_100 = 30 + (100 - 30) * rand(num_30_100, 1);

% すべてのデータを結合
X = [x_0_30; x_30_100];

% データをソート（必要に応じて）
X = sort(X);

% Xに対してyを生成
Y = generate_original_data(X);

% Xに対してノイズを加えたyを生成
Y_noisy = generate_observation_data(X);


% プロットして確認
figure;
hold on;

% Y_originalのプロット（黒い線）
plot(X, Y, 'k-', 'LineWidth', 1.5); 

% Y_observationのプロット（黒い点）
plot(X, Y_noisy, 'ko', 'MarkerSize', 5);

% プロットの装飾
title('元関数と観測データのグラフ');
xlabel('X');
ylabel('Y');
legend({'元関数', '観測データ'}, 'Location', 'Best');
grid on;
hold off;