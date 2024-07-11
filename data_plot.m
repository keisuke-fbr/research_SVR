% サンプルデータの生成とプロット
x = linspace(0, 100, 1000);  % xの範囲は0から100
y_noisy = generate_observation_data(x);  % 関数を使用してノイズ付きデータを生成

% 結果をプロット（点で表示）
figure;
scatter(x, y_noisy, '.');
title('Noisy Periodic Function with Different Noise Levels');
xlabel('x');
ylabel('y');
grid on;