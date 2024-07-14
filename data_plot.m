data = readmatrix('AMZN_data.csv'); % AMAZON社の5年間の株価(元のデータには複数の会社のデータが記載されてる)
X = data(:, 3); % 前日の最高値
Y = data(:, 4); % 当日の最高値

% プロット
figure;
plot(X, Y, 'o');
xlabel('前日の最高値');
ylabel('当日の最高値');
title('前日の最高値と当日の最高値の関係 (AMAZON社)');
grid on;