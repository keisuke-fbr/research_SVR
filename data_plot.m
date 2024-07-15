data = readmatrix('diamond_train.csv'); % AMAZON社の5年間の株価(元のデータには複数の会社のデータが記載されてる)
data = data(1:1000,:);
X = data(:, 7); % 
Y = data(:, 8); % 当日の最高値

% プロット
figure;
plot(X, Y, 'o');
xlabel('前日の最高値');
ylabel('強度');
title('前日の最高値と当日の最高値の関係 (AMAZON社)');
grid on;