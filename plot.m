% 例のデータ


data = readmatrix('diamond_train.csv'); % コンクリートに関するデータ(1030個)
data = data(1:1000,:);
X1 = data(:, 2) ; % セメントの量, 水の量, 粗骨材の量, 細骨材の量
X2 = data(:, 6); % コンクリートの強度
Y = data(:,8);


% 3D散布図の作成
figure;
scatter3(X1, X2, Y, 'filled');
xlabel('X1軸');
ylabel('X2軸');
zlabel('Y軸');
title('3D Scatter Plot');
grid on;
