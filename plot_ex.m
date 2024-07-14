% 例のデータ


data = readmatrix('Concrete_Data.xls'); % コンクリートに関するデータ(1030個)
X1 = data(:, 1) ; % セメントの量, 水の量, 粗骨材の量, 細骨材の量
X2 = data(:, 4); % コンクリートの強度
Y = data(:,9);


% 3D散布図の作成
figure;
scatter3(X1, X2, Y, 'filled');
xlabel('X1軸');
ylabel('X2軸');
zlabel('Y軸');
title('3D Scatter Plot');
grid on;
