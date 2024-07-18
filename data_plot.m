data = readmatrix('Concrete_Data.xls'); % AMAZON社の5年間の株価(元のデータには複数の会社のデータが記載されてる)
data = data(1:1000,:);
cement = data(:, 1);
slag = data(:, 2);
ash = data(:, 3);
water = data(:,4 );
superplasticizer = data(:,5 );
ca = data(:, 6);
fa = data(:, 7);
age = data(:, 8);

% プロット
figure;
plot(X, Y, 'o');
xlabel('前日の最高値');
ylabel('強度');
title('前日の最高値と当日の最高値の関係 (AMAZON社)');
grid on;