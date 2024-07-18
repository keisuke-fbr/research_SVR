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

% ラベルの設定
variableNames = {'cement', 'slag', 'ash', 'water', 'superplasticizer', 'ca','fa','age'};

matrix_data = [cement,slag,ash,water,superplasticizer,ca,fa,age];

R = corrcoef(matrix_data);
disp(R);
figure;

heatmap(variableNames,variableNames,R);


title("heatmap");
