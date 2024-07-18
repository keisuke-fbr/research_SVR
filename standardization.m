data = readmatrix('Concrete_Data.xls'); % AMAZON社の5年間の株価(元のデータには複数の会社のデータが記載されてる)
data = data(1:100,:);
cement = data(:, 1);
water = data(:,4 );

cement_st = (cement - mean(cement)) ./ std(cement);
water_st = (water - mean(water)) ./ std(water);


disp(cement_st);
disp(water_st);
