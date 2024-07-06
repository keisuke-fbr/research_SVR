% 水山先生の質問
% 「線形εインセンシティブ損失関数を用いた場合は, 絶対値誤差最小化問題を解いているはずなのに,
% 　なぜ2次epsilonインセンシティブ損失関数を用いた場合よりも予測精度の平均絶対値誤差が劣るのか？
% 　2次epsilonインセンシティブ損失関数を用いた場合は二乗誤差最小化問題を解いているのに？」

% 回答(イメージ)
% 「サポートベクターが少ないことが原因である。
% 　変動の大きいデータに対しては, 線形εインセンシティブ損失関数を用いた場合にサポートベクターの数が少なくなっていることはすでに述べた.
% 　回帰関数を作成する際に、サポートベクターの数が少なくなってしまうと, 上下の学習データのみを使って, 回帰関数を作成してしまう.
%   つまり, 上下のデータの真ん中を通るような回帰関数が出来上がってしまう. そのため, 平均絶対値誤差が悪化してしまう.


% 以下, サポートベクターが少なくなってしまうことで, 回帰関数がうまく作れなくなることを示す. 
% 簡単な例 論文では二重交差検証でパラメータをちゃんと決めているけど, 今回は意図的にサポートベクターを減らす

% 人工データの生成
num = 100;
X = 2*5*rand(1, num) - 5;
X = sort(X);
Y = X; % 直線状にデータを生成
X_plot = linspace(-5, 5, num*10);
X_plot = reshape(X_plot, num*10, 1);
Y_plot = X_plot;

% グラフの描画範囲
x_range = 5.5;
y_range = 7.2;

% ノイズを付与
rng default; % 乱数を固定
Y_noize = Y + normrnd(0, 1, 1, num); % 平均0, 標準偏差1の正規分布乱数を付与

% SVR用に入力データを整える(入力引数としてエラーが出ないように)
X = reshape(X, num, 1);
Y = reshape(Y, num, 1);
Y_noize = reshape(Y_noize, num, 1);

% %{
% 人工データを生成するもととなった関数とノイズを付与した後の人工データ
figure;
plot(X_plot, Y_plot, 'green', 'LineWidth', 1.0); % もととなった関数
hold on;
scatter(X, Y_noize, 'o', MarkerEdgeColor = 'black'); % ノイズを付与した後の人工データの散布図
xlim([-x_range x_range]);
ylim([-y_range y_range]);
set(gca,'FontSize',14) % 目盛りの大きさ
xlabel('$x$', 'FontSize',18, 'Interpreter', 'latex'); % x軸ラベルの大きさ
ylabel('$y$', 'FontSize',18, 'Interpreter', 'latex'); % y軸ラベルの大きさ
% saveas(gcf, '例に用いた人工データ(線形).eps', 'epsc') % このコードでグラフを画像ファイルとして保存
% saveas(gcf, '例に用いた人工データ(線形).jpg') % このコードでグラフを画像ファイルとして保存
% %}

% %{
% ハイパーパラメータ
% %{
epsilon = 0.5;
C = 10;
% %}

mdl_L1SVR = fitrsvm(X, Y_noize, 'BoxConstraint', C, 'KernelFunction', 'linear', 'Epsilon', epsilon);
Y_pred = predict(mdl_L1SVR, X_plot);

figure;
scatter(X(~mdl_L1SVR.IsSupportVector), Y_noize(~mdl_L1SVR.IsSupportVector), 'o', MarkerEdgeColor = 'black'); % サポートベクター以外をoで表示
hold on;
scatter(X(mdl_L1SVR.IsSupportVector), Y_noize(mdl_L1SVR.IsSupportVector), '*', MarkerEdgeColor = 'black'); % サポートベクターを*で表示
xlim([-x_range x_range]);
ylim([-y_range y_range]);
plot(X_plot, Y_plot, 'green', 'LineWidth', 1.0); % 人工データを生成するもととなった関数
plot(X_plot, Y_pred, 'blue', 'LineWidth', 1.0); % SVRの回帰関数のプロット
plot(X_plot, Y_pred + epsilon, 'black--', 'LineWidth', 1.0); % εチューブの表示(+ε)
plot(X_plot, Y_pred - epsilon, 'black--', 'LineWidth', 1.0); % εチューブの表示(-ε)
set(gca,'FontSize',14) % 目盛りの大きさ
xlabel('$x$', 'FontSize',18, 'Interpreter', 'latex'); % x軸ラベルの大きさ
ylabel('$y$', 'FontSize',18, 'Interpreter', 'latex'); % y軸ラベルの大きさ
% saveas(gcf, 'サポートベクターが多いとき(線形).eps', 'epsc') % このコードでグラフを画像ファイルとして保存
% saveas(gcf, 'サポートベクターが多いとき(線形).jpg') % このコードでグラフを画像ファイルとして保存

% サポートベクターの数を表示
disp("L1損失線形SVRのサポートベクターの数(epsilon=1, サポートベクターが多いとき)");
length(mdl_L1SVR.SupportVectors)
%}

% ハイパーパラメータ この例では意図的にサポートベクターを少なくする
% %{
epsilon = 5;
C = 10;
% %}

mdl_L1SVR = fitrsvm(X, Y_noize, 'BoxConstraint', C, 'KernelFunction', 'linear', 'Epsilon', epsilon);
Y_pred = predict(mdl_L1SVR, X_plot);

figure;
scatter(X(~mdl_L1SVR.IsSupportVector), Y_noize(~mdl_L1SVR.IsSupportVector), 'o', MarkerEdgeColor = 'black'); % サポートベクター以外をoで表示
hold on;
scatter(X(mdl_L1SVR.IsSupportVector), Y_noize(mdl_L1SVR.IsSupportVector), '*', MarkerEdgeColor = 'black'); % サポートベクターを*で表示
xlim([-x_range x_range]);
ylim([-y_range y_range]);
plot(X_plot, Y_plot, 'green', 'LineWidth', 1.0); % 人工データを生成するもととなった関数
plot(X_plot, Y_pred, 'blue', 'LineWidth', 1.0); % SVRの回帰関数のプロット
plot(X_plot, Y_pred + epsilon, 'black--', 'LineWidth', 1.0); % εチューブの表示(+ε)
plot(X_plot, Y_pred - epsilon, 'black--', 'LineWidth', 1.0); % εチューブの表示(-ε)
set(gca,'FontSize',14) % 目盛りの大きさ
xlabel('$x$', 'FontSize',18, 'Interpreter', 'latex'); % x軸ラベルの大きさ
ylabel('$y$', 'FontSize',18, 'Interpreter', 'latex'); % y軸ラベルの大きさ
% saveas(gcf, 'サポートベクターが少ないとき(線形).eps', 'epsc') % このコードでグラフを画像ファイルとして保存
% saveas(gcf, 'サポートベクターが少ないとき(線形).jpg') % このコードでグラフを画像ファイルとして保存

% サポートベクターの数を表示
disp("L1損失線形SVRのサポートベクターの数(epsilon=4, サポートベクターを少ないとき)");
length(mdl_L1SVR.SupportVectors)