% 人工データ(2次元) 大きな変動が欠損値に依存する場合
% 曲線上にデータを生成し, 以下の二つの条件を併せて考えて, 予測精度を比較
% (1) 欠損値を考慮(一定範囲のデータを削除してみる. 約10%)
% (2) データ数を変化

num = [100, 500, 1000, 2000, 3000, 5000];
std = [1]; % ホワイトノイズの標準偏差
X_range = 5; % Xの生成範囲の上限

for num_i = num
    rng default;
    X = 2*X_range*rand(1, num_i) - X_range;
    X = sort(X);
    Y = 5*sin(pi*X); % 曲線状にデータを生成

    % SVR用に入力データを整える(入力引数としてエラーが出ないように)
    X = reshape(X, num_i, 1);
    Y = reshape(Y, num_i, 1);

    %{
    figure;
    plot(X, Y, 'green'); % もととなった関数
    hold on;
    xlim([-X_range-0.5, X_range+0.5]);
    ylim([-15, 15]);
    set(gca,'FontSize',14) % 目盛りの大きさ
    xlabel('$x$', 'FontSize',18,'Interpreter','latex'); % x軸ラベルの大きさ
    ylabel('$y$', 'FontSize',18,'Interpreter','latex'); % y軸ラベルの大きさ
    % saveas(gcf, '人工データを生成するもとになった関数.jpg') % このコードでグラフを画像ファイルとして保存
    %}

    for std_i = std
        % ノイズの付与
        rng default;
        Y_noize = Y + normrnd(0, std_i, num_i, 1); % Yに平均0 標準偏差=std_iの正規分布のノイズを付与

        % 欠損値の作成
        range_l = 0;
        range_u = 1;
        missing_idx = find(X >= range_l & X <= range_u);
        X(missing_idx) = [];
        Y_noize(missing_idx) = [];
        Y(missing_idx) = [];
        num_missing = length(Y_noize);

        % disp(append('(データ数=', string(num_missing), ', 欠損値有りのとき)'));

        %{
        % ノイズを付与し, 欠損値も考慮した後の人工データ
        figure;
        scatter(X, Y_noize, 'o', MarkerEdgeColor = 'black'); % ノイズを付与した後の人工データの散布図
        hold on;
        xlim([-X_range-0.5, X_range+0.5]);
        ylim([-15, 15]);
        set(gca,'FontSize',14) % 目盛りの大きさ
        xlabel('$x$', 'FontSize',18,'Interpreter','latex'); % x軸ラベルの大きさ
        ylabel('$y$', 'FontSize',18,'Interpreter','latex'); % y軸ラベルの大きさ
        % saveas(gcf, append('人工データ(データ数=', string(num_missing), ', 欠損値ありのとき).jpg')) % このコードでグラフを画像ファイルとして保存
        %}
        
        % SVRのハイパーパラメータの候補
        epsilon_L1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5]; 
        epsilon_L2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
        C_L1 = [0.1, 1, 10, 100];
        C_L2 = [0.1, 1, 10, 100];
        KernelScale = [10, sqrt(10), 1]; % Pythonにおけるgammaと同じ役割. ただし, gamma=1/(KernelScale^2)の関係. KernelScale = [10, sqrt(10), 1]は gamma=0.01, 0.1, 1と同じ関係

        Epsilon_L1_opt = epsilon_L1(1);
        C_L1_opt = C_L1(1);
        KernelScale_L1_opt = KernelScale(1);
        
        Epsilon_L2_opt = epsilon_L2(1);
        C_L2_opt = C_L2(1);
        KernelScale_L2_opt = KernelScale(1);
        
        
        % グリッドサーチのための変数
        y_pred_Tuning_L1 = 0;
        min_absolute_error_L1 = Inf;
        
        y_pred_Tuning_L2 = 0;
        min_squared_error_L2 = Inf;
        
        % 二重交差検証
        
        rng('default'); % 乱数を固定しないと, データの分割が等しくなくなる
        cv1 = cvpartition(num_missing, 'KFold', 5); % まず, データを学習データとテストデータに分ける(10分割. つまり9:1)
        
        y_plot_L1 = double.empty(0,2);
        y_plot_L2 = double.empty(0,2);

        
        % %{
        % L1損失SVR
        for i = 1:5 % 外側の交差検証で数えて, i番目の組み合わせ(5個)
        
            learn_idx = training(cv1, i); % 学習データのindex(のちに訓練データ, 検証データに分ける)
            test_idx = test(cv1, i); % テストデータのindex(のちに予測精度の算出に利用)
            rng('default'); % 乱数を固定しないと, データの分割が等しくなくなる
            cv2 = cvpartition(nnz(learn_idx), 'KFold', 5); % 学習データを訓練データと検証データに分ける(5分割. 4:1). nnz(learn_idx)は学習データの数
        
            ME_flag = 0; % 最適化プログラムが収束しなかったら1.
        
            % 内側の交差検証でグリッドサーチ
            for e = epsilon_L1
                for c = C_L1
                    for k = KernelScale
                        absolute_error_L1 = Inf;
                        for j = 1:5 % 内側の交差検証で数えて, j番目の組み合わせ(5個)
                            train_idx = training(cv2, j); % 訓練データのindex(グリッドサーチの際の回帰モデル構築に使用)
                            validate_idx = test(cv2, j); % 検証データのindex(グリッドサーチの際の評価に使用)
        
                            % 訓練データを使って, 回帰モデル作成
                            try
                                mdl_L1SVR_Tuning = fitrsvm(X(train_idx), Y_noize(train_idx), "BoxConstraint", c, "KernelFunction", "rbf", "KernelScale", k, "Epsilon", e);
                            catch ME
                                ME_flag = 1; % もし最適化プログラムが収束しなかった場合はME_flag=1にして, break
                                break;
                            end
        
                            % 検証用データで平均絶対値誤差を算出
                            y_pred_Tuning_L1 = predict(mdl_L1SVR_Tuning, X(validate_idx));
                            absolute_error_L1 = absolute_error_L1 + mean(abs(y_pred_Tuning_L1 - Y_noize(validate_idx)));
        
                        end
                        
                        if ME_flag == 1 % 最適化プログラムが収束しなかったハイパーパラメータの組み合わせは飛ばして, 次の組み合わせへ
                            ME_flag = 0;
                            continue;
                        end
                        
                        % L1損失SVRの場合は,検証用データの予測値の平均絶対値誤差がこれまでのものよりも小さい場合は絶対値誤差とハイパーパラメータの値を更新
                        % 回帰関数を推定する際に, L1損失で評価しているため
                        if absolute_error_L1 <= min_absolute_error_L1
                            min_absolute_error_L1 = absolute_error_L1;
                            Epsilon_L1_opt = e;
                            C_L1_opt = c;
                            KernelScale_L1_opt =k;
                        end
        
                    end
                end
            end
        
            % 内側の交差検証でハイパーパラメータが決まったら, 外側の交差検証で予測精度を算出
            % 学習データ(訓練データ+検証データ)を使い, 内側の交差検証で求めたハイパーパラメータで回帰モデルを作成
            mdl_L1SVR = fitrsvm(X(learn_idx), Y_noize(learn_idx), "BoxConstraint", C_L1_opt, "KernelFunction", "rbf", "KernelScale", KernelScale_L1_opt, "Epsilon", Epsilon_L1_opt);
            y_pred_L1 = predict(mdl_L1SVR, X(test_idx));
            y_plot_L1 = [y_plot_L1; Y(test_idx), y_pred_L1]; % これで実測値と予測値が格納されていく. 最終的に500行2列. 1列目に実測値. 2列目に予測値(重複はなし)
        end
        
        %{
        % L1損失SVRの予測精度
        figure;
        scatter(y_plot_L1(:, 1), y_plot_L1(:, 2), 'o', MarkerEdgeColor = 'black', LineWidth = 1); % 横軸がYの実測値, 縦軸がYの予測値とした散布図(予測がうまくいっていれば, y=x(右斜め45度にプロットされる))
        hold on;
        plot(y_plot_L1(:, 1), y_plot_L1(:, 1), 'red'); % 斜め45度の直線
        set(gca,'FontSize',14) % 目盛りの大きさ
        xlabel('true', 'FontSize',18); % x軸ラベルの大きさ
        ylabel('pred', 'FontSize',18); % y軸ラベルの大きさ
        % saveas(gcf, append('L1損失SVRの予測精度(データ数=', string(num_missing), ', 欠損値有りのとき).jpg')) % このコードでグラフを画像ファイルとして保存
        %}

        disp(append('L1損失SVRの予測精度(データ数=', string(num_missing), ', 欠損値有りのとき)'));
        disp("真値との平均絶対値誤差");
        mean(abs(y_plot_L1(:, 1)-y_plot_L1(:, 2)))
        disp("真値との最大絶対値誤差");
        max(abs(y_plot_L1(:, 1)-y_plot_L1(:, 2)))
        % %}
        
        % %{
        % L2損失SVR
        for i = 1:5 % 外側の交差検証で数えて, i番目の組み合わせ(5個)
        
            learn_idx = training(cv1, i); % 学習データのindex(のちに訓練データ, 検証データに分ける)
            test_idx = test(cv1, i); % テストデータのindex(のちに予測精度の算出に利用)
            rng('default'); % 乱数を固定しないと, データの分割が等しくなくなる
            cv2 = cvpartition(nnz(learn_idx), 'KFold', 5); % 学習データを訓練データと検証データに分ける(5分割. 4:1). nnz(learn_idx)は学習データの数
        
            ME_flag = 0; % 最適化プログラムが収束しなかったら1.
        
            % 内側の交差検証でグリッドサーチ
            for e = epsilon_L2
                for c = C_L2
                    for k = KernelScale
                        squared_error_L2 = Inf;
                        for j = 1:5 % 内側の交差検証で数えて, j番目の組み合わせ(5個)
                            train_idx = training(cv2, j); % 訓練データのindex(グリッドサーチの際の回帰モデル構築に使用)
                            validate_idx = test(cv2, j); % 検証データのindex(グリッドサーチの際の評価に使用)
        
                            % 訓練データを使って, 回帰モデル作成
                            try
                                mdl_L2SVR_Tuning = fitrsvm2(X(train_idx), Y_noize(train_idx), "BoxConstraint", c, "KernelFunction", "rbf", "KernelScale", k, "Epsilon", e, 'Solver', 'L1QP');
                            catch ME
                                ME_flag = 1; % もし最適化プログラムが収束しなかった場合はME_flag=1にして, break
                                break;
                            end
        
                            % 検証用データで平均二乗誤差を算出
                            y_pred_Tuning_L2 = predict(mdl_L2SVR_Tuning, X(validate_idx));
                            squared_error_L2 = squared_error_L2 + mean((y_pred_Tuning_L2 - Y_noize(validate_idx)).^2);
        
                        end
                        
                        if ME_flag == 1 % 最適化プログラムが収束しなかったハイパーパラメータの組み合わせは飛ばして, 次の組み合わせへ
                            ME_flag = 0;
                            continue;
                        end
                        
                        % L2損失SVRの場合は,検証用データの予測値の平均絶対値誤差がこれまでのものよりも小さい場合は絶対値誤差とハイパーパラメータの値を更新
                        % 回帰関数を推定する際に, L2損失で評価しているため
                        if squared_error_L2 <= min_squared_error_L2
                            min_squared_error_L2 = squared_error_L2;
                            Epsilon_L2_opt = e;
                            C_L2_opt = c;
                            KernelScale_L2_opt =k;
                        end
        
                    end
                end
            end
        
            % 内側の交差検証でハイパーパラメータが決まったら, 外側の交差検証で予測精度を算出
            % 学習データ(訓練データ+検証データ)を使い, 内側の交差検証で求めたハイパーパラメータで回帰モデルを作成
            mdl_L2SVR = fitrsvm2(X(learn_idx), Y_noize(learn_idx), "BoxConstraint", C_L2_opt, "KernelFunction", "rbf", "KernelScale", KernelScale_L2_opt, "Epsilon", Epsilon_L2_opt, 'Solver', 'L1QP');
            y_pred_L2 = predict(mdl_L2SVR, X(test_idx));
            y_plot_L2 = [y_plot_L2; Y(test_idx), y_pred_L2]; % これで実測値と予測値が格納されていく. 最終的に500行2列. 1列目に実測値. 2列目に予測値(重複はなし)
        end
        
        %{
        % L2損失SVRの予測精度
        figure;
        scatter(y_plot_L2(:, 1), y_plot_L2(:, 2), 'o', MarkerEdgeColor = 'black', LineWidth = 1); % 横軸がYの実測値, 縦軸がYの予測値とした散布図(予測がうまくいっていれば, y=x(右斜め45度にプロットされる))
        hold on;
        plot(y_plot_L2(:, 1), y_plot_L2(:, 1), 'red'); % 斜め45度の直線
        set(gca,'FontSize',14) % 目盛りの大きさ
        xlabel('true', 'FontSize',18); % x軸ラベルの大きさ
        ylabel('pred', 'FontSize',18); % y軸ラベルの大きさ
        % title("L2損失SVR");
        % saveas(gcf, append('L2損失SVRの予測精度(データ数=', string(num_missing), ', 欠損値有りのとき).jpg')) % このコードでグラフを画像ファイルとして保存
        %}

        disp(append('L2損失SVRの予測精度(データ数=', string(num_missing), ', 欠損値有りのとき)'));
        disp("真値との平均絶対値誤差");
        mean(abs(y_plot_L2(:, 1)-y_plot_L2(:, 2)))
        disp("真値との最大絶対値誤差");
        max(abs(y_plot_L2(:, 1)-y_plot_L2(:, 2)))
        % %}
    end
end