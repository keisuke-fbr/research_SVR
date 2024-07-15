num = [1259];


for num_i = num
    rng default;

    data = readmatrix('AMZN_data.csv'); % AMAZON社の5年間の株価(元のデータには複数の会社のデータが記載されてる)
    X = data(:, 3); % 前日の最高値
    Y = data(:, 4); % 当日の最高値
   
    
    % SVRのハイパーパラメータの候補
    epsilon_L1 = [0, 5, 10, 15, 20, 25, 30];
    epsilon_L2 = [0, 5, 10, 15, 20, 25, 30];
    C_L1 = [0.1, 1, 10, 100];
    C_L2 = [0.1, 1, 10, 100];
    KernelScale = [10, 1, 1/sqrt(10), 0.1]; % Pythonにおけるgammaの役割を果たすパラメータ. gamma=1/(KernelScale^2)の関係.
    
    
    %最適パラメータ保持用
    epsilon_L1_data = zeros(1,5);
    C_L1_data = zeros(1,5);
    KernelScale_L1_data = zeros(1,5);

    epsilon_L2_data = zeros(1,5);
    C_L2_data = zeros(1,5);
    KernelScale_L2_data = zeros(1,5);


    %サポートベクターの割合を保持
    ratio_SV_L1 = zeros(1,5);
    ratio_SV_L2 = zeros(1,5);
    
    
    % グリッドサーチのための変数
    y_pred_Tuning_L1 = 0;
    min_absolute_error_L1 = Inf;   
    y_pred_Tuning_L2 = 0;
    min_squared_error_L2 = Inf;

    Epsilon_L1_opt = epsilon_L1(1);
    C_L1_opt = C_L1(1);
    KernelScale_L1_opt = KernelScale(1);
    
    Epsilon_L2_opt = epsilon_L2(1);
    C_L2_opt = C_L2(1);
    KernelScale_L2_opt = KernelScale(1);
    
    % 二重交差検証  
    rng('default'); % 乱数を固定しないと, データの分割が等しくなくなる
    cv1 = cvpartition(num_i, 'KFold', 5); % まず, データを学習データとテストデータに分ける(10分割. つまり9:1)
    
    y_learned_L1 = zeros(num_i,1);
    y_learned_L2 = zeros(num_i,1);

        
    
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
                    absolute_error_L1 = 0;
                    for j = 1:5 % 内側の交差検証で数えて, j番目の組み合わせ(5個)
                        train_idx = training(cv2, j); % 訓練データのindex(グリッドサーチの際の回帰モデル構築に使用)
                        validate_idx = test(cv2, j); % 検証データのindex(グリッドサーチの際の評価に使用)
    
                        % 訓練データを使って, 回帰モデル作成
                        try
                            disp("モデル作成開始L1:ハイパーパラメータ：ε="+string(e)+",C="+string(c)+",kernel="+string(k)+",内部"+string(j)+"回目、外部"+string(i)+"回目");
                            mdl_L1SVR_Tuning = fitrsvm(X(train_idx), Y(train_idx), "BoxConstraint", c, "KernelFunction", "rbf", "KernelScale", k, "Epsilon", e);
                            disp("モデル作成終了")
                        catch ME
                            ME_flag = 1; % もし最適化プログラムが収束しなかった場合はME_flag=1にして, break
                            break;
                        end
    
                        % 検証用データで平均絶対値誤差を算出
                        y_pred_Tuning_L1 = predict(mdl_L1SVR_Tuning, X(validate_idx));
                        absolute_error_L1 = absolute_error_L1 + mean(abs(y_pred_Tuning_L1 - Y(validate_idx)));
    
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
        mdl_L1SVR = fitrsvm(X(learn_idx), Y(learn_idx), "BoxConstraint", C_L1_opt, "KernelFunction", "rbf", "KernelScale", KernelScale_L1_opt, "Epsilon", Epsilon_L1_opt);
        
        y_pred_L1 = predict(mdl_L1SVR, X(test_idx));

        y_learned_L1(test_idx) = y_pred_L1;
        
        epsilon_L1_data(i) = Epsilon_L1_opt;
        C_L1_data(i) = C_L1_opt;
        KernelScale_L1_data(i) = KernelScale_L1_opt;

        ratio_SV_L1(i) = size(mdl_L1SVR.SupportVectors,1)/(size(X(learn_idx),1));
    end
    


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
                    squared_error_L2 = 0;
                    for j = 1:5 % 内側の交差検証で数えて, j番目の組み合わせ(5個)
                        train_idx = training(cv2, j); % 訓練データのindex(グリッドサーチの際の回帰モデル構築に使用)
                        validate_idx = test(cv2, j); % 検証データのindex(グリッドサーチの際の評価に使用)

                        % 訓練データを使って, 回帰モデル作成
                        try
                            disp("モデル作成開始L2:ハイパーパラメータ：ε="+string(e)+",C="+string(c)+",kernel="+string(k)+",内部"+string(j)+"回目、外部"+string(i)+"回目");
                            mdl_L2SVR_Tuning = fitrsvm2(X(train_idx), Y(train_idx), "BoxConstraint", c, "KernelFunction", "rbf", "KernelScale", k, "Epsilon", e, 'Solver', 'L1QP');
                            disp("モデル作成終了");
                        catch ME
                            ME_flag = 1; % もし最適化プログラムが収束しなかった場合はME_flag=1にして, break
                            break;
                        end

                        % 検証用データで平均二乗誤差を算出
                        y_pred_Tuning_L2 = predict(mdl_L2SVR_Tuning, X(validate_idx));
                        squared_error_L2 = squared_error_L2 + mean((y_pred_Tuning_L2 - Y(validate_idx)).^2);

                    end

                    if ME_flag == 1 % 最適化プログラムが収束しなかったハイパーパラメータの組み合わせは飛ばして, 次の組み合わせへ
                        ME_flag = 0;
                        continue;
                    end

                    % L2損失SVRの場合は,検証用データの予測値の平均絶対値誤差がこれまでのものよりも小さい場合は絶対値誤差とハイパーパラメータの値を更新
                    % 回帰関数を推定する際に, 損失を二乗で評価しているため
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
        mdl_L2SVR = fitrsvm2(X(learn_idx), Y(learn_idx), "BoxConstraint", C_L2_opt, "KernelFunction", "rbf", "KernelScale", KernelScale_L2_opt, "Epsilon", Epsilon_L2_opt, 'Solver', 'L1QP');
        y_pred_L2 = predict(mdl_L2SVR, X(test_idx));

        y_learned_L2(test_idx) = y_pred_L2;

        epsilon_L2_data(i) = Epsilon_L2_opt;
        C_L2_data(i) = C_L2_opt;
        KernelScale_L2_data(i) = KernelScale_L2_opt;

        ratio_SV_L2(i) = size(mdl_L2SVR.SupportVectors,1)/(size(X(learn_idx),1));
    end



    %L1の結果の表示


    disp(append('L1損失SVRの予測精度(データ数=', string(num_i), 'のとき)'));
    disp("真値との平均絶対値誤差");
    disp(mean(abs(Y-y_learned_L1)))
    disp("真値との最大絶対値誤差");
    disp(max(abs(Y-y_learned_L1)))

    disp("hp:ε:" + string(epsilon_L1_data));
    disp("hp:C:" + string(C_L1_data));
    disp("hp:k:" + string(KernelScale_L1_data));

    disp("svの割合:" + string(ratio_SV_L1));


    % プロットの作成
    figure;

    % Y_originalのプロット
    plot(X, Y, 'o-', 'LineWidth', 1.5); 
    hold on; 
    % y_pred_L1のプロット
    plot(X, y_learned_L1, 'g-', 'LineWidth', 1.5);
    %ε帯の描写
    plot(X, y_learned_L1 + mean(epsilon_L1_data), 'r--', 'LineWidth',0.5);
    plot(X, y_learned_L1 - mean(epsilon_L1_data), 'r--', 'LineWidth',0.5);
    % プロットの装飾
    title('L1:元関数, 観測値, 予測値のグラフ,データ数：' + string(num_i));
    xlabel('X');
    ylabel('Y');
    legend({'観測値', '予測値', 'ε帯'}, 'Location', 'Best');
    grid on;
    hold off;

     % プロットの作成
    figure;

    % インデックスの生成
    index = 1:num_i;

    % Y_originalのプロット
    plot(index, Y, 'o-', 'LineWidth', 1.5); 
    hold on; 
    % y_learned_L1のプロット
    plot(index, y_learned_L1, 'g-', 'LineWidth', 1.5);
    %ε帯の描写
    % εが0の場合、プロットの意味がなくなるので描写しないか、固定値を与える
    plot(index, y_learned_L1 + mean(epsilon_L1_data), 'r--', 'LineWidth',0.5);
    plot(index, y_learned_L1 - mean(epsilon_L1_data), 'r--', 'LineWidth',0.5);
    % プロットの装飾
    title('L1:元関数, 観測値, 予測値のグラフ, データ数：' + string(num_i));
    xlabel('インデックス');
    ylabel('当日の最高値');
    legend({'観測値', '予測値', 'ε帯'}, 'Location', 'Best');
    grid on;
    hold off;

    %L2の結果の表示

    disp(append('L2損失SVRの予測精度(データ数=', string(num_i), 'のとき)'));
    disp("真値との平均絶対値誤差");
    disp(mean(abs(Y -y_learned_L2)))
    disp("真値との最大絶対値誤差");
    disp(max(abs(Y-y_learned_L2)))

    disp("hpの出力:ε:" + string(epsilon_L2_data));
    disp("hpの出力:C:" + string(C_L2_data));
    disp("hpの出力:k:" + string(KernelScale_L2_data));

    disp("svの割合:" + string(ratio_SV_L2));


    % プロットの作成
    figure;

    % Y_originalのプロット
    plot(X, Y, 'b-', 'LineWidth', 1.5); 
    hold on; 
    % Y_observationのプロット
    plot(X, Y, 'ko', 'MarkerSize', 5);
    % y_pred_L2のプロット
    plot(X, y_learned_L2, 'g-', 'LineWidth', 1.5);
    %ε帯の描写
    plot(X, y_learned_L2 + mean(epsilon_L2_data), 'r--', 'LineWidth',0.5);
    plot(X, y_learned_L2 - mean(epsilon_L2_data), 'r--', 'LineWidth',0.5);
    % プロットの装飾
    title('L2:元関数, 観測値, 予測値のグラフ'+ string(num_i));
    xlabel('X');
    ylabel('Y');
    legend({'元関数', '観測値', '予測値', 'ε帯'}, 'Location', 'Best');
    grid on;
    hold off;
    
end