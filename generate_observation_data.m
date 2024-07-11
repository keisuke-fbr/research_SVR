function y_noisy = generate_observation_data(x)
    % 基本となる周期関数（-10から10の範囲）
    y_base = 10 * sin(x);
    
    % ホワイトノイズの追加
    num_points = length(x);
    noise1 = randn(1, num_points) * 10;  % 標準偏差10のホワイトノイズ
    noise2 = randn(1, num_points) * 50;  % 標準偏差50のホワイトノイズ

    % 90%のデータに標準偏差10のホワイトノイズを適用
    noise = noise1;

    % 10%のデータに標準偏差50のホワイトノイズを適用
    num_high_noise_points = round(0.1 * num_points);
    high_noise_indices = randperm(num_points, num_high_noise_points);
    noise(high_noise_indices) = noise2(high_noise_indices);

    % yにノイズを追加
    y_noisy = y_base + noise;
end