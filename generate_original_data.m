function y = generate_original_data(x)
    % Define y values based on x ranges with even shorter periods (doubling the frequency again)
    y = zeros(size(x));
    y(x <= 30) = 15 + 15 * sin(8 * pi * x(x <= 30) / 30);  % Even shorter period for 0-30 range
    y(x > 30) = 90 + 10 * sin(8 * pi * (x(x > 30) - 30) / 70);  % Even shorter period for 30-100 range
end