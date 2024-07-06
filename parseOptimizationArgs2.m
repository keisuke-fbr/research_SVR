function [IsOptimizing, RemainingArgs] = parseOptimizationArgs2(Args)
% Find the two NVPs 'OptimizeHyperparameters' and
% 'HyperparameterOptimizationOptions'. Error if
% 'HyperparameterOptimizationOptions' is there without
% 'OptimizeHyperparameters'. Return true if optimization is requested.
% Return RemainingArgs as the arglist without those two NVPs.

%   Copyright 2016-2019 The MathWorks, Inc.

[OptimizeHyperparameters,HyperparameterOptimizationOptions,~,RemainingArgs] = internal.stats.parseArgs(...
    {'OptimizeHyperparameters', 'HyperparameterOptimizationOptions'}, {[], []}, Args{:});
if isempty(OptimizeHyperparameters) && ~isempty(HyperparameterOptimizationOptions) && ~isPrefixEqual(HyperparameterOptimizationOptions, 'none')
    bayesoptim.err('OptimOptionsPassedAlone');
end
IsOptimizing = ~isempty(OptimizeHyperparameters) && ~isPrefixEqual(OptimizeHyperparameters, 'none');
end

function tf = isPrefixEqual(thing, targetString)
tf = ~isempty(thing) && ischar(thing) && strncmpi(thing, targetString, length(thing));
end