function [varargout] = fitoptimizing2(FitFunctionName, Predictors, Response, varargin)
%

%   Copyright 2016-2020 The MathWorks, Inc.

% Check for tall data compatibility
IsTallData = istall(Predictors);
SupportedForTall = {'fitcdiscr','fitcecoc','fitckernel','fitclinear','fitctree','fitrkernel','fitrlinear','fitrtree'};
if IsTallData && ~ismember(FitFunctionName, SupportedForTall)
    classreg.learning.paramoptim.err('FitFcnOptimNotSupportedForTall', FitFunctionName);
end
% Parse and check args
verifyNoValidationArgs(varargin);
NumObservations = size(Predictors, 1);
[OptimizeHyperparametersArg, HyperparameterOptimizationOptions, FitFunctionArgs] = ...
    parseFitoptimizingArgs2(varargin, IsTallData, NumObservations, false);
% Create optimization info
BOInfo = BayesoptInfo2.makeBayesoptInfo(FitFunctionName, Predictors, Response, FitFunctionArgs);
% Make sure we have a non-null optimization problem
iVerifyValidOptimizationProblem(BOInfo);
% Create objective function
[ValidationMethod, ValidationVal] = getPassedValidationArgs(HyperparameterOptimizationOptions);
objFcn = classreg.learning.paramoptim.createObjFcn(BOInfo, FitFunctionArgs, Predictors, Response, ...
    ValidationMethod, ValidationVal, HyperparameterOptimizationOptions.Repartition, HyperparameterOptimizationOptions.Verbose,...
    HyperparameterOptimizationOptions.Optimizer);
% Perform optimization
VariableDescriptions = getVariableDescriptions(BOInfo, OptimizeHyperparametersArg);
switch HyperparameterOptimizationOptions.Optimizer
    case 'bayesopt'
        [OptimizationResults, XBest] = doBayesianOptimization(objFcn, BOInfo, VariableDescriptions, HyperparameterOptimizationOptions);
    case 'gridsearch'
        [OptimizationResults, XBest] = doNonBayesianOptimization('grid', objFcn, BOInfo, VariableDescriptions, HyperparameterOptimizationOptions);
    case 'randomsearch'
        [OptimizationResults, XBest] = doNonBayesianOptimization('random', objFcn, BOInfo, VariableDescriptions, HyperparameterOptimizationOptions);
end
% Fit final model using best parameters and return/store optimization results
if isempty(XBest)
    classreg.learning.paramoptim.warn('NoFinalModel');
    [varargout{1:nargout}] = [];
else
    if BOInfo.CanStoreResultsInModel
        [varargout{1:nargout}] = classreg.learning.paramoptim.fitToFullDataset(XBest, BOInfo, ...
            FitFunctionArgs, Predictors, Response);
        varargout{1} = setParameterOptimizationResults(varargout{1}, OptimizationResults);
    elseif nargout == BOInfo.OutputArgumentPosition
        % Return results as output arg
        [varargout{1:nargout-1}] = classreg.learning.paramoptim.fitToFullDataset(XBest, BOInfo, ...
            FitFunctionArgs, Predictors, Response);
        varargout{BOInfo.OutputArgumentPosition} = OptimizationResults;
    else
        % Can't store Results in model, and not requested as output arg
        [varargout{1:nargout}] = classreg.learning.paramoptim.fitToFullDataset(XBest, BOInfo, ...
            FitFunctionArgs, Predictors, Response);
    end
end
end

function [OptimizationResults, XBest] = doBayesianOptimization(objFcn, BOInfo, ...
    VariableDescriptions, HyperparameterOptimizationOptions)
% Create args to bayesopt
if HyperparameterOptimizationOptions.ShowPlots
    PlotFcn = {@plotMinObjective};
    if sum([VariableDescriptions.Optimize]) <= 2
        PlotFcn{end+1} = @plotObjectiveModel;
    end
else
    PlotFcn = {};
end
if HyperparameterOptimizationOptions.SaveIntermediateResults
    OutputFcn = @assignInBase;
else
    OutputFcn = {};
end
% Call bayesopt
OptimizationResults = bayesopt(objFcn, VariableDescriptions, ...
    'AcquisitionFunctionName', HyperparameterOptimizationOptions.AcquisitionFunctionName,...
    'MaxObjectiveEvaluations', HyperparameterOptimizationOptions.MaxObjectiveEvaluations, ...
    'MaxTime', HyperparameterOptimizationOptions.MaxTime, ...
    'XConstraintFcn', BOInfo.XConstraintFcn, ...
    'ConditionalVariableFcn', BOInfo.ConditionalVariableFcn, ...
    'Verbose', HyperparameterOptimizationOptions.Verbose, ...
    'UseParallel', HyperparameterOptimizationOptions.UseParallel, ...
    'PlotFcn', PlotFcn,...
    'OutputFcn', OutputFcn,...
    'AlwaysReportObjectiveErrors', true,...
    'IsClassregRegressionFunction', BOInfo.IsRegression);

% Choose best point
XBest = chooseBestPointBayesopt(OptimizationResults);
end

function [OptimizationResults, XBest] = doNonBayesianOptimization(AFName, objFcn, BOInfo, ...
    VariableDescriptions, HyperparameterOptimizationOptions)
% Create args to bayesopt
if HyperparameterOptimizationOptions.ShowPlots
    PlotFcn = {@plotMinObjective};
else
    PlotFcn = {};
end
% Call bayesopt
BOResults = bayesopt(objFcn, VariableDescriptions, ...
    'AcquisitionFunctionName', AFName,...
    'NumGridDivisions', HyperparameterOptimizationOptions.NumGridDivisions,...
    'FitModels', false,...
    'MaxObjectiveEvaluations', HyperparameterOptimizationOptions.MaxObjectiveEvaluations, ...
    'MaxTime', HyperparameterOptimizationOptions.MaxTime, ...
    'ConditionalVariableFcn', BOInfo.ConditionalVariableFcn, ...
    'XConstraintFcn', BOInfo.XConstraintFcn, ...
    'Verbose', HyperparameterOptimizationOptions.Verbose, ...
    'UseParallel', HyperparameterOptimizationOptions.UseParallel, ...
    'PlotFcn', PlotFcn,...
    'OutputFcn', [],...
    'AlwaysReportObjectiveErrors', true,...
    'IsClassregRegressionFunction', BOInfo.IsRegression);
% Choose best point
XBest = chooseBestPointNonBayesopt(BOResults);
% Build results table
OptimizationResults = BOResults.XTrace;
OptimizationResults.Objective = BOResults.ObjectiveTrace;
OptimizationResults.Rank = rankVector(BOResults.ObjectiveTrace);
end

function R = rankVector(V)
R = zeros(size(V));
[~,I] = sort(V);
R(I) = 1:numel(V);
end

function BestXTable = chooseBestPointBayesopt(BO)
% Try default method. If that fails, return best observed point.
BestXTable = bestPoint(BO);
if isempty(BestXTable)
    BestXTable = bestPoint(BO, 'Criterion','minobserved');
end
end

function XBest = chooseBestPointNonBayesopt(BO)
if isfinite(BO.MinObjective)
    XBest = BO.XAtMinObjective;
else
    XBest = [];
end
end

function verifyNoValidationArgs(Args)
if classreg.learning.paramoptim.anyArgPassed({'CrossVal', 'CVPartition', 'Holdout', 'KFold', 'Leaveout'}, Args)
    classreg.learning.paramoptim.err('ValidationArgLocation');
end
end

function [ValidationMethod, ValidationVal] = getPassedValidationArgs(ParamOptimOptions)
% Assumes that exactly one validation field is nonempty.
if ~isempty(ParamOptimOptions.KFold)
    ValidationMethod	= 'KFold';
    ValidationVal       = ParamOptimOptions.KFold;
elseif ~isempty(ParamOptimOptions.Holdout)
    ValidationMethod	= 'Holdout';
    ValidationVal       = ParamOptimOptions.Holdout;
elseif ~isempty(ParamOptimOptions.CVPartition)
    ValidationMethod	= 'CVPartition';
    ValidationVal       = ParamOptimOptions.CVPartition;
end
end

function iVerifyValidOptimizationProblem(BOInfo)
% Check if we're optimizing fitcnb with all categorical predictors
if isequal(BOInfo.FitFcn, @fitcnb) && BOInfo.AllCategoricalPredictors
    classreg.learning.paramoptim.err('FitcnbAllCategoricalPredictors');
end
% Check if we're optimizing fitcsvm with one-class learning
if isequal(BOInfo.FitFcn, @fitcsvm) && BOInfo.OneClassLearning
    classreg.learning.paramoptim.err('FitcsvmOneClassLearning');
end
end
