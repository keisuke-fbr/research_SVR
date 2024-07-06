function [OptimizeHyperparameters, HyperparameterOptimizationOptions, RemainingArgs] = ...
    parseFitoptimizingArgs2(Args, IsTallData, NumObservations,IsAutoMLFcn)
% Parse the two parameters 'OptimizeHyperparameters' and
% 'HyperparameterOptimizationOptions'. Fill options defaults.
    
%   Copyright 2016-2021 The MathWorks, Inc.

[OptimizeHyperparameters, Opts, ~, RemainingArgs] = internal.stats.parseArgs(...
    {'OptimizeHyperparameters', 'HyperparameterOptimizationOptions'}, ...
    {'auto', []}, ...
    Args{:});
HyperparameterOptimizationOptions = validateAndFillParameterOptimizationOptions(Opts, IsTallData, NumObservations,IsAutoMLFcn);
end

function Opts = validateAndFillParameterOptimizationOptions(Opts, IsTallData, NumObservations,IsAutoMLFcn)
if isempty(Opts)
    Opts = struct;
elseif ~isstruct(Opts)
    classreg.learning.paramoptim.err('OptimOptionsNotStruct');
end
Opts = charifyStrings(Opts);
Opts = validateAndCompleteStructFields(Opts, {'Optimizer', 'MaxObjectiveEvaluations', 'MaxTime',...
    'AcquisitionFunctionName', 'NumGridDivisions', 'ShowPlots', 'SaveIntermediateResults', 'Verbose', ...
    'CVPartition', 'Holdout', 'KFold', 'Repartition', 'UseParallel',...
    'MinTrainingSetSize', 'MaxTrainingSetSize', 'TrainingSetSizeMultiplier', 'NumTrainingSetSizes'});
% Validate and fill Optimizer:
if ~isempty(Opts.Optimizer)
    Opts.Optimizer = validateOptimizer(Opts.Optimizer,IsAutoMLFcn);
else
    Opts.Optimizer = 'bayesopt';
end

% Validate and fill MaxTime:
if ~isempty(Opts.MaxTime)
    validateMaxTime(Opts.MaxTime);
else
    Opts.MaxTime = Inf;
end
% Validate and fill AcquisitionFunctionName:
if ~isempty(Opts.AcquisitionFunctionName)
    validateAcquisitionFunctionName(Opts.AcquisitionFunctionName);
else
    Opts.AcquisitionFunctionName = 'expected-improvement-per-second-plus';
end
if isequal(Opts.Optimizer,'asha')
    Opts.AcquisitionFunctionName = 'asha';
end
% Validate and fill NumGridDivisions:
if ~isempty(Opts.NumGridDivisions)
    validateNumGrid(Opts.NumGridDivisions);
else
    Opts.NumGridDivisions = 10;
end

% Validate and fill ShowPlots:
if ~isempty(Opts.ShowPlots)
    validateShowPlots(Opts.ShowPlots);
else
    Opts.ShowPlots = true;
end
% Validate and fill SaveIntermediateResults:
if ~isempty(Opts.SaveIntermediateResults)
    validateSaveIntermediateResults(Opts.SaveIntermediateResults);
else
    Opts.SaveIntermediateResults = false;
end
% Validate and fill Verbose:
if ~isempty(Opts.Verbose)
    validateVerbose(Opts.Verbose);
else
    Opts.Verbose = 1;
end
% Validate and fill UseParallel. THIS MUST PRECEDE validateAndFillValidationOptions:
if ~isempty(Opts.UseParallel)
    validateUseParallel(Opts.UseParallel);
else
    Opts.UseParallel = false;
end
% Validate and fill validation options. THIS MUST FOLLOW validateUseParallel
Opts = validateAndFillValidationOptions(Opts, IsTallData);

% Validate ASHA parameters are empty when optimizer is set to bayesopt
if isequal(Opts.Optimizer, 'bayesopt') && (~isempty(Opts.NumTrainingSetSizes) || ~isempty(Opts.TrainingSetSizeMultiplier)...
        || ~isempty(Opts.MinTrainingSetSize) || ~isempty(Opts.MaxTrainingSetSize))
    classreg.learning.paramoptim.err('BayesoptSetIncorrectly');    
end

% Validate and fill ASHA args:
if isequal(Opts.Optimizer, 'asha')
    if ~isempty(Opts.TrainingSetSizeMultiplier)
        validateTrainingSetSizeMultiplier(Opts.TrainingSetSizeMultiplier);
    else
        Opts.TrainingSetSizeMultiplier = 4;
    end
    if ~isempty(Opts.NumTrainingSetSizes)
        validateNumTrainingSetSizes(Opts.NumTrainingSetSizes);
    else
        Opts.NumTrainingSetSizes = 5;
    end
    Opts.MaxTrainingSetSize = validateAndFillMaxTrainingSetSize(Opts, NumObservations);
    [Opts.MinTrainingSetSize, Opts.NumTrainingSetSizes] = validateAndFillMinTrainingSetSize(Opts);
end

% Validate and fill MaxObjectiveEvaluations:
if ~isempty(Opts.MaxObjectiveEvaluations)
    validateMaxFEvals(Opts.MaxObjectiveEvaluations);    
else
    Opts.MaxObjectiveEvaluations = [];      % use bayesopt default.
end

end

function Opts = charifyStrings(Opts)
% Make all string values char arrays
fields = fieldnames(Opts);
values = struct2cell(Opts);
for i=1:numel(values)
    if isstring(values{i})
        Opts.(fields{i}) = char(values{i});
    end
end
end

function [NewMinTrainingSetSize, NewNumTrainingSetSizes] = validateAndFillMinTrainingSetSize(Opts)
% Calculate the min size from the max size, the multiplier and the number
% of rungs. If the result is smaller than MinTrainingSetSize, reduce the
% number of rungs until it is not.
% Opts.MaxTrainingSetSize must be set to its final value before calling this.
NewNumTrainingSetSizes = Opts.NumTrainingSetSizes;
if ~isempty(Opts.MinTrainingSetSize) 
    validateattributes(Opts.MinTrainingSetSize,{'numeric'},{'scalar' 'integer' 'positive'}, mfilename,'MinTrainingSetSize');
    Opts.MinTrainingSetSize = double(Opts.MinTrainingSetSize);
end
if isempty(Opts.MinTrainingSetSize)
    lowerBound = 100;
else
    lowerBound = Opts.MinTrainingSetSize;
    if Opts.MinTrainingSetSize <100        
        classreg.learning.paramoptim.warn('MinTrainingSetSize100')
    end
    if Opts.MinTrainingSetSize >= Opts.MaxTrainingSetSize
        classreg.learning.paramoptim.err('MaxTrainMinTrainValueMismatch',Opts.MaxTrainingSetSize);        
    end
end
calculatedMin = ceil(Opts.MaxTrainingSetSize * Opts.TrainingSetSizeMultiplier^(1-Opts.NumTrainingSetSizes));
if calculatedMin < lowerBound
    if Opts.MaxTrainingSetSize < lowerBound
        NewNumTrainingSetSizes = 1; %dataset too small to create multiple training sets
    else
        NewNumTrainingSetSizes = floor(1 - log(lowerBound/Opts.MaxTrainingSetSize)/log(Opts.TrainingSetSizeMultiplier));
    end
    if NewNumTrainingSetSizes == 1 % random search instead of ASHA
        classreg.learning.paramoptim.warn('ASHASwitchToRandomSearch')        
    end
end
NewMinTrainingSetSize = ceil(Opts.MaxTrainingSetSize * Opts.TrainingSetSizeMultiplier^(1-NewNumTrainingSetSizes));
end

function val = validateAndFillMaxTrainingSetSize(Opts, NumObservations)
% Default max training set size is equal to the largest training
% partition given the selected validation method.
% Assumes that exactly one validation field is nonempty.
if ~isempty(Opts.MaxTrainingSetSize) 
    validateattributes(Opts.MaxTrainingSetSize,{'numeric'},{'scalar' 'integer' 'positive'}, mfilename,'MaxTrainingSetSize');
    Opts.MaxTrainingSetSize = double(Opts.MaxTrainingSetSize);
end
if ~isempty(Opts.KFold)
    defaultMax = ceil(NumObservations*(Opts.KFold-1)/Opts.KFold);
elseif ~isempty(Opts.Holdout)
    defaultMax = ceil(NumObservations*(1-Opts.Holdout));
elseif ~isempty(Opts.CVPartition)
    defaultMax = max(Opts.CVPartition.TrainSize);
end
val = Opts.MaxTrainingSetSize;
if isempty(val) || val>defaultMax
    val = defaultMax;
end
end

function validateTrainingSetSizeMultiplier(TrainingSetSizeMultiplier)
    if TrainingSetSizeMultiplier <=1        
        classreg.learning.paramoptim.err('IncorrectTrainingSetSizeMultiplier');
    end
end
function validateNumTrainingSetSizes(NumTrainingSetSize)
    validateattributes(NumTrainingSetSize,{'numeric'},{'scalar' 'integer' 'positive'},'fitrauto','NumTrainingSetSizes');
end
            
function Optimizer = validateOptimizer(Optimizer,IsAutoMLFcn)
    if IsAutoMLFcn
        Optimizer = validatestring(Optimizer,{'bayesopt','asha'},mfilename,'optimizer');
    else
        Optimizer = validatestring(Optimizer,{'bayesopt','gridsearch','randomsearch'},mfilename,'optimizer');
    end
end        

function validateMaxFEvals(MaxObjectiveEvaluations)
if ~bayesoptim.isNonnegativeInteger(MaxObjectiveEvaluations)
    classreg.learning.paramoptim.err('MaxObjectiveEvaluations');
end
end

function validateMaxTime(MaxTime)
if ~bayesoptim.isNonnegativeRealScalar(MaxTime)
    classreg.learning.paramoptim.err('MaxTime');
end
end

function validateAcquisitionFunctionName(AcquisitionFunctionName)
RepairedString = bayesoptim.parseArgValue(AcquisitionFunctionName, {...
    'expectedimprovement', ...
    'expectedimprovementplus', ...
    'expectedimprovementpersecond',...
    'expectedimprovementpersecondplus',...
    'lowerconfidencebound',...
    'probabilityofimprovement',...
    'asha'});
if isempty(RepairedString)
    classreg.learning.paramoptim.err('AcquisitionFunctionName');
end
end

function validateNumGrid(NumGridDivisions)
if ~all(arrayfun(@(x)bayesoptim.isLowerBoundedIntScalar(x,2), NumGridDivisions))
    classreg.learning.paramoptim.err('NumGridDivisions');
end
end

function validateShowPlots(ShowPlots)
if ~bayesoptim.isLogicalScalar(ShowPlots)
    classreg.learning.paramoptim.err('ShowPlots');
end
end

function validateUseParallel(UseParallel)
if ~bayesoptim.isLogicalScalar(UseParallel)
    classreg.learning.paramoptim.err('UseParallel');
end
end

function validateSaveIntermediateResults(SaveIntermediateResults)
if ~bayesoptim.isLogicalScalar(SaveIntermediateResults)
    classreg.learning.paramoptim.err('SaveIntermediateResultsType');
end
end

function validateVerbose(Verbose)
if ~(bayesoptim.isAllFiniteReal(Verbose) && ismember(Verbose, [0,1,2]))
    classreg.learning.paramoptim.err('Verbose');
end
end

function validateRepartition(Repartition, Options)
% Assumes UseParallel has been filled by this point.
if ~bayesoptim.isLogicalScalar(Repartition)
    classreg.learning.paramoptim.err('RepartitionType');
end
if Repartition && ~isempty(Options.CVPartition)
    classreg.learning.paramoptim.err('RepartitionCondition');
end
end

function Options = validateAndFillValidationOptions(Options, IsTallData)
% Assumes UseParallel has been filled by this point.
NumPassed = ~isempty(Options.CVPartition) + ~isempty(Options.Holdout) + ~isempty(Options.KFold);
if NumPassed > 1
    classreg.learning.paramoptim.err('MultipleValidationArgs');
elseif NumPassed == 0
    Options = chooseDefaultValidation(Options, IsTallData);
elseif ~isempty(Options.CVPartition)
    if ~isa(Options.CVPartition, 'cvpartition')
        classreg.learning.paramoptim.err('CVPartitionType');
    end
    if isequal(Options.CVPartition.Type, 'kfold') && IsTallData
        classreg.learning.paramoptim.err('KFoldNotSupportedForTall');
    end
    if isequal(Options.CVPartition.Type, 'leaveout') && IsTallData
        classreg.learning.paramoptim.err('LeaveoutNotSupportedForTall');
    end
elseif ~isempty(Options.Holdout)
    v = Options.Holdout;
    if ~(bayesoptim.isAllFiniteReal(v) && v>0 && v<1)
        classreg.learning.paramoptim.err('Holdout');
    end
elseif ~isempty(Options.KFold)
    if IsTallData
        classreg.learning.paramoptim.err('KFoldNotSupportedForTall');
    end
    v = Options.KFold;
    if ~(bayesoptim.isLowerBoundedIntScalar(v,2))
        classreg.learning.paramoptim.err('KFold');
    end
end
% Repartition
if ~isempty(Options.Repartition)
    validateRepartition(Options.Repartition, Options);
else
    Options.Repartition = false;
end
end

function Options = chooseDefaultValidation(Options, IsTallData)
if IsTallData
    Options.Holdout = .2;
else
    Options.KFold = 5;
end
end

function S = validateAndCompleteStructFields(S, FieldNames)
% Make sure all fields of S are prefixes of the character vectors in
% FieldNames, and return a complete struct.
f = fieldnames(S);
ArgList = cell(1,2*numel(f));
ArgList(1:2:end) = f;
ArgList(2:2:end) = struct2cell(S);
Defaults = cell(1,numel(f));
[values{1:numel(FieldNames)}, ~, extra] = internal.stats.parseArgs(...
    FieldNames, Defaults, ArgList{:});
if ~isempty(extra)
    classreg.learning.paramoptim.err('BadStructField', extra{1});
end
StructArgs = cell(1,2*numel(FieldNames));
StructArgs(1:2:end) = FieldNames;
StructArgs(2:2:end) = values;
S = struct(StructArgs{:});
end


