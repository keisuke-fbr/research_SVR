function objFcn = createObjFcn2(BOInfo, FitFunctionArgs, Predictors, Response, ...
    ValidationMethod, ValidationVal, Repartition, Verbose, Optimizer)
% Create and return the objective function. If 'Repartition' is false and
% no cvpartition is passed, we first create a cvpartition to be used in all
% function evaluations. The cvp is stored in the workspace of the function
% handle and can be accessed later from the function handle like this:
% f=functions(h);cvp=f.workspace{1}.cvp

%   Copyright 2016-2021 The MathWorks, Inc.

% Set validation value
if ~Repartition && ~isa(ValidationVal, 'cvpartition')
    ValidationVal    = createStaticCVP(BOInfo, Predictors, Response, FitFunctionArgs, ValidationMethod, ValidationVal);
    ValidationMethod = 'CVPartition';
end

ClassSummary = [];
classSummaryIdx = find(strcmp(FitFunctionArgs,'ClassSummary'));

if ~isempty(classSummaryIdx)
    ClassSummary = FitFunctionArgs{classSummaryIdx+1};
    FitFunctionArgs(classSummaryIdx:classSummaryIdx+1) = [];
end

isFitcauto = find(strcmp(FitFunctionArgs,'isFitcauto'), 1);        
if ~isempty(isFitcauto)
    FitFunctionArgs(isFitcauto) = [];
    isFitcauto = true;
else
    isFitcauto = false;
end
isUsingCosts = any(strcmpi('cost',FitFunctionArgs));

% Choose objfcn
if isequal(Optimizer, 'asha')
    objFcn = @ashaObjFcn;
elseif istall(Predictors)
    objFcn = @tallObjFcn;
else
    objFcn = @inMemoryObjFcn;
end

    function Objective = inMemoryObjFcn(XTable)
        % (1) Set up args
        NewFitFunctionArgs = updateArgsFromTable(BOInfo, FitFunctionArgs, XTable);
        % (2) Call fit fcn, suppressing specific warnings
        C = classreg.learning.paramoptim.suppressWarnings();
        PartitionedModel = BOInfo.FitFcn(Predictors, Response, ValidationMethod, ValidationVal, NewFitFunctionArgs{:});
        % (3) Compute kfoldLoss if possible
        if PartitionedModel.KFold == 0
            Objective = NaN;
            if Verbose >= 2
                classreg.learning.paramoptim.printInfo('ZeroFolds');
            end
        else
            if BOInfo.IsRegression
                Objective = log1p(kfoldLoss(PartitionedModel));
            elseif isFitcauto && isUsingCosts
                Objective = computeCost(PartitionedModel,ClassSummary);
            else
                Objective = kfoldLoss(PartitionedModel);
            end
            if ~isscalar(Objective)
                % For cases like fitclinear where the user passes Lambda as a vector.
                Objective = Objective(1);
                if Verbose >= 2
                    classreg.learning.paramoptim.printInfo('ObjArray');
                end
            end
        end
    end

    function Objective = ashaObjFcn(XTable)
        % The asha objective function is in-memory only. It always uses the
        % full test data, but it trains on a different random, stratified
        % subset of the training data each time it is called.
        
        % (1) Remove the ASHA variables from XTable
        TrainingSetSize = XTable.TrainingSetSize;
        XTable.TrainingSetSize = [];
        XTable.ASHARung            = [];
        % (2) Set up the usual fit fcn args
        NewFitFunctionArgs = updateArgsFromTable(BOInfo, FitFunctionArgs, XTable);
        % (3) Append the undocumented asha NVPs
        NewFitFunctionArgs = [NewFitFunctionArgs {'TrainingSubsampleSizeCV', TrainingSetSize, 'TrainingSubsampleStratifyCV', ~BOInfo.IsRegression}];
        % (4) Call fit fcn, suppressing specific warnings
        C = classreg.learning.paramoptim.suppressWarnings();
        PartitionedModel = BOInfo.FitFcn(Predictors, Response, ValidationMethod, ValidationVal, NewFitFunctionArgs{:});
        % (5) Compute kfoldLoss if possible
        if PartitionedModel.KFold == 0
            Objective = NaN;
            if Verbose >= 2
                classreg.learning.paramoptim.printInfo('ZeroFolds');
            end
        else
            if BOInfo.IsRegression
                Objective = log1p(kfoldLoss(PartitionedModel));
            elseif isFitcauto && isUsingCosts
                Objective = computeCost(PartitionedModel,ClassSummary);
            else
                Objective = kfoldLoss(PartitionedModel);
            end
            if ~isscalar(Objective)
                % For cases like fitclinear where the user passes Lambda as a vector.
                Objective = Objective(1);
                if Verbose >= 2
                    classreg.learning.paramoptim.printInfo('ObjArray');
                end
            end
        end
    end

    function Objective = tallObjFcn(XTable)
        % (1) Set up fit fcn args
        NewFitFunctionArgs = updateArgsFromTable(BOInfo, FitFunctionArgs, XTable);
        % (2) Set up validation
        if Repartition
            cvp = cvpartition(Predictors(:,1), 'Holdout', ValidationVal, 'Stratify', false);
        else
            cvp = ValidationVal;
        end
        % (3) Split weight arg if present
        [TrainingWeightArgs, TestWeightArgs] = splitWeightArgs(NewFitFunctionArgs, cvp);
        % (4) Call fit fcn on training set, suppressing specific warnings
        C = classreg.learning.paramoptim.suppressWarnings();
        if istall(Response)
            TrainResp = Response(cvp.training);
            TestResp = Response(cvp.test);
        else
            TrainResp = Response;
            TestResp = Response;
        end
        try
            Model = BOInfo.FitFcn(Predictors(cvp.training,:), TrainResp, NewFitFunctionArgs{:}, TrainingWeightArgs{:});
            % (5) Compute validation loss
            if BOInfo.IsRegression
                Objective = gather(log1p(loss(Model, Predictors(cvp.test,:), TestResp, TestWeightArgs{:})));
            else
                Objective = gather(loss(Model, Predictors(cvp.test,:), TestResp, TestWeightArgs{:}));
            end
            if ~isscalar(Objective)
                % For cases like fitclinear where the user passes Lambda as a vector.
                Objective = Objective(1);
                if Verbose >= 2
                    classreg.learning.paramoptim.printInfo('ObjArray');
                end
            end
        catch msg
            disp(msg.message);
            % Return NaN for MATLAB errors
            Objective = NaN;
        end
    end
end

function [TrainingWeightArgs, TestWeightArgs] = splitWeightArgs(NVPs, cvp)
% If the 'Weights' NVP is present, split it into training and test, and
% return two cell arrays, each containing a NVP.
[WeightsFound, W] = classreg.learning.paramoptim.parseWeightArg(NVPs);
if ~WeightsFound
    TrainingWeightArgs = {};
    TestWeightArgs = {};
elseif istall(W)
    TrainingWeightArgs = {'Weights',W(cvp.training)};
    TestWeightArgs = {'Weights',W(cvp.test)};
else
    TrainingWeightArgs = {'Weights',W};
    TestWeightArgs = {'Weights',W};
end
end

function cvp = createStaticCVP(BOInfo, Predictors, Response, FitFunctionArgs, ValidationMethod, ValidationVal)
if istall(Predictors)
    assert(isequal(lower(ValidationMethod), 'holdout'));
    cvp = cvpartition(Predictors(:,1), 'Holdout', ValidationVal, 'Stratify', false);
else
    [~,PrunedY] = BOInfo.PrepareDataFcn(Predictors, Response, FitFunctionArgs{:}, 'IgnoreExtraParameters', true);
    if BOInfo.IsRegression
        cvp = cvpartition(numel(PrunedY), ValidationMethod, ValidationVal);
    else
        cvp = cvpartition(PrunedY, ValidationMethod, ValidationVal);
    end
end
end

function output = computeCost(partitonedModel,ClassSummary)
    classnames = ClassSummary.NonzeroProbClasses;
    
    cost = ClassSummary.Cost;
    if isempty(cost)
        numClasses = length(ClassSummary.NonzeroProbClasses);
        cost = ones(numClasses) - eye(numClasses);
    end
    y = classreg.learning.internal.ClassLabel(partitonedModel.Y);
    yhat = classreg.learning.internal.ClassLabel(kfoldPredict(partitonedModel));
    C = membership(y,classnames);
    Chat = membership(yhat,classnames);
    [~,y] = max(C,[],2);
    [~,yhat] = max(Chat,[],2);
    e = cost(sub2ind(size(cost),y,yhat));
    output = sum(e)/length(yhat);
end