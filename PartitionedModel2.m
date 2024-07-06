classdef PartitionedModel2 < classreg.learning.internal.DisallowVectorOps
%PartitionedModel Cross-validated model.
%   PartitionedModel is the super class for cross-validated models.
        
%   Copyright 2010-2020 The MathWorks, Inc.


    properties(GetAccess=public,SetAccess=protected,Dependent=true)
        %CROSSVALIDATEDMODEL Name of the cross-validated model.
        %   The CrossValidatedModel is a string with the name of the
        %   cross-validated model, for example, 'Tree' for a cross-validated
        %   decision tree.
        %
        %   See also classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
        CrossValidatedModel;
        
        %PREDICTORNAMES Names of predictors used for this model.
        %   The PredictorNames is a cell array of strings with names of predictor
        %   variables, one name per column of X.
        %
        %   See also classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
        PredictorNames;
        
        %CATEGORICALPREDICTORS Indices of categorical predictors.
        %   The CategoricalPredictors property is an array with indices of
        %   categorical predictors. The indices are in the range from 1 to the
        %   number of columns in X.
        %
        %   See also classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
        CategoricalPredictors;
        
        %RESPONSENAME Name of the response variable.
        %   The ResponseName is a string with the name of the response variable Y.
        %
        %   See also classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
        ResponseName;
        
        %NUMOBSERVATIONS Number of observations.
        %   The NumObservations property is a numeric positive scalar showing the
        %   number of observations in the training data.
        %
        %   See also classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
        NumObservations;

        %X X data used to train this model.
        %   The X property contains the predictor values. It is a table for a model
        %   trained on a table, or a numeric matrix for a model trained on a
        %   matrix. It has size N-by-P, where N is the number of rows and P is the
        %   number of predictor variables or columns in the training data.
        %
        %   See also classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
        X;
        
        %Y Y data used to cross-validate this model.
        %   The Y property is an array of true class labels for classification, or
        %   response values for regression. For classification, Y is of the same
        %   type as the passed-in Y data: a cell array of strings, categorical,
        %   logical, numeric or a character matrix. For regression, Y is a numeric
        %   vector.
        %
        %   See also classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
        Y;       
        
        %W Weights of observations used to cross-validate this model.
        %   The W property is a numeric vector of size N, where N is the number of
        %   observations.
        %
        %   See also classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
        W;
        
        %MODELPARAMETERS Cross-validation parameters.
        %   The ModelParameters property holds parameters used for cross-validating
        %   this model.
        %
        %   See also classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
        ModelParameters;
        
        %TRAINED Compact models trained on cross-validation folds.
        %   The Trained property is a cell array of models trained on
        %   cross-validation folds.
        %
        %   See also classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
        Trained;
        
        %KFOLD Number of cross-validation folds.
        %   The KFold property is a positive integer showing on how many folds this
        %   model has been cross-validated.
        %
        %   See also classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
        KFold;
        
        %PARTITION Data partition used to cross-validate this model.
        %   The Partition property is an object of type cvpartition specifying how
        %   the data are split into cross-validation folds.
        %
        %   See also classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
        Partition;
        
        %BINEDGES Bin edges for predictors.
        %   The BinEdges property is a cell array of P numeric vectors with bin
        %   edges for P predictors. This property is filled only if the 'NumBins'
        %   parameter is passed to the fit function; otherwise this property is
        %   empty. For categorical predictors, elements of the cell array are
        %   empty.
        %
        %   See also PredictorNames,
        %   classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
        BinEdges
    end
    
    properties(GetAccess=public,SetAccess=protected,Hidden=true,Dependent=true)
        NObservations;
        ModelParams;
    end
    
    properties(GetAccess=public,SetAccess=protected,Hidden=true)
        Ensemble;
    end
    
    methods
        function cvmodel = get.CrossValidatedModel(this)
            cvmodel = '';
            if numel(this.Ensemble.ModelParams.LearnerTemplates)==1
                cvmodel = this.Ensemble.ModelParams.LearnerTemplates{1}.Method;
            end
        end
        
        function predictornames = get.PredictorNames(this)
            predictornames = this.Ensemble.PredictorNames;
        end
        
        function catpreds = get.CategoricalPredictors(this)
            catpreds = this.Ensemble.CategoricalPredictors;
        end
        
        function respname = get.ResponseName(this)
            respname = this.Ensemble.ResponseName;
        end
        
        function n = get.NumObservations(this)
            if this.Ensemble.ObservationsInRows
                n = size(this.Ensemble.X,1);
            else
                n = size(this.Ensemble.X,2);
            end
        end
        
        function n = get.NObservations(this)
            if this.Ensemble.ObservationsInRows
                n = size(this.Ensemble.X,1);
            else
                n = size(this.Ensemble.X,2);
            end
        end

        function x = get.X(this)
            x = this.Ensemble.X;
        end
        
        function y = get.Y(this)
            y = this.Ensemble.Y;
        end
        
        function w = get.W(this)
            w = this.Ensemble.W;
        end

        function mp = get.ModelParameters(this)
            mp = this.Ensemble.ModelParameters;
        end
        
        function mp = get.ModelParams(this)
            mp = this.Ensemble.ModelParams;
        end
        
        function trained = get.Trained(this)
            trained = this.Ensemble.Trained;
        end
        
        function ntrained = get.KFold(this)
            ntrained = this.Ensemble.NTrained;
        end
        
        function p = get.Partition(this)
            p = this.ModelParams.Generator.Partition;
        end
        
        function edges = get.BinEdges(this)
            try
                edges = this.Ensemble.DataSummary.BinEdges;
            catch
                edges = {}; % may be empty for an old object
            end
        end
    end
    
    methods(Access=protected)
        function this = PartitionedModel2()
            this = this@classreg.learning.internal.DisallowVectorOps();
        end
        
        function s = propsForDisp(this,s)
            if nargin<2 || isempty(s)
                s = struct;
            else
                if ~isstruct(s)
                    error(message('stats:classreg:learning:partition:PartitionedModel:propsForDisp:BadS'));
                end
            end
            s.CrossValidatedModel   = this.CrossValidatedModel;
            s.PredictorNames        = this.PredictorNames;
            if ~isempty(this.CategoricalPredictors)
                s.CategoricalPredictors = this.CategoricalPredictors;
            end
            s.ResponseName          = this.ResponseName;
            if this.Ensemble.ObservationsInRows
                s.NumObservations       = size(this.X,1);
            else
                s.NumObservations       = size(this.X,2);
            end
            s.KFold                 = this.KFold;
            s.Partition             = this.Partition;
        end
    end
    
    methods                
        function [varargout] = kfoldPredict(this,varargin)
            [varargin{:}] = convertStringsToChars(varargin{:});
            Ensemble2.catchUOFL(varargin{:});
            PartitionedModel2.catchFolds(varargin{:});
            FullClassificationRegressionModel2.catchWeights(varargin{:});
            [mode,~,args] = checkFoldArgs(this,varargin{:});
            [varargout{1:nargout}] = predict(this.Ensemble,this.Ensemble.X,...
                'useobsforlearner',~this.Ensemble.ModelParams.Generator.UseObsForIter,...
                'mode',mode,args{:});
        end
        
        function vals = kfoldfun(this,funeval)
        %KFOLDFUN Cross-validate function.
        %   KFOLDVALS=KFOLDFUN(OBJ,FUN) cross-validates function FUN by applying to
        %   the data stored in the cross-validated model OBJ. You must pass FUN as
        %   a function handle. FUN is called on every fold as follows:
        %
        %   TESTVALS = FUN(CMP,XTRAIN,YTRAIN,WTRAIN,XTEST,YTEST,WTEST)
        %
        %   Above, CMP is a compact model stored in one element of the Trained
        %   property, XTRAIN is the training matrix of predictor values, YTRAIN is
        %   the training array of response values, WTRAIN are weights for the
        %   training observations, and XTEST, YTEST, and WTEST are test data. The
        %   size of TESTVALS must be the same across all folds. 
        %
        %   KFOLDFUN concatenates arrays of TESTVALS from all folds vertically and
        %   returns them as KFOLDVALS. For example, if TESTVALS from every fold is
        %   a numeric vector of length N, KFOLDFUN returns a KFold-by-N numeric
        %   matrix with one row per fold.
        %
        %   Example:
        %   % Cross-validate a regression tree and obtain cross-validated MSE
        %   load imports-85
        %   t = fitrtree(X(:,[4 5]),X(:,16),...
        %       'predictornames',{'length' 'width'},'responsename','price')
        %   cv = crossval(t)
        %   kfoldLoss(cv)
        %   % What accuracy do we get by simple averaging?
        %   f = @(cmp,Xtrain,Ytrain,Wtrain,Xtest,Ytest,Wtest) mean((Ytest-mean(Ytrain)).^2)
        %   mean(kfoldfun(cv,f))
        %
        %   See also classreg.learning.partition.ClassificationPartitionedModel,
        %   classreg.learning.partition.RegressionPartitionedModel.
            
            % Init and get learners from folds
            vals = [];
            T = this.Ensemble.NTrained;
            if T<1
                return;
            end
            trained = this.Ensemble.Trained;
            usenfort = this.Ensemble.ModelParams.Generator.UseObsForIter;
            x = this.Ensemble.X;
            y = this.Ensemble.Y;
            w = this.Ensemble.W;
            
            % Loop over the folds
            vals = zeros(T,1);
            for t=1:T
                use = usenfort(:,t);
                cmp = trained{t};
                cmp.TableInput = this.Ensemble.TableInput;
                cmp.VariableRange = this.Ensemble.VariableRange;
                valt = funeval(cmp,x(use,:),y(use,:),w(use),x(~use,:),y(~use,:),w(~use));
                if t==1
                    vals(1,1:numel(valt)) = valt;
                else
                    if numel(valt)~=size(vals,2)
                        error(message('stats:classreg:learning:partition:PartitionedModel:kfoldfun:BadDimsPerFold', numel( valt ), t, size( vals, 2 )));
                    end
                    vals(t,:) = valt;
                end
            end
        end
    end

    
    methods(Hidden)
        function this = compactPartitionedModel(this)
            this.Ensemble = compact(this.Ensemble);
        end
        
        function disp(this)
            internal.stats.displayClassName(this);
            
            % Display body
            s = propsForDisp(this,[]);
            disp(s);
            
            internal.stats.displayMethodsProperties(this);
        end

        function [ensembleMode,folds,extraArgs] = checkFoldArgs(this,varargin)
            kfold = length(this.Trained);
            
            args = {'mode'     'folds' 'learners'};
            defs = {'average' 1:kfold          []};
            [cvmode,folds,learners,~,extraArgs] = ...
                internal.stats.parseArgs(args,defs,varargin{:});
            
            if ~isempty(learners)
                error(message('stats:classreg:learning:partition:PartitionedModel:checkFoldArgs:LearnersNoop'));
            end
            
            cvmode = lower(cvmode);
            if     strncmpi(cvmode,'average',length(cvmode))
                ensembleMode = 'ensemble';
            elseif strncmpi(cvmode,'individual',length(cvmode))
                ensembleMode = 'individual';
            else
                error(message('stats:classreg:learning:partition:PartitionedModel:checkFoldArgs:BadMode'));
            end
            
            if islogical(folds)
                if ~isvector(folds) || length(folds)~=kfold
                    error(message('stats:classreg:learning:partition:PartitionedModel:checkFoldArgs:BadLogicalIndices', kfold));
                end
                folds = find(folds);
            end
            if isempty(folds) || ~isnumeric(folds) || ~isvector(folds) || min(folds)<=0 || max(folds)>kfold
                error(message('stats:classreg:learning:partition:PartitionedModel:checkFoldArgs:BadNumericIndices', kfold));
            end
            folds = ceil(folds);
        end
    end
    
    methods(Static,Hidden)
        function catchFolds(varargin)
            args = {'folds'};
            defs = {     []};
            [folds,~,~] = internal.stats.parseArgs(args,defs,varargin{:});
            if ~isempty(folds)
                error(message('stats:classreg:learning:partition:PartitionedModel:catchFolds:NonEmptyFolds'));
            end
        end
    end
    
end
