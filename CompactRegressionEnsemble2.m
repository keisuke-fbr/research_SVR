classdef CompactRegressionEnsemble2 < ...
        RegressionModel2 & CompactEnsemble2
%CompactRegressionEnsemble Compact regression ensemble.
%   CompactRegressionEnsemble is a set of trained weak learner models.
%   It can predict ensemble response for new data by aggregating
%   predictions from its weak learners.
%
%   CompactRegressionEnsemble properties:
%       PredictorNames        - Names of predictors used for this ensemble.
%       ExpandedPredictorNames - Names of expanded predictors used for this tree.
%       CategoricalPredictors - Indices of categorical predictors.
%       ResponseName          - Name of the response variable.
%       ResponseTransform     - Transformation applied to predicted regression response.
%       NumTrained            - Number of trained learners in the ensemble.
%       Trained               - Trained learners.
%       TrainedWeights        - Learner weights.
%       CombineWeights        - Prescription for combining weighted learner predictions.
%       UsePredForLearner     - Use predictors for learners.
%
%   CompactRegressionEnsemble methods:
%       loss                  - Regression loss.
%       predict               - Predicted response of this model.
%       predictorImportance   - Importance of predictors for this model.
%       removeLearners        - Remove learners from this ensemble.
%
%   See also RegressionEnsemble.

%   Copyright 2010-2020 The MathWorks, Inc.


    methods(Access=protected)
        function this = CompactRegressionEnsemble2(dataSummary,responseTransform,usepredforlearner)
            this = this@RegressionModel2(dataSummary,responseTransform);
            this = this@CompactEnsemble2(usepredforlearner);
        end
        
        function r = response(this,X,varargin)
            if isfield(this.DataSummary,'ObservationsInRows') && ~this.DataSummary.ObservationsInRows
                X = X';
            end
            r = classreg.learning.ensemble.CompactEnsemble.aggregatePredict(...
                X,this.Impl.Combiner,this.Impl.Trained,[],[],NaN,...
                'usepredforlearner',this.UsePredForLearner,varargin{:});
        end
        
        function s = propsForDisp(this,s)
            s = propsForDisp@RegressionModel2(this,s);
            s = propsForDisp@CompactEnsemble2(this,s);
        end
    end
     
    % Concrete methods
    methods
        function yfit = predict(this,X,varargin)
        %PREDICT Predict response of the ensemble.
        %   YFIT=PREDICT(ENS,X) returns predicted response YFIT for regression
        %   ensemble ENS and predictors X. X must be a table if ENS was
        %   originally trained on a table, or a numeric matrix if ENS was
        %   originally trained on a matrix. If X is a table, it must contain all
        %   the predictors used for training this model. If X is a matrix, it must
        %   have P columns, where P is the number of predictors used for training.
        %   YFIT is a vector of type double with size(X,1) elements.
        %
        %   YFIT=PREDICT(ENS,X,'PARAM1',val1,'PARAM2',val2,...) specifies
        %   optional parameter name/value pairs:
        %       'useobsforlearner' - Logical matrix of size N-by-NumTrained, where
        %                            N is the number of observations in X and
        %                            NumTrained is the number of weak learners.
        %                            This matrix specifies what learners in the
        %                            ensemble are used for what observations. By
        %                            default, all elements of this matrix are set
        %                            to true.
        %       'learners'         - Indices of weak learners in the ensemble
        %                            ranging from 1 to NumTrained. Only these
        %                            learners are used for making predictions. By
        %                            default, all learners are used.
        %
        %   See also RegressionEnsemble, CompactRegressionEnsemble.
            
            [varargin{:}] = convertStringsToChars(varargin{:});
            % Handle input data such as "tall" requiring a special adapter
            adapter = classreg.learning.internal.makeClassificationModelAdapter(this,X,varargin{:});
            if ~isempty(adapter)            
                yfit = predict(adapter,X,varargin{:});
                return
            end
        
            if isempty(X)
                if this.TableInput || istable(X)
                    vrange = getvrange(this);
                    X = classreg.learning.internal.table2PredictMatrix(X,[],[],...
                        vrange,...
                        this.CategoricalPredictors,this.PredictorNames);
                end
                % Check num of predictors in data
                classreg.learning.internal.numPredictorsCheck(X,...
                    getNumberPredictors(this),this.ObservationsInRows)

                yfit = predictEmptyX(this);
                return;
            end
            if isfield(this.DataSummary,'ObservationsInRows') && ...
                    ~this.DataSummary.ObservationsInRows
                varargin{end+1} = 'ObservationsIn';
                varargin{end+1} = 'Columns';
            end
            yfit = predict@RegressionModel2(this,X,varargin{:});
        end
        
        function l = loss(this,X,varargin)
        %LOSS Regression error.
        %   L=LOSS(ENS,X,Y) returns mean squared error for ensemble ENS computed
        %   using matrix of predictors X and observed response Y. X must be a table
        %   if ENS was originally trained on a table, or a numeric matrix if ENS
        %   was originally trained on a matrix. If X is a table, it must contain
        %   all the predictors used for training this model. If X is a matrix, it
        %   must have P columns, where P is the number of predictors used for
        %   training. Y must be a vector of floating-point numbers with N elements.
        %   Y can be omitted if X is a table that contains the response variable.
        %
        %   L=LOSS(ENS,X,Y,'PARAM1',val1,'PARAM2',val2,...) specifies optional
        %   parameter name/value pairs:
        %       'lossfun'          - Function handle for loss, or string
        %                            representing a built-in loss function.
        %                            Available loss functions for regression:
        %                            'mse'. If you pass a function handle FUN, LOSS
        %                            calls it as shown below:
        %                               FUN(Y,Yfit,W)
        %                            where Y, Yfit and W are numeric vectors of
        %                            length N. Y is observed response, Yfit is
        %                            predicted response, and W is observation
        %                            weights. Default: 'mse'
        %       'weights'          - Vector of observation weights. By default the
        %                            weight of every observation is set to 1. The
        %                            length of this vector must be equal to the
        %                            number of rows in X.
        %       'useobsforlearner' - Logical matrix of size N-by-NumTrained, where
        %                            N is the number of observations in X and
        %                            NumTrained is the number of weak learners.
        %                            This matrix specifies what learners in the
        %                            ensemble are used for what observations. By
        %                            default, all elements of this matrix are set
        %                            to true.
        %       'learners'         - Indices of weak learners in the ensemble
        %                            ranging from 1 to NumTrained. Only these
        %                            learners are used for making predictions. By
        %                            default, all learners are used.
        %       'mode'             - 'ensemble' (default), 'individual' or
        %                            'cumulative'. If 'ensemble', this method
        %                            returns a scalar value for the full ensemble.
        %                            If 'individual', this method returns a vector
        %                            with one element per trained learner. If
        %                            'cumulative', this method returns a vector in
        %                            which element J is obtained by using learners
        %                            1:J from the input list of learners.
        %
        %   See also RegressionEnsemble, CompactRegressionEnsemble, predict. 
        
            [varargin{:}] = convertStringsToChars(varargin{:});
            % Handle input data such as "tall" requiring a special adapter
            adapter = classreg.learning.internal.makeClassificationModelAdapter(this,X,varargin{:});
            if ~isempty(adapter)            
                l = loss(adapter,X,varargin{:});
                return
            end
            
            [Y,varargin] = classreg.learning.internal.inferResponse(this.ResponseName,X,varargin{:});
            N = size(X,1);
            args = {                  'lossfun' 'weights'};
            defs = {@classreg.learning.loss.mse ones(N,1)};
            [funloss,W,~,extraArgs] = ...
                internal.stats.parseArgs(args,defs,varargin{:});
            if this.TableInput
                vrange = this.VariableRange;
            else
                vrange = {};
            end
            [X,Y,W] = prepareDataForLoss(this,X,Y,W,vrange,true);

            l = CompactEnsemble2.aggregateLoss(...
                this.NTrained,X,Y,W,[],funloss,this.Impl.Combiner,...
                @CompactEnsemble2.predictOneWithCache,...
                this.Impl.Trained,[],[],this.PrivResponseTransform,NaN,...
                'usepredforlearner',this.UsePredForLearner,extraArgs{:});
        end
        
        function [varargout] = predictorImportance(this,varargin)
        %PREDICTORIMPORTANCE Estimates of predictor importance.
        %   IMP=PREDICTORIMPORTANCE(ENS) computes estimates of predictor importance
        %   for ensemble ENS by summing these estimates over all weak learners in
        %   the ensemble. The returned vector IMP has one element for each input
        %   predictor in the data used to train this ensemble. A high value
        %   indicates that this predictor is important for this ensemble.
        %
        %   [IMP,MA]=PREDICTORIMPORTANCE(ENS) for ensembles of decision trees also
        %   returns a P-by-P matrix with predictive measures of association for P
        %   predictors. Element MA(I,J) is the predictive measure of association
        %   averaged over surrogate splits on predictor J for which predictor I is
        %   the optimal split predictor. PREDICTORIMPORTANCE averages this
        %   predictive measure of association over all trees in the ensemble.
        %
        %   See also RegressionEnsemble, CompactRegressionEnsemble,
        %   classreg.learning.regr.CompactRegressionTree/predictorImportance,
        %   classreg.learning.regr.CompactRegressionTree/meanSurrVarAssoc.
            
            [varargout{1:nargout}] = predictorImportance(this.Impl,varargin{:});
        end
    end
    
    methods ( Hidden)
        function s = toStruct(this)
            % Convert to a struct for codegen.
            
            warnState  = warning('query','all');
            warning('off','MATLAB:structOnObject');
            cleanupObj = onCleanup(@() warning(warnState));

            % convert common properties to struct
            s = classreg.learning.coderutils.regrToStruct(this);   
            % save the path to the fromStruct method
            s.FromStructFcn = 'CompactRegressionEnsemble2.fromStruct';
            
            % weak learners
            trained = this.Trained;
            L = numel(trained);
            
            if L == 0
                error(message('stats:classreg:learning:classif:CompactClassificationEnsemble:toStruct:EmptyModelNotSupported'));
            end
            
            trained_struct = struct;
            
            for j=1:L
                fname = ['Learner_' num2str(j)];
                if isempty(trained{j})
                    trained_struct.(fname) = trained{j};
                else
                    trained_struct.(fname) = trained{j}.toStruct;
                end
            end
            
            s.NumTrained = L;
            s.Impl.Trained = trained_struct;
            s.UsePredForLearner = this.UsePredForLearner;
            s.Impl.Combiner = struct('LearnerWeights',this.Impl.Combiner.LearnerWeights,'IsCached',this.Impl.Combiner.IsCached);
            combinerClassFull = class(this.Impl.Combiner);
            combinerClassList = strsplit(combinerClassFull,'.');
            combinerClass = combinerClassList{end};
            s.Impl.CombinerClass = combinerClass;
                          
        end        
    end
    methods(Static=true,Hidden=true)
        function obj = fromStruct(s)
            % Make an Ensemble object from a codegen struct.
            
            s.ResponseTransform = s.ResponseTransformFull;
            s = classreg.learning.coderutils.structToRegr(s);
            
            % Prepare a cell array of learners
            L = s.NumTrained;
            trained = cell(L,1);
            
            for j=1:L
                fname = ['Learner_' num2str(j)];
                trained_struct = s.Impl.Trained.(fname);
                if ~isempty(trained_struct)
                    fcn = str2func(trained_struct.FromStructFcn);
                    trained{j} = fcn(trained_struct);
                else
                    trained{j} = trained_struct;
                end
            end
      
            % Make an object
            obj = CompactRegressionEnsemble2(...
                s.DataSummary,s.ResponseTransform,...
                s.UsePredForLearner);

            learnerweights = s.Impl.Combiner.LearnerWeights;
            combinerClassFull = ['classreg.learning.combiner.' s.Impl.CombinerClass];
            combinerClass = str2func(combinerClassFull);
            combiner = combinerClass(learnerweights);
            impl = classreg.learning.impl.CompactEnsembleImpl(trained,combiner);   
            obj.Impl = impl;

        end
    end
    
    methods(Hidden, Static)
        function name = matlabCodegenRedirect(~)
            name = 'CompactRegressionEnsemble2';
        end
    end 
end
