classdef FullRegressionModel2 < ...
        FullClassificationRegressionModel2 & RegressionModel2
%FullRegressionModel Full regression model.
%   FullRegressionModel is the super class for full regression
%   models represented by objects storing the training data. This class is
%   derived from RegressionModel.
%
%   See also classreg.learning.regr.RegressionModel.

%   Copyright 2010-2021 The MathWorks, Inc.

    properties(GetAccess=public,SetAccess=protected,Dependent=true)
        %Y Observed response used to train this model.
        %   The Y property is a vector of type double.
        %
        %   See also classreg.learning.regr.FullRegressionModel.
        Y;
    end
    
    methods
        function y = get.Y(this)
            y = this.PrivY;
        end
    end
        
    methods(Access=protected)
        function this = FullRegressionModel2(X,Y,W,modelParams,dataSummary,responseTransform)
            this = this@FullClassificationRegressionModel2(...
                dataSummary,X,Y,W,modelParams);
            this = this@RegressionModel2(dataSummary,responseTransform);
            this.ModelParams = fillIfNeeded(modelParams,X,Y,W,dataSummary,[]);
        end
        
        function s = propsForDisp(this,s)
            s = propsForDisp@RegressionModel2(this,s);
            s = propsForDisp@FullClassificationRegressionModel2(this,s);
        end
    end
    
    methods
        function partModel = crossval(this,varargin)
        %CROSSVAL Cross-validate this model.
        %   CVMODEL = CROSSVAL(MODEL) builds a partitioned model CVMODEL from model
        %   MODEL represented by a full object for regression. You can then
        %   assess the predictive performance of this model on cross-validated data
        %   using methods and properties of CVMODEL. By default, CVMODEL is built
        %   using 10-fold cross-validation on the training data.
        %
        %   CVMODEL = CROSSVAL(MODEL,'PARAM1',val1,'PARAM2',val2,...) specifies
        %   optional parameter name/value pairs:
        %      'KFold'      - Number of folds for cross-validation, a numeric
        %                     positive scalar; 10 by default.
        %      'Holdout'    - Holdout validation uses the specified
        %                     fraction of the data for test, and uses the rest of
        %                     the data for training. Specify a numeric scalar
        %                     between 0 and 1.
        %      'Leaveout'   - If 'on', use leave-one-out cross-validation.
        %      'CVPartition' - An object of class CVPARTITION; empty by default. If
        %                      a CVPARTITION object is supplied, it is used for
        %                      splitting the data into subsets.
        %
        %   See also classreg.learning.regr.FullRegressionModel,
        %   cvpartition,
        %   classreg.learning.partition.RegressionPartitionedModel.
            [varargin{:}] = convertStringsToChars(varargin{:});
            idxBaseArg = find(ismember(varargin(1:2:end),...
                classreg.learning.FitTemplate.AllowedBaseFitObjectArgs));
            if ~isempty(idxBaseArg)
                error(message('stats:classreg:learning:regr:FullRegressionModel:crossval:NoBaseArgs', varargin{ 2*idxBaseArg - 1 }));
            end
            nbins = max(cellfun(@numel,this.DataSummary.BinEdges)) + 1;
            temp = classreg.learning.FitTemplate.make(this.ModelParams.Method,...
                'type','regression','responsetransform',this.PrivResponseTransform,...
                'modelparams',this.ModelParams,'CrossVal','on','NumBins',nbins,...
                varargin{:});
            
            if this.DataSummary.ObservationsInRows
                observationsIn = 'rows';
            else
                observationsIn = 'columns';
            end
            
            partModel = fit(temp,this.X,this.Y,'Weights',this.W,...
                'predictornames',this.PredictorNames,'categoricalpredictors',this.CategoricalPredictors,...
                'responsename',this.ResponseName,...
                'ObservationsIn',observationsIn);
        end

        function [varargout] = resubPredict(this,varargin)
            [varargin{:}] = convertStringsToChars(varargin{:});
            if isfield(this.DataSummary,'ObservationsInRows') && ...
                    ~this.DataSummary.ObservationsInRows
                varargin{end+1} = 'ObservationsIn';
                varargin{end+1} = 'Columns';
            end
            FullClassificationRegressionModel2.catchWeights(varargin{:});
            [varargout{1:nargout}] = predict(this,this.X,varargin{:});
        end
        
        function [varargout] = resubLoss(this,varargin)
            [varargin{:}] = convertStringsToChars(varargin{:});
            if isfield(this.DataSummary,'ObservationsInRows') && ...
                    ~this.DataSummary.ObservationsInRows
                varargin{end+1} = 'ObservationsIn';
                varargin{end+1} = 'Columns';
            end
            FullClassificationRegressionModel2.catchWeights(varargin{:});
            [varargout{1:nargout}] = ...
                loss(this,this.X,this.Y,'Weights',this.W,varargin{:});
        end
        
        function [pd,varargout] = partialDependence(this,features,varargin)
        %PARTIALDEPENDENCE Partial Dependence for one and two variables
        %   PD = partialDependence(MODEL,VAR) returns a vector PD of partial 
        %   dependence for a single predictor variable VAR, and input data DATA, 
        %   based on the response of a fitted regression model MODEL. VAR is a 
        %   scalar containing the index of a predictor or a character array 
        %   specifying a predictor name.
        %
        %   PD = partialDependence(MODEL,VARS) returns a two dimensional 
        %   array PD of the partial dependences for two predictors VARS, based on 
        %   the response of a fitted regression model MODEL. VARS is either a cell 
        %   array containing two predictor names, or a two-element vector containing 
        %   the indices of two predictors.
        %
        %   PD = partialDependence(...,DATA) specifies the data to be used for 
        %   averaging. DATA is a matrix or table of data to be used in place of the
        %   data used in fitting the model.
        %
        %   PD = partialDependence(...,Name,Value) specifies additional options
        %   using one or more name-value pair arguments:
        % 
        %       'NumObservationsToSample':      An integer specifying the number
        %                                       of rows to sample at random from the
        %                                       dataset (either the data specified by
        %                                       argument DATA or the data used to train
        %                                       MODEL). The default is to use all rows.
        % 
        %       'QueryPoints':                  The points at which to calculate
        %                                       the partial dependence. For the case of
        %                                       a single predictor, the value of
        %                                       'QueryPoints' must be a column vector.
        %                                       For the case of two predictors, the
        %                                       value of 'QueryPointsâ€' must be a 1x2
        %                                       cell array containing a separate vector
        %                                       for each predictor. The default is to
        %                                       use 100 points equally spaced across
        %                                       the range of the predictor.
        % 
        %       'UseParallel'                   Can be true to instruct the method to
        %                                       perform the averaging calculations in
        %                                       parallel (using parfor), or false
        %                                       (default) to instruct the method not to
        %                                       use parallel computation.
        % 
        %   [PD,X,Y] = partialDependence(...) also returns the arrays X and Y
        %   containing the query point values of the first and second predictor in VARS,
        %   respectively.
        %
        %   Examples:
        %       % Partial Dependence Plot of Regression Tree
        %       load carsmall
        %       tbl = table(Weight,Cylinders,Origin,MPG);
        %       f = fitrtree(tbl,'MPG');
        % 
        %       pd = partialDependence(f,'Weight');
        %       pd = partialDependence(f,{'Weight','Origin'});
        % 
        % 
        %       % Obtain optional output Axes handle
        %       [pd,x] = partialDependence(f,1);
        %       plot(x,pd)
        % 
        %       % Obtain values x and y of the first and second variables
        %       [pd,x,y] = partialDependence(f,[1,3]);
        %       surf(x,y,pd)
        % 
        %       % With additional Data
        %       load carbig
        %       tbl2 = table(Weight,Cylinders,Origin);
        %       pd = partialDependence(f,'Weight',tbl2);
        %       pd = partialDependence(f,1,tbl2);
        % 
        %       % With optional name-value pairs
        %       pd = partialDependence(f,1,tbl2,'NumObservationsToSample',100);
        %       pd = partialDependence(f,1,tbl2,'UseParallel',true);
        % 
        %       % Provide alternative query points
        %       xi = linspace(min(Weight),max(Weight))';
        %       pd = partialDependence(f,1,'QueryPoints',xi);
        % 
        %       xi = cell(1,2);
        %       xi{1} = linspace(min(Weight),max(Weight))';
        %       xi{2} = linspace(min(Cylinders),max(Cylinders))';
        %       pd = partialDependence(f,[1,2],'QueryPoints',xi);
        
        %   Copyright 2020 The MathWorks, Inc.
        
        %-------Check number of inputs----
        narginchk(2,9);
        features = convertStringsToChars(features);
        [varargin{:}] = convertStringsToChars(varargin{:});

        % Check inputs with inputParser. This step ensures that the third
        % argument is either a Name-Value pair or data, no other strings/char
        % array allowed.
        p = inputParser;        
        addRequired(p,'Model');
        addRequired(p,'Var');
        addOptional(p,'Data',this.X); % Default - training data
        addParameter(p,'NumObservationsToSample',0);
        addParameter(p,'QueryPoints',[]);
        addParameter(p,'UseParallel',false);
        parse(p,this,features,varargin{:});
        X = p.Results.Data;
       
        % Since this is a regression model, the classification flag is false
        % and class label is empty
        classif = false;
        label = [];
        
        %------Parse Data-----------------
        % If third argument is a char, its a parameter name else it is Data
        if(nargin>2 && ~ischar(varargin{1}))
            % Pass everything but the first argument (Data) to compact method
            varargin = varargin(2:end);
        end
        
        % Call the function from classreg.learning.impl
        [pd,x,y] = classreg.learning.impl.partialDependence(this,features,...
            label,X,classif,varargin{:});
        varargout{1} = x;
        varargout{2} = y;
        end

        function [AX] = plotPartialDependence(this,features,varargin)
        %PLOTPARTIALDEPENDENCE Partial Dependence Plot for 1-D or 2-D visualization
        %   plotPartialDependence(MODEL,VAR) creates a plot of the partial 
        %   dependence of the response by a fitted regression model MODEL on the 
        %   predictor VAR. VAR is a scalar containing the index of a predictor or a 
        %   character array specifying a predictor name.
        %   
        %   plotPartialDependence(MODEL,VARS) creates a partial dependence plot 
        %   of response by a fitted regression model MODEL against two predictors
        %   VARS. VARS is either a cell array containing two predictor names, or a 
        %   two-element vector containing the indices of two predictors.
        %
        %   plotPartialDependence(...,DATA) specifies the data to be used for
        %   averaging. DATA is a matrix or table of data to be used in place of the
        %   data used in fitting the model.
        %
        %   plotPartialDependence(...,Name,Value) specifies additional options using
        %   one or more name-value pair arguments:
        %
        %   'Conditional'               'none' (default) instructs the method to create
        %                               a partial dependence plot with no conditioning,
        %                               'absolute' to create an individual conditional
        %                               expectation (ICE) plot, and 'centered' to
        %                               create an ICE plot with centered data.
        %
        %   'NumObservationsToSample'	An integer specifying the number
        %                               of rows to sample at random from the dataset
        %                               (either the data specified by the argument DATA
        %                               or the data used to train MODEL). The default
        %                               is to use all rows.
        %
        %   'QueryPoints'               The points at which to calculate
        %                               the partial dependence. For the case of a
        %                               single predictor, the value of 'QueryPoints'                            
        %                               must be a column vector. For the case of two
        %                               predictors, the value of 'QueryPoints' must be
        %                               a 1x2 cell array containing a separate vector
        %                               for each predictor. The default is to use 100
        %                               points equally spaced across the range of the
        %                               predictor.
        %
        %   'UseParallel'               Can be true to instruct the method to perform
        %                               the averaging calculations in parallel (using
        %                               parfor), or false (default) to instruct the
        %                               method not to use parallel computations.
        %
        %   'Parent'                    Instructs the method to create a partial
        %                               dependence plot using the axes with handle
        %                               specified by the value of this parameter.
        %
        %   AX = plotPartialDependence(...) returns a handle AX to the axes of the
        %   plot.
        %
        %   Examples:
        %      % Partial Dependence Plot of Regression Tree
        %      load carsmall
        %      tbl = table(Weight,Cylinders,Origin,MPG);
        %      f = fitrtree(tbl,'MPG');
        %
        %      plotPartialDependence(f,'Weight');
        %      plotPartialDependence(f,{'Weight','Origin'});
        %      plotPartialDependence(f,[1,3]);
        %
        %      % Obtain optional output Axes handle
        %      ax = plotPartialDependence(f,1);
        %
        %      % With additional Data
        %      load carbig
        %      tbl2 = table(Weight,Cylinders,Origin);
        %      plotPartialDependence(f,'Weight',tbl2);
        %      plotPartialDependence(f,1,tbl2);
        %
        %      % With optional name-value pairs
        %      plotPartialDependence(f,1,tbl2,'NumObservationsToSample',100);
        %      plotPartialDependence(f,1,tbl2,'UseParallel',true);
        %      plotPartialDependence(f,1,tbl2,'UseParallel',true,'Conditional','none');
        %      
        %      % Plot the Individual Conditional Expectation
        %      plotPartialDependence(f,1,tbl2,'Conditional','absolute');
        %
        %      % Provide alternative query points
        %      xi = linspace(min(Weight),max(Weight))';
        %      plotPartialDependence(f,1,'QueryPoints',xi);
        %      
        %      xi = cell(1,2);
        %      xi{1} = linspace(min(Weight),max(Weight))';
        %      xi{2} = linspace(min(Cylinders),max(Cylinders))';
        %      plotPartialDependence(f,[1,2],'QueryPoints',xi);
        
        %   Copyright 2020 The MathWorks, Inc.

        %-------Check number of inputs----
        narginchk(2,13);
        features = convertStringsToChars(features);
        [varargin{:}] = convertStringsToChars(varargin{:});

        % Check inputs with inputParser. This step ensures that the third
        % argument is either a Name-Value pair or data, no other strings/char
        % array allowed.
        p = inputParser;        
        addRequired(p,'Model');
        addRequired(p,'Var');
        addOptional(p,'Data',this.X); % Default - training data
        addParameter(p,'Conditional',{'none','absolute','centered'});
        addParameter(p,'NumObservationsToSample',0);
        addParameter(p,'QueryPoints',[]);
        addParameter(p,'UseParallel',false);
        addParameter(p,'ParentAxisHandle',[]);
        parse(p,this,features,varargin{:});
        X = p.Results.Data;
        
        %------Parse Data-----------------
        % If third argument is a char, its a parameter name else it is Data
        if(nargin>2 && ~ischar(varargin{1}))
            % Pass everything but the first argument(Data)to compact method
            varargin = varargin(2:end);
        end
        
        % Since this is a regression model, the classification flag is false
        % and class label is empty
        classif = false;
        label = [];
        
        % Call the function from classreg.learning.impl
        ax = classreg.learning.impl.plotPartialDependence(this,features,...
            label,X,classif,varargin{:});
        if(nargout > 0)
            AX = ax;
        end
        end
        
    end
        
    methods(Static,Hidden)
        function [X,Y,W,dataSummary,responseTransform] = prepareData(X,Y,varargin)
            [X,Y,vrange,wastable,varargin] = classreg.learning.internal.table2FitMatrix(X,Y,varargin{:});
            
            % Process input args
            args = {'responsetransform'};
            defs = {                 []};
            [transformer,~,crArgs] = ...
                internal.stats.parseArgs(args,defs,varargin{:});
            
            % Check Y type
            if ~isfloat(Y) || ~isvector(Y)
                error(message('stats:classreg:learning:regr:FullRegressionModel:prepareData:BadYType'));
            end
            internal.stats.checkSupportedNumeric('Y',Y,true);
            Y = Y(:);

            % Pre-process
            [X,Y,W,dataSummary] = ...
                FullClassificationRegressionModel2.prepareDataCR(...
                X,Y,crArgs{:},'VariableRange',vrange,'TableInput',wastable);
            if ~isfloat(X) && ~isa(X,'int32')
                error(message('stats:classreg:learning:regr:FullRegressionModel:prepareData:BadXType'));
            end

            [X,Y,W,dataSummary] = FullRegressionModel2.removeNaNs(X,Y,W,dataSummary);
      
            % Renormalize weights
           W = W/sum(W);

            % Make output response transformation
           responseTransform = ...
                FullRegressionModel2.processResponseTransform(transformer);
        end
        
        function [X,Y,W,dataSummary] = removeNaNs(X,Y,W,dataSummary)
            t = isnan(Y);
            if any(t)
                obsInRows = dataSummary.ObservationsInRows;
                rowsused = dataSummary.RowsUsed;
                
                Y(t) = [];
                if obsInRows
                    X(t,:) = [];
                    if ~isempty(dataSummary.BinnedX)
                        dataSummary.BinnedX(t,:) = [];
                    end
                else
                    X(:,t) = [];
                    if ~isempty(dataSummary.BinnedX)
                        dataSummary.BinnedX(:,t) = [];
                    end
                end
                W(t) = [];
                if isempty(rowsused)
                    rowsused = ~t;
                else
                    rowsused(rowsused) = ~t;
                end
                
                dataSummary.RowsUsed = rowsused;
            end
            
            if isempty(X)
                error(message('stats:classreg:learning:regr:FullRegressionModel:prepareData:NoGoodYData'));
            end
        end
            
        function responseTransform = processResponseTransform(transformer)
            if isempty(transformer)
                responseTransform = @classreg.learning.transform.identity;
            elseif ischar(transformer)
                if strcmpi(transformer,'none')
                    responseTransform = @classreg.learning.transform.identity;
                else
                    responseTransform = str2func(['classreg.learning.transform.' transformer(:)']);
                end
            else
                if ~isa(transformer,'function_handle')
                    error(message('stats:classreg:learning:regr:FullRegressionModel:prepareData:BadResponseTransformation'));
                end
                responseTransform = transformer;
            end
        end
    end
end
