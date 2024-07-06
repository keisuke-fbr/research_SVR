classdef Predictor2 < classreg.learning.internal.DisallowVectorOps
%Predictor Super class for all supervised-learning models.

%   Copyright 2010-2020 The MathWorks, Inc.


    properties(GetAccess=public,SetAccess=public,Hidden=true)
        Impl = [];
    end
    
    properties(Dependent=true,GetAccess=public,SetAccess=public,Hidden=true)
        VariableRange;
        TableInput;
    end
    
    properties(Dependent=true,GetAccess=public,SetAccess=protected,Hidden=true)
        ObservationsInRows;
    end
    
    properties(GetAccess=public,SetAccess=protected,Hidden=true)
        CategoricalVariableCoding = 'index'; % or 'dummy'
        DataSummary = struct(...
            'PredictorNames',{},...
            'CategoricalPredictors',[],...
            'ResponseName','',...
            'VariableRange',{},...
            'TableInput',false,...
            'RowsUsed',[],...
            'ObservationsInRows',true,...
            'BinnedX',[],...
            'BinEdges',{},...
            'OrdinalPredictors',[],...
            'ReferenceLevels',[]);
    end

    properties(GetAccess=public,SetAccess=protected,Dependent=true)
        %PREDICTORNAMES Names of predictors used for this model.
        %   The PredictorNames property is a cell array of strings with names of
        %   predictor variables, one name per column of X.
        %
        %   See also classreg.learning.Predictor.
        PredictorNames;
        
        %CATEGORICALPREDICTORS Indices of categorical predictors.
        %   The CategoricalPredictors property is an array with indices of
        %   categorical predictors. The indices are in the range from 1 to the
        %   number of columns in X.        
        %
        %   See also classreg.learning.Predictor.
        CategoricalPredictors;
        
        %RESPONSENAME Name of the response variable.
        %   The ResponseName property is a string with the name of the response
        %   variable Y.
        %
        %   See also classreg.learning.Predictor.
        ResponseName;
        
        %EXPANDEDPREDICTORNAMES Names of the expanded predictors used for this model.
        %   The ExpandedPredictorNames property is a cell array of strings
        %   with names of the expanded predictor variables. For many
        %   models, this is the same as PredictorNames. For models that use
        %   an expanded (dummy variable) representation of categorical
        %   predictors, this describes the expanded variables. The expanded
        %   names describe the columns of the SupportVectors matrix for an
        %   SVM model, and the columns of the ActiveSetVectors matrix for a
        %   Gaussian Process model.
        %
        %   See also PredictorNames, classreg.learning.Predictor.
        ExpandedPredictorNames;
    end
    
    properties(GetAccess=protected,SetAccess=protected,Dependent=true)
        OrdinalPredictors
        ReferenceLevels
    end

    methods(Abstract)
        varargout = predict(this,X,varargin)
    end
    
    methods(Hidden)
        function disp(this)
            internal.stats.displayClassName(this);
            
            % Body display
            s = propsForDisp(this,[]);
            disp(s);
            
            internal.stats.displayMethodsProperties(this);
        end
        
        function vrange = getvrange(this)
            % some classes are prepared to
            % deal with whatever categorical variables appear in
            % variables in a matrix, so use an empty vrange in that case
            if this.TableInput % && ~strcmp(this.CategoricalVariableCoding,'index')
                vrange = this.VariableRange;
            else
                vrange = {};
            end
        end
    end
    
    methods
        function n = get.PredictorNames(this)
            names = this.DataSummary.PredictorNames;
            if isnumeric(names)
                n = classreg.learning.internal.defaultPredictorNames(names);
            else
                n = names;
            end
        end
        
        function c = get.CategoricalPredictors(this)
            c = this.DataSummary.CategoricalPredictors;
        end
        
        function r = get.ResponseName(this)
            r = this.DataSummary.ResponseName;
        end
        
        function vr = get.VariableRange(this)
            try
                vr = this.DataSummary.VariableRange;
            catch
                vr = {}; % field may be missing for old saved object
            end
        end
        
        function this = set.VariableRange(this,vr)
            this.DataSummary.VariableRange = vr;
        end
        
        function ti = get.TableInput(this)
            try
                ti = this.DataSummary.TableInput;
            catch
                ti = false; % field may be missing for old saved object
            end
        end
        
        function this = set.TableInput(this,t)
            this.DataSummary.TableInput = t;
        end
        
        function n = get.ExpandedPredictorNames(this)
            n = getExpandedPredictorNames(this);
        end
        
        function tf = get.ObservationsInRows(this)
            try
                tf = this.DataSummary.ObservationsInRows;
            catch
                tf = true; % field may be missing for old saved object
            end
        end
        
        function ord = get.OrdinalPredictors(this)
            try
                ord = this.DataSummary.OrdinalPredictors;
            catch
                ord = []; % field may be missing for old saved object
            end
        end
        
        function ref = get.ReferenceLevels(this)
            try
                ref = this.DataSummary.ReferenceLevels;
            catch
                ref = []; % field may be missing for old saved object
            end
        end
    end

    methods(Access=protected)
        function this = Predictor2(dataSummary)
            this = this@classreg.learning.internal.DisallowVectorOps();
            this.Impl = [];
            if ~isfield(dataSummary,'VariableRange')
                dataSummary.VariableRange = {};
            end
            if ~isfield(dataSummary,'TableInput')
                dataSummary.TableInput = false;
            end
            if ~isfield(dataSummary,'RowsUsed')
                dataSummary.RowsUsed = [];
            end
            if ~isfield(dataSummary,'OrdinalPredictors')
                if isnumeric(dataSummary.PredictorNames)
                    dataSummary.OrdinalPredictors = false(1,dataSummary.PredictorNames);
                else
                    dataSummary.OrdinalPredictors = false(1,numel(dataSummary.PredictorNames));
                end
            end
            if ~isfield(dataSummary,'ReferenceLevels')
                dataSummary.ReferenceLevels = [];
            end
            
            this.DataSummary = dataSummary;
        end
        
        function n = getExpandedPredictorNames(this)
            % Default implementation; override in classes that use an
            % expanded design matrix.
            n = this.PredictorNames;
        end
        
        function n = getOptionalPredictorNames(this)
            % Used when the numeric version is okay; no need to create
            % default names (mostly used when before calling
            % classreg.learning.internal.table2PredictMatrix)
            n = this.DataSummary.PredictorNames;
        end
        
        function n = getNumberPredictors(this)
            % Used instead of numel(this.PredictorNames) to avoid
            % constructing defaultPredictorNames if not needed
            n = this.DataSummary.PredictorNames;
            if ~isnumeric(n)
                n = numel(n);
            end
            end
        
        function s = propsForDisp(this,s)
            % The 2nd input argument is a struct accumulating fields to be
            % displayed by derived classes.
            if nargin<2 || isempty(s)
                s = struct;
            else
                if ~isstruct(s)
                    error(message('stats:classreg:learning:Predictor:propsForDisp:BadS'));
                end
            end
            
            names = this.DataSummary.PredictorNames;
            if iscellstr(names) && numel(names)<=100
                s.PredictorNames = this.PredictorNames;
            end
            s.ResponseName = this.ResponseName;
            s.CategoricalPredictors = this.CategoricalPredictors;
        end
    end
end
