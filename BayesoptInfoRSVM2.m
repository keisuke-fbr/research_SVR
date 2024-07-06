classdef BayesoptInfoRSVM2 < BayesoptInfo2
    %

    %   Copyright 2016-2018 The MathWorks, Inc.
    
    properties
        FitFcn = @fitrsvm2;
        PrepareDataFcn = @RegressionSVM2.prepareData;
        AllVariableDescriptions;
    end
    
    methods
        function this = BayesoptInfoRSVM2(Predictors, Response, FitFunctionArgs)
            this@BayesoptInfo2(Predictors, Response, FitFunctionArgs, false, true);
            this.AllVariableDescriptions = configureVariableDescriptions(this, Predictors, Response);
            this.ConditionalVariableFcn = @fitrsvmCVF;
        end
    end
    
    methods(Access=protected)
        function Descriptions = configureVariableDescriptions(this, Predictors, Response)
            ResponseIqr = this.ResponseIqr;
            if ResponseIqr == 0
                ResponseIqr = 1;
            end
            Descriptions = [...
                optimizableVariable('BoxConstraint', [1e-3, 1e3], 'Transform', 'log');
                optimizableVariable('KernelScale', [1e-3, 1e3], 'Transform', 'log');
                optimizableVariable('Epsilon', [1e-3*ResponseIqr/1.349, 1e2*ResponseIqr/1.349], ...
                'Transform', 'log');
                optimizableVariable('KernelFunction', {'gaussian', 'linear', 'polynomial'}, 'Optimize', false);
                optimizableVariable('PolynomialOrder', [2, 4], 'Type', 'integer', 'Optimize', false);
                optimizableVariable('Standardize', {'true', 'false'}, 'Optimize', false)];
        end
    end
end

function XTable = fitrsvmCVF(XTable)
% KernelScale is irrelevant if KernelFunction~='rbf' or 'gaussian'
if classreg.learning.paramoptim.BayesoptInfo.hasVariables(XTable, {'KernelScale', 'KernelFunction'})
    XTable.KernelScale(~ismember(XTable.KernelFunction, {'rbf','gaussian'})) = NaN;
end
% PolynomialOrder is irrelevant if KernelFunction~='polynomial'
if classreg.learning.paramoptim.BayesoptInfo.hasVariables(XTable, {'PolynomialOrder', 'KernelFunction'})
    XTable.PolynomialOrder(XTable.KernelFunction ~= 'polynomial') = NaN;
end
end
