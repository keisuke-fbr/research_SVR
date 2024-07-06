classdef SVMImpl2 < CompactSVMImpl2
% SVM algorithm implementation. 
%
% The methods in the following class accept also gpuArrays. To determine
% the array class we should always use isUnderlyingType, instead of isa.

% Copyright 2013-2020 The MathWorks, Inc.

    properties(SetAccess=protected,GetAccess=public)
        Active                      = [];
        C                           = [];
        CacheInfo                   = [];
        ClipAlphas                  = [];
        ConvergenceInfo             = [];
        FractionToExclude           = [];
        Gradient                    = [];
        IsSupportVector             = [];
        Nu                          = [];
        NumIterations               = 0;
        NumMatrixVectorProductTaken = 0;
        PreferredUpdate             = [];
        Shrinkage                   = [];
        WorkingSet                  = [];
        OriginalToUnique            = [];
        UniqueToOriginal            = [];
    end
    
     
    methods(Access=protected)
        function this = SVMImpl2()
            this = this@CompactSVMImpl2();
        end
    end
    
    methods
        function cmp = compact(this,saveSV)
            cmp = CompactSVMImpl2();
            cmp.Beta                = this.Beta;
            cmp.Bias                = this.Bias;
            cmp.KernelParameters    = this.KernelParameters;
            cmp.Mu                  = this.Mu;
            cmp.NumPredictors       = this.NumPredictors;
            cmp.Sigma               = this.Sigma;          
            cmp.Epsilon             = this.Epsilon;
            if saveSV
                cmp.Alpha               = this.Alpha;
                cmp.SupportVectors      = this.SupportVectors;
                cmp.SupportVectorLabels = this.SupportVectorLabels;
            end
        end
        
        function this = resume(this,X,Y,numIter,doclass,verbose,nprint,iscat,isord,vrange)
        %RESUME Resume training by SMO or ISDA
        %   OBJ=RESUME(OBJ,X,Y,NUMITER,DOCLASS,VERBOSE,NPRINT) resumes training the
        %   model OBJ on data X and Y for NUMITER iterations. Pass Y as a vector
        %   filled with -1 and +1. Pass DOCLASS as true or false. Pass VERBOSE as a
        %   non-negative integer. Pass NPRINT as a non-negative integer.

            if isUnderlyingType(X, "single")
                Y = single(Y);
            end
            
            if this.PreferredUpdate==0
                % The model must've been obtained by quadprog
                error(message('stats:classreg:learning:impl:SVMImpl:resume:CannotResumeForSolver'));
            end
            
            if this.ConvergenceInfo.Converged
                error(message('stats:classreg:learning:impl:SVMImpl:resume:CannotResumeAfterConvergence'));
            end
            
            X = classreg.learning.internal.expandCategorical(X,iscat,isord,vrange);
            
            orig_to_unq = this.OriginalToUnique;
            unq_to_orig = this.UniqueToOriginal;
            
            Norig = size(X,1);
            if ~isempty(orig_to_unq)
                X = X(orig_to_unq,:);
                Y = Y(orig_to_unq);
            end
            
            mu = this.Mu;
            if ~isempty(mu) && ~all(mu==0)
                X = X - mu;
            end
            sigma = this.Sigma;
            if ~isempty(sigma) && ~all(sigma==1)
                nonZeroSigma = sigma>0;
                if any(nonZeroSigma)
                    X(:,nonZeroSigma) = ...
                        bsxfun(@rdivide,X(:,nonZeroSigma),sigma(nonZeroSigma));
                end
            end
            
            N = size(X,1);
            
            alphas = zeros(N,1);
            alphas(this.IsSupportVector(mapIdx(orig_to_unq,Norig,1))) = this.Alpha;
                  
            if isUnderlyingType(X, "double")
                alphaTol = sqrt(eps);
            else % single
                alphaTol = 10*sqrt(eps);
            end
            
            if doclass==0
                alphas = convertAlphas(alphas);
                c = [this.C; this.C];
            else
                c = this.C;
            end
            
            c      = c(mapIdx(orig_to_unq,Norig,doclass));
            grad   = this.Gradient(mapIdx(orig_to_unq,Norig,doclass));
            active = this.Active(mapIdx(orig_to_unq,Norig,doclass));
            
            clipAlphas = this.ClipAlphas;            
            
            kernelFun     = this.KernelParameters.Function;
            polyOrder     = this.KernelParameters.PolyOrder;
            sigmoidParams = this.KernelParameters.Sigmoid;
            kernelScale   = this.KernelParameters.Scale;
            kernelOffset  = this.KernelParameters.Offset;
            
            wssAlgs = this.WorkingSet.Algorithms;
            preferredUpdate = this.PreferredUpdate;

            maxIter = this.NumIterations + numIter;
            nFirstIter = this.NumIterations;
            
            kktTol             = this.ConvergenceInfo.KKTTolerance;
            gapTol             = this.ConvergenceInfo.GapTolerance;
            deltaGradTol       = this.ConvergenceInfo.DeltaGradientTolerance;
            outlierHistory     = this.ConvergenceInfo.OutlierHistory;
            
            nIterHistory       = this.ConvergenceInfo.History.NumIterations;
            gapHistory         = this.ConvergenceInfo.History.Gap;
            deltaGradHistory   = this.ConvergenceInfo.History.DeltaGradient;
            worstViolHistory   = this.ConvergenceInfo.History.LargestKKTViolation;
            nsvHistory         = this.ConvergenceInfo.History.NumSupportVectors;
            objHistory         = this.ConvergenceInfo.History.Objective;
            changeSetHistory   = this.ConvergenceInfo.ChangeSetHistory;

            cacheSize = this.CacheInfo.Size;
            cachingAlg = this.CacheInfo.Algorithm;

            nShrinkAfter = this.Shrinkage.Period;
            shrinkAlgs   = this.Shrinkage.Algorithms;
            
            fExclude = this.FractionToExclude;
            epsilon  = this.Epsilon;
            
            if isempty(this.Nu)
                if doclass==0
                    nu = NaN;
                else
                    % Object prior to 19a does not have Nu info.
                    nu = sum(alphas)/N;
                end
            else
                nu = this.Nu;
            end            
            
            [alphas,active,grad,bias,nIter,...
                reasonForConvergence,...
                gap,deltaGradient,kktViolation,Q,...
                nMtimesv,~,wssCounts,...
                outlierHistory,...
                nIterHistory,gapHistory,deltaGradHistory,...
                worstViolHistory,nsvHistory,objHistory,changeSetHistory] = ...
                iDispatchSolve(...
                alphas,grad,active,...
                X,Y,...
                kernelFun,polyOrder,sigmoidParams,kernelScale,kernelOffset,...
                wssAlgs,preferredUpdate,...
                c,maxIter,alphaTol,kktTol,gapTol,deltaGradTol,...
                cacheSize,cachingAlg,nShrinkAfter,shrinkAlgs,fExclude,...
                nFirstIter,outlierHistory,...
                nIterHistory,gapHistory,deltaGradHistory,...
                worstViolHistory,nsvHistory,objHistory,changeSetHistory,...
                verbose,nprint,doclass==0,epsilon,clipAlphas,nu);
            
            if doclass==1
                bias = -1 - quantile(grad,fExclude);
            end
            if doclass==0
                 alphas =  alphas(1:N)-alphas(N+1:2*N);
            end
            idxSV = abs(alphas)>0; 

            ws.Algorithms = this.WorkingSet.Algorithms;
            ws.Names      = this.WorkingSet.Names;
            if     isempty(wssCounts)
                ws.Counts = this.WorkingSet.Counts;
            elseif isempty(this.WorkingSet.Counts)
                ws.Counts = wssCounts;
            else
                ws.Counts = this.WorkingSet.Counts + wssCounts;
            end
            
            beta = [];
            if strcmpi(kernelFun,'linear')
                if any(idxSV)
                    % Compute beta
                    if doclass==0
                        beta = sum( X(idxSV,:).*alphas(idxSV), 1 )' / kernelScale;
                    else
                        beta = sum( X(idxSV,:).*Y(idxSV).*alphas(idxSV), 1 )' / kernelScale;
                    end
                else
                    beta = [];
                end
            end
            
            history.NumIterations       = nIterHistory;
            history.Gap                 = gapHistory;
            history.DeltaGradient       = deltaGradHistory;
            history.LargestKKTViolation = worstViolHistory;
            history.NumSupportVectors   = nsvHistory;
            history.Objective           = objHistory;
            
            % When training on a GPU, the data is on the GPU. Every model
            % parameter depending on X may potentially end up on the GPU.
            % We then gather every potential GPU parameter here to ensure
            % support vectors are always on the host. This should not
            % represent an issue, since gather is now supported even if PCT
            % is not installed.
            this.Active(mapIdx(orig_to_unq,Norig,doclass))    = active;
            this.Alpha                                        = alphas(idxSV);
            this.Beta                                         = beta;
            this.Bias                                         = bias;
            this.ConvergenceInfo.Converged                    = ~strcmpi(reasonForConvergence,'NoConvergence');
            this.ConvergenceInfo.ReasonForConvergence         = reasonForConvergence;
            this.ConvergenceInfo.Gap                          = gap;
            this.ConvergenceInfo.DeltaGradient                = deltaGradient;
            this.ConvergenceInfo.LargestKKTViolation          = kktViolation;
            this.ConvergenceInfo.Objective                    = Q;
            this.ConvergenceInfo.OutlierHistory               = outlierHistory;
            this.ConvergenceInfo.History                      = history;
            this.ConvergenceInfo.ChangeSetHistory             = changeSetHistory;
            this.Gradient                                     = grad(mapIdx(unq_to_orig,N,doclass));            
            this.IsSupportVector(mapIdx(orig_to_unq,Norig,1)) = idxSV;
            this.NumIterations                                = nIter;
            this.NumMatrixVectorProductTaken                  = this.NumMatrixVectorProductTaken + nMtimesv;
            this.WorkingSet                                   = ws;
            this.SupportVectors                               = gather(X(idxSV,:));
            
            if doclass~=0
                this.SupportVectorLabels                      = Y(idxSV);
            end
        end
    end
    
    methods(Static)
        function this = make(X,Y,W,alphas,clipAlphas,...
                kernelFun,polyOrder,sigmoidParams,kernelScale,kernelOffset,...
                doscale,doclass,solver,c,nu,...
                maxIter,kktTol,gapTol,deltaGradTol,...
                cacheSize,cachingAlg,nShrinkAfter,fExclude,...
                verbose,nprint,epsilon,iscat,isord,vrange,removeDups)

            [X,catcols] = classreg.learning.internal.expandCategorical(X,iscat,isord,vrange);
                        
            % Remove duplicates
            Norig = size(X,1);
            orig_to_unq = [];
            unq_to_orig = [];
            if removeDups
                [A,orig_to_unq,unq_to_orig] = unique([X Y],'rows','stable');
                if size(A,1) < size(X,1)
                    W = accumarray(unq_to_orig,W);
                    X = A(:,1:end-1);
                    Y = A(:,end);
                end
                clear A;
            end
            
            [N,D] = size(X);

            % doclass:
            %    0 - regression
            %    1 - one-class classification
            %    2 - two-class classification

            if isUnderlyingType(X, "single")
                Y = single(Y);
                W = single(W);
                c = single(c);
            end
                                    
            if ~strcmpi(kernelFun,'polynomial')
                polyOrder = 0; % dummy just to pass in something
            end
            
            if ~strcmpi(kernelFun,'builtin-sigmoid')
                sigmoidParams = [0 0]; % dummy just to pass in something
            end            
            
            mu    = [];
            sigma = [];
            if doscale && ~all(catcols)
                mu = zeros(1,size(X,2),'like',X);
                sigma = ones(1,size(X,2),'like',X);
                mu(~catcols) = classreg.learning.internal.wnanmean(X(:,~catcols),W);
                sigma(~catcols) = classreg.learning.internal.wnanvar(X(:,~catcols),W,1);
                X = X - mu;
                nonZeroSigma = sigma>0;
                sigma(~nonZeroSigma) = 0;
                if any(nonZeroSigma)
                    sigma = sqrt(sigma);
                    X(:,nonZeroSigma) = X(:,nonZeroSigma)./sigma(nonZeroSigma);
                end
            end
            
            if     strcmpi(solver,'SMO')
                wssAlgs = {'MaxViolatingPair' 'MaxDeltaQ'};
                shrinkAlgs = {'UpDownGradient'};
                preferredUpdate = 2;
            elseif strcmpi(solver,'ISDA')
                wssAlgs = {'WorstViolator' 'MaxGainFromHalves'};
                shrinkAlgs = {'KKTViolators'}; 
                preferredUpdate = 1;
            elseif strcmpi(solver,'All2D')
                wssAlgs = {'MaxDeltaQ' ... 
                    'MaxGainAndPrevFound' 'MaxGainFromHalves' 'MaxGainAndNearby'};
                shrinkAlgs = {'KKTViolators'};
                preferredUpdate = 2;
            else 
                preferredUpdate = 0;
            end

            if strcmpi(kernelScale,'auto')
                kernelScale = classreg.learning.svmutils.optimalKernelScale(...
                    X,Y,doclass);
            end

            kernelParams.Function  = kernelFun;
            kernelParams.PolyOrder = polyOrder;
            kernelParams.Sigmoid   = sigmoidParams;
            kernelParams.Scale     = kernelScale;
            kernelParams.Offset    = kernelOffset;
            
            if doclass==2 || doclass==0
                c = W*N*c; %incorporate the observation weights
                
                % Box constraint may have changed above - rescale alphas to
                % ensure that bound constraints on alpha are satisfied.
                if ~isempty(alphas)
                    isRegressionSVM = doclass==0;
                    alphas = iScaleAlphasForBoxConstraint(alphas, c, isRegressionSVM);
                end
            end
            
            if doclass==0 %regression
                nu = NaN;
                linearTerm = [sqrt(epsilon)-Y; sqrt(epsilon)+Y];
                active = true(2*N,1);
                grad   = linearTerm; %default initial gradient
                c =[c;c];
            else
                active = true(N,1);
                grad   = -ones(N,1);
            end
           
            alphasAtBC = false;
          
            if isempty(alphas)
                if doclass==1
                    alphas = repmat(nu,N,1); % sum(alphas)=nu*N
                    if nu==1
                        alphasAtBC = true;
                    end
                else % regression or binary classification
                    alphas = zeros(N,1);
                end
            end
          
            if any(alphas>0)
                if ~strcmpi(solver,'L1QP') || alphasAtBC
                    % Compute the initial gradient
                    if doclass==0
                        idxSV = abs(alphas) > 0;
                        f = iDispatchPredict(...
                            alphas(idxSV),kernelOffset*sum(alphas),X(idxSV,:),...
                            kernelFun,polyOrder,sigmoidParams,kernelScale,X);
                        grad = [f+linearTerm(1:N) ; -f+linearTerm(N+1:2*N)];
                    else
                        idxSV = alphas>0;
                        y_times_alphas = Y(idxSV).*alphas(idxSV);
                        grad = Y.*iDispatchPredict(...
                            y_times_alphas,kernelOffset*sum(y_times_alphas),X(idxSV,:),...
                            kernelFun,polyOrder,sigmoidParams,kernelScale,X) - 1;
                    end
                end
            end
            
            if doclass==0 %Convert alpha_up-alpha_low to [alpha_up alpha_low]
                alphas = convertAlphas(alphas);
            end
            
            if doclass==1
                c = ones(N,1);
                fExcludeOneClass = fExclude;
                fExclude = 0; % do not remove observations with large gradients
            end
            
            % Tolerance on alpha coefficients.
            % Experimentally, these values work best.
            if isUnderlyingType(X, "double")
                alphaTol = sqrt(eps);
            else % single
                alphaTol = 10*sqrt(eps);
            end

            if     alphasAtBC
                % Handle special case: all alphas are at the box constraint
                % for one-class learning
                convergenceInfo.Converged                  = true;
                convergenceInfo.ReasonForConvergence       = ...
                    getString(message('stats:classreg:learning:impl:SVMImpl:make:AllAlphasMustBeAtBoxConstraint'));
                convergenceInfo.GapTolerance               = gapTol;
                convergenceInfo.DeltaGradientTolerance     = deltaGradTol;
                convergenceInfo.KKTTolerance               = kktTol;
                nIter = 0;
                nMtimesv = 0;
                shrinkage = [];
                ws = [];
                Q = alphas'*(grad-1)/2;
                
            elseif strcmpi(solver,'L1QP')
                % Check Optim license
                if isempty(ver('Optim'))
                    error(message('stats:classreg:learning:impl:SVMImpl2:make:QPNeedsOptim'));
                end
                
                cachingAlg = '';
                nMtimesv = [];
                shrinkage = [];
                ws = [];
                %G(i,j)= K(X_i,X_j)
                G = classreg.learning.svmutils.computeGramMatrix(...
                    X,kernelFun,polyOrder,sigmoidParams,kernelScale,kernelOffset);
                G = (G+G')/2;
                G = G + eye(N)/(2*c(1));
                % G = G + speye(N)/(2*c(1)); % 少しでもメモリの軽減. 単位行列みたいなスパースな行列はspeyeで定義したほうがメモリの節約になる.
                % ただし注意してほしいのが, quadprogに対する引数はdouble型でないといけないので,
                % 入力データをsingleにして, メモリの節約とかはできない.
                
                % Increase tolerance for post-QP processing
                oldAlphaTol = alphaTol;
                alphaTol = 100*alphaTol;
                idxbad = find(c<alphaTol,1);
                if ~isempty(idxbad)
                    error(message('stats:classreg:learning:impl:SVMImpl:make:BadBoxConstraintForObservation',...
                        idxbad,sprintf('%e',c(idxbad))));
                end
                
                opts = optimoptions(@quadprog,...
                    'Algorithm','interior-point-convex',...
                    'TolX',oldAlphaTol,'TolCon',oldAlphaTol);
                opts.MaxIter = maxIter;
                if     verbose>0
                    opts.Display = 'iter';
                else
                    opts.Display = 'none';
                end
                
                if doclass==0
                     Z =[ones(N,1); -ones(N,1)];
                else 
                     Z = Y;
                end
                if  doclass==1 %One-class learning
                    [alphas,Q,exitflag,output] = ...
                        quadprog(double(G),[],[],[],...
                        ones(1,N),double(sum(alphas)),...
                        zeros(N,1),double(c),[],opts);
                elseif doclass==2 %Two-class classification
                    [alphas,Q,exitflag,output] = ...
                        quadprog(double((Y*Y').*G),-ones(N,1),[],[],...
                        double(Y)',0,...
                    zeros(N,1),double(c),[],opts);        
                elseif doclass==0 %Regression
                   [alphas,Q,exitflag,output] =...
                       quadprog(double([G -G;-G G]),linearTerm,[],[],...
                       Z', 0,...
                       zeros(2*N,1),[],[],opts);
               end
                
                if exitflag < 0
                    error(message('stats:classreg:learning:impl:SVMImpl:make:NoQPConvergence',...
                        sprintf('\n %s',output.message)));
                end
                
                convergenceInfo.Converged      = exitflag==1;
                convergenceInfo.QuadprogOutput = output;
                
                % Find at least one support vector
                if any(alphas>0)
                    while ~any(alphas>alphaTol)
                        alphaTol = alphaTol/10;
                    end
                end
                
                nIter = output.iterations;
                                   
                if doclass==0 %regression
                    % grad = [G -G;-G G] *alphas + linearTerm;
                    alphas2 =  alphas(1:N)-alphas(N+1:2*N);
                    idxSV = abs(alphas2) > alphaTol;
                    sumAlphas = sum(alphas2);
                    if any(idxSV)
                        f = classreg.learning.svmutils.predict(...
                            alphas2(idxSV),kernelOffset*sum(alphas2),X(idxSV,:),...
                            kernelFun,polyOrder,sigmoidParams,kernelScale,X);
                        grad= [f+linearTerm(1:N) ; -f+linearTerm(N+1:2*N)];
                    end
                else %classification and one class learning
                    idxSV = alphas>alphaTol;     %
                    sumAlphas = 0;
                    if any(idxSV)
                        y_times_alphas = Y(idxSV).*alphas(idxSV);
                        sumAlphas = sum(y_times_alphas);
                        % grad = (Y*Y').*G *alphas - 1;
                        grad = Y.*classreg.learning.svmutils.predict(...
                            y_times_alphas,kernelOffset*sumAlphas,X(idxSV,:),...
                            kernelFun,polyOrder,sigmoidParams,kernelScale,X) - 1;
                    end
                end
                %Are there any Free SVs?
                isFree = alphas>alphaTol & alphas<Inf; %0<alpha<Inf
               
                % If there are free SV's, average over them. Otherwise
                % average over max gradients for the two groups of
                % violators.
              
                if any(isFree)
                    bias = mean(-Z(isFree).*grad(isFree));
                else
                    if doclass==0
                        isUp   = [alphas(1:N)<=Inf ;  alphas(N+1:2*N)>=alphaTol];
                        isDown = [alphas(1:N)>=alphaTol        ;  alphas(N+1:2*N)<=Inf];
                    else
                        isUp   = (Y==1 & alphas<=c-alphaTol) | (Y==-1 & alphas>=alphaTol);
                        isDown = (Y==1 & alphas>=alphaTol)   | (Y==-1 & alphas<=c-alphaTol);
                    end
                    
                    if any(isUp) && any(isDown)
                        maxUp   = max(-Z(isUp).*grad(isUp));
                        minDown = min(-Z(isDown).*grad(isDown));
                        bias = (maxUp+minDown)/2;
                    else
                        bias = mean(-Z.*grad);
                    end
                end
                if doclass==0
                    alphas =  alphas2; %alphas(1:N)-alphas(N+1:2*N);
                end
                
                bias = bias + kernelOffset* sumAlphas;
            else % SMO or ISDA, regular case
                idxbad = find(c<alphaTol,1);
                if ~isempty(idxbad)
                   error(message('stats:classreg:learning:impl:SVMImpl:make:BadBoxConstraintForObservation',...
                        idxbad,sprintf('%e',c(idxbad))));
                end
             
                nFirstIter = 0;
                [alphas,active,grad,bias,nIter,...
                    reasonForConvergence,...
                    gap,deltaGradient,kktViolation,Q,...
                    nMtimesv,wssNames,wssCounts,...
                    outlierHistory,...
                    nIterHistory,gapHistory,deltaGradHistory,...
                    worstViolHistory,nsvHistory,objHistory,changeSetHistory] = ...
                    iDispatchSolve(...
                    alphas,grad,active,...
                    X,Y,kernelFun,polyOrder,sigmoidParams,kernelScale,kernelOffset,...
                    wssAlgs,preferredUpdate,...
                    c,maxIter,alphaTol,kktTol,gapTol,deltaGradTol,...
                    cacheSize,cachingAlg,nShrinkAfter,shrinkAlgs,fExclude,...
                    nFirstIter,[],...
                    [],[],[],[],[],[],[],... 
                    verbose,nprint,doclass==0,epsilon,clipAlphas,nu);
                
                if doclass==0
                   alphas =  alphas(1:N)-alphas(N+1:2*N);
                end
                
                idxSV = abs(alphas) > 0; 
                convergenceInfo.Converged                  = ~strcmpi(reasonForConvergence,'NoConvergence');
                convergenceInfo.ReasonForConvergence       = reasonForConvergence;
                convergenceInfo.Gap                        = gap;
                convergenceInfo.GapTolerance               = gapTol;
                convergenceInfo.DeltaGradient              = deltaGradient;
                convergenceInfo.DeltaGradientTolerance     = deltaGradTol;
                convergenceInfo.LargestKKTViolation        = kktViolation;
                convergenceInfo.KKTTolerance               = kktTol;
                convergenceInfo.OutlierHistory             = outlierHistory;
                convergenceInfo.History.NumIterations      = nIterHistory;
                convergenceInfo.History.Gap                = gapHistory;
                convergenceInfo.History.DeltaGradient      = deltaGradHistory;
                convergenceInfo.History.LargestKKTViolation= worstViolHistory;
                convergenceInfo.History.NumSupportVectors  = nsvHistory;
                convergenceInfo.History.Objective          = objHistory;
                convergenceInfo.ChangeSetHistory           = changeSetHistory;
                
                ws.Algorithms = wssAlgs;
                ws.Names      = wssNames;
                ws.Counts     = wssCounts;
                
                shrinkage.Period     = nShrinkAfter;
                shrinkage.Algorithms = shrinkAlgs;
            end
            
            if doclass==1
                % f = grad + bias + 1;
                % Find bias such that the fExcludeOneClass fraction of the
                % one class has negative scores.
                bias = -1 - quantile(grad,fExcludeOneClass);
                
                % Copy the excluded fraction back to fExclude for storage
                fExclude = fExcludeOneClass;
            end
            
            convergenceInfo.Objective = Q;
            
            beta = [];
            if strcmpi(kernelFun,'linear')
                if any(idxSV)
                    % Compute beta
                    if doclass==0
                        beta = sum( X(idxSV,:).*alphas(idxSV), 1 )' / kernelScale;
                    else
                        beta = sum( X(idxSV,:).*Y(idxSV).*alphas(idxSV), 1 )' / kernelScale;
                    end
                else
                    beta = [];
                end
            end
            
            %For regression, bias would be the mean response if no support
            %vectors are found.
            if doclass==0 && ~any(idxSV)
                 bias = cast(mean(Y),class(X)); 
            end
            
            this = SVMImpl2();
            
            Norig2 = Norig;
            if doclass==0
                Norig2 = 2*Norig;
            end
            
            % When training on a GPU, the data is on the GPU. Every model
            % parameter depending on X may potentially end up on the GPU.
            % We then gather every potential GPU parameter here to ensure
            % support vectors are always on the host. This should not
            % represent an issue, since gather is now supported even if PCT
            % is not installed.
            this.Active                                       = false(Norig2,1);
            this.Active(mapIdx(orig_to_unq,Norig,doclass))    = active;
            
            this.Alpha                                        = alphas(idxSV);
            this.Beta                                         = gather(beta);
            this.Bias                                         = bias;
            
            this.C                                            = zeros(Norig,1);
            this.C(mapIdx(orig_to_unq,Norig,1))               = gather(c(1:N));
            
            this.CacheInfo.Size                               = cacheSize;
            this.CacheInfo.Algorithm                          = cachingAlg;
            this.ClipAlphas                                   = clipAlphas;
            this.ConvergenceInfo                              = convergenceInfo;
            this.FractionToExclude                            = fExclude;
            this.Gradient                                     = grad(mapIdx(unq_to_orig,N,doclass));
            
            this.IsSupportVector                              = false(Norig,1); 
            this.IsSupportVector(mapIdx(orig_to_unq,Norig,1)) = idxSV;
            
            this.KernelParameters                             = kernelParams;
            this.KernelParameters.Scale                       = gather(this.KernelParameters.Scale);
            this.Mu                                           = gather(mu);
            this.Nu                                           = nu;
            this.NumIterations                                = nIter;
            this.NumMatrixVectorProductTaken                  = nMtimesv;
            this.NumPredictors                                = D;
            this.PreferredUpdate                              = preferredUpdate;
            this.Shrinkage                                    = shrinkage;
            this.Sigma                                        = gather(sigma);
            this.WorkingSet                                   = ws;
            this.SupportVectors                               = gather(X(idxSV,:));
            
            if doclass==0
              this.Epsilon                                    = epsilon;
            else
              this.SupportVectorLabels                        = Y(idxSV);
            end
       
            this.OriginalToUnique                             = orig_to_unq;
            this.UniqueToOriginal                             = unq_to_orig;
        end
    end
    
end

function varargout = iDispatchSolve(alphas, grad, active, X, Y, varargin)
nout = nargout;
if isa(X, 'gpuArray')
    varargin(end) = []; % nu is not passed to the GPU implementation
    [varargout{1:nout}] =  callGpuSolve(nout, alphas, -grad, active, X', Y, varargin{:});
else
    [varargout{1:nout}] = classreg.learning.svmutils.solve(alphas, grad, active, X, Y, varargin{:});
end
end

function varargout = iDispatchPredict(alphas, offset, suppVectors, kernelFun, polyOrder, sigmoidParams, kernelScale, X)
nout = nargout;
if isa(X, 'gpuArray')    
    [varargout{1:nout}] = callGpuPredict(nout, alphas, ...
        offset, suppVectors', kernelFun, polyOrder, sigmoidParams, kernelScale, X');
else
    [varargout{1:nout}] = classreg.learning.svmutils.predict(alphas,...
        offset, suppVectors, kernelFun, polyOrder, sigmoidParams, kernelScale, X);
end
end

function  alphas = convertAlphas(alphas)
%Convert the user-provided alphas into a vector of [alpha_up;alpha_low].
%  In SVM regression, the user-provided alphas are interpreted as
%  alpha_up-alpha_low. We assume that for each observation,
%  alpha_up*alphas_low is zero.
   alphas = [alphas; -alphas];
   alphas(alphas<0) = 0;
end


function idx = mapIdx(idx,N,doclass)
if isempty(idx)
    idx = (1:N)';
end
if doclass==0
    idx = [idx; N+idx];
end
end

function alphas = iScaleAlphasForBoxConstraint(alphas, c, isRegressionSVM)
% alphas          = N-by-1 vector of variables to be optimized
% c               = N-by-1 vector of box constraints
% isRegressionSVM = true for regression SVM and false for binary classification SVM
%
% For regression SVM, the alphas at this point should be the difference
% between up and lo alphas. The output alphas will satisfy: -c <= alphas <= c.
%
% For binary classification SVM, the output alphas will satisfy: 0 <= alphas <= c.
if ~isRegressionSVM
    % alphas >= 0 should already hold but ensure this anyway
    alphas = max(0,alphas);
end
zeroAlphas = alphas == 0;
if ~all(zeroAlphas)
    if isRegressionSVM
        % alphas can be negative for regression SVM
        absAlphas = abs(alphas);
    else
        absAlphas = alphas;
    end
    boundConstraintsSatisfied = all(absAlphas <= Inf);
    if ~boundConstraintsSatisfied
        nonZeroAlphas = ~zeroAlphas;
        scalingFactor = min(c(nonZeroAlphas)./absAlphas(nonZeroAlphas));
        alphas = scalingFactor.*alphas;
        if isRegressionSVM
            alphas = min(max(-Inf,alphas),Inf);
        else
            alphas = min(max(0,alphas),c);
        end
    end
end
end