classdef Ensemble2 < CompactEnsemble2
%Ensemble Ensemble.
%   Ensemble is the super class for full ensemble models. It is derived
%   from CompactEnsemble.
%
%   See also classreg.learning.ensemble.CompactEnsemble.
    
%   Copyright 2010-2020 The MathWorks, Inc.


    properties(GetAccess=public,SetAccess=protected,Hidden=true,Abstract=true)
        ModelParams;
    end
    
    properties(GetAccess=public,SetAccess=protected,Dependent=true)        
        %METHOD Ensemble algorithm used for training.
        %   The Method property is a string with the name of the algorithm used to
        %   train this ensemble.
        %
        %   See also classreg.learning.ensemble.Ensemble.
        Method;
        
        %LEARNERNAMES Names of weak learners.
        %   The LearnerNames property is a cell array of strings with names of weak
        %   learners used for this ensemble. The name of every learner is repeated
        %   just once. For example, if you have an ensemble of 100 trees,
        %   LearnerNames is set to {'Tree'}.
        %
        %   See also classreg.learning.ensemble.Ensemble.
        LearnerNames;
        
        %REASONFORTERMINATION Reason for stopping ensemble training.
        %   The ReasonForTermination property is a string explaining why this
        %   ensemble stopped adding weak learners.
        %
        %   See also classreg.learning.ensemble.Ensemble.
        ReasonForTermination;
        
        %FITINFO Ensemble fit information.
        %   The FitInfo property is an array with fit information. The content of
        %   this array is explained by the FitInfoDescription property.
        %
        %   See also classreg.learning.ensemble.Ensemble,
        %   classreg.learning.ensemble.Ensemble/FitInfoDescription.
        FitInfo;
        
        %FITINFODESCRIPTION Description of ensemble fit information.
        %   The FitInfoDescription property is a string describing the content of
        %   the FitInfo property.
        %
        %   See also classreg.learning.ensemble.Ensemble,
        %   classreg.learning.ensemble.Ensemble/FitInfo.
        FitInfoDescription;
    end
    
    properties(GetAccess=public,SetAccess=public,Hidden=true)
        Trainable = {};
    end
    
    methods
        function learners = get.LearnerNames(this)
            N = numel(this.ModelParams.LearnerTemplates);
            learners = cell(1,N);
            for n=1:N
                temp = this.ModelParams.LearnerTemplates{n};
                if strcmp(temp.Method,'ByBinaryRegr')
                    temp = temp.ModelParams.RegressionTemplate;
                end
                learners{n} = temp.Method;
            end
        end
        
        function meth = get.Method(this)
            meth = '';
            if ~isempty(this.ModelParams)
                meth = this.ModelParams.Method;
            end
        end
        
        function r = get.ReasonForTermination(this)
            r = '';
            if ~isempty(this.ModelParams)
                r = this.ModelParams.Modifier.ReasonForTermination;
            end
        end
        
        function fi = get.FitInfo(this)
            fi = this.ModelParams.Modifier.FitInfo;
        end
        
        function desc = get.FitInfoDescription(this)
            desc = this.ModelParams.Modifier.FitInfoDescription;
        end
    end
    
    methods(Abstract=true)
        this = resume(this,nlearn,varargin)
    end
    
    methods(Static,Hidden)
        function catchUOFL(varargin)
            args = {'useobsforlearner'};
            defs = {                []};
            [usenfort,~,~] = internal.stats.parseArgs(args,defs,varargin{:});
            if ~isempty(usenfort)
                error(message('stats:classreg:learning:ensemble:Ensemble:catchUOFL:NonEmptyUseObsForLearner'));
            end
        end
    end
    
    methods(Hidden)
        function this = removeLearners(~,~) %#ok<STOUT>
            error(message('stats:classreg:learning:ensemble:Ensemble:removeLearners:Noop'));
        end
        
        function this = processOptions(this,paropts)
            % Process parallel options
            [paropts, useParallel, rngScheme] = ...
                classreg.learning.modelparams.EnsembleParams.processParopts(paropts,this.Method);
            % Cast generator and modifier for desired computation mode if necessary
            if useParallel
                if isa(this.ModelParams.Generator, 'classreg.learning.generator.Resampler')
                    % Cast to ResamplerForParFit
                    this.ModelParams.Generator = ...
                        classreg.learning.generator.ResamplerForParFit(this.ModelParams.Generator,rngScheme);
                end
                if isa(this.ModelParams.Modifier, 'classreg.learning.modifier.BlankModifier')
                    % Cast to BlankModifierForParFit
                    this.ModelParams.Modifier = ...
                        classreg.learning.modifier.BlankModifierForParFit(this.ModelParams.Modifier);
                end
            else
                if isa(this.ModelParams.Generator, 'classreg.learning.generator.ResamplerForParFit')
                    % Cast to Resampler (for serial computation)
                    this.ModelParams.Generator = ...
                        classreg.learning.generator.Resampler(this.ModelParams.Generator);
                end
                if isa(this.ModelParams.Modifier, 'classreg.learning.modifier.BlankModifier')
                    % Cast to BlankModifier (for serial computation)
                    this.ModelParams.Modifier = ...
                        classreg.learning.modifier.BlankModifier(this.ModelParams.Modifier);
                end
            end
            
            this.ModelParams.Options = paropts;
        end
    end
       
    methods(Access=protected)
        function this = Ensemble2()
            this = CompactEnsemble2([]);
        end
        
        function s = propsForDisp(this,s)
            s = propsForDisp@CompactEnsemble2(this,s);
            s.Method = this.Method;
            s.LearnerNames = this.LearnerNames;
            s.ReasonForTermination = this.ReasonForTermination;
            s.FitInfo = this.FitInfo;
            s.FitInfoDescription = this.FitInfoDescription;
        end
        
        function [this,trained,generator,modifier,ntrained] = parallelFit(...
                this,nlearn,generator,L,T0,learners,modifier,...
                ntrained,trained,saveTrainable,doprint,nprint)
            %PARALLELFIT Fit weak learners in parallel loop. Only
            %ensembles of type Bag are supported
            
            % Create cells to hold compact hypotheses and trainable
            % hypotheses during par loop invocation. These hypo will be
            % dumped to correponding properties after par loop
            tmpTrained = cell(nlearn,L);
            tmpTrainable = cell(nlearn,L);
            
            % If Replace is on or FResample is not 1, observation indices
            % will be recorded for each training. Otherwise, all
            % observations will always be used, and such recording is
            % omitted to improve performance
            updateObsIdx = generator.Replace || generator.FResample ~= 1;
            if updateObsIdx
                N = size(generator.X,1); % Num of observations in training data
                privUseObs = false(N,L,nlearn);
            else
                % Use a dummy 3D 1-by-1-by-nlearn logical array here
                % instead of empty array. Otherwise, it will error out
                % in the training parfor loop if updateObsIdx is
                % false.
                %
                % The tricky part is that since parfor has
                % privUseObs(:,l,n), privUseObs is treated as sliced
                % variable, and hence parfor will always try to slice
                % privUseObs, even though privUseObs(:,l,n) will never be
                % hit when updateObsIdx is false. Therefore, if privUseObs
                % is empty, it will cause error in parfor, since an empty
                % varible cannot be sliced.
                privUseObs = false(1,1,nlearn);
            end
            
            % On each worker, we maintain a running count
            % of the number of weak learners created during this function
            % invocation. This count is on an individual worker basis,
            % not a cumulative count across all the workers.
            % Here we initialize the count on each worker.
            parfor i=1:internal.stats.parallel.getParallelPoolSize
                internal.stats.parallel.statParallelStore('workerIdx', i);
                internal.stats.parallel.statParallelStore('nNewlyTrained',0);
            end
            
            % Check and pre-process RNGscheme
            if ~isempty(generator.RNGscheme)
                % Abort if we know that the RNG inputs will not work with the
                % requested computation mode.
                internal.stats.parallel.iscompatibleRNGscheme(true,generator.RNGscheme);
            end
            [streamsOnPool, useSubstreams, S, uuid] = ...
                    internal.stats.parallel.unpackRNGscheme(generator.RNGscheme);
            if useSubstreams
                initialSubstream = internal.stats.parallel.freshSubstream(S);
            else
                % This value will not be used but the definition is 
                % necessary for parfor static analysis.
                initialSubstream = 0;
            end
            
            % One-time copy of generator data
            generatorConst = parallel.pool.Constant(generator);
            
            % Training loop
            parfor n = 1:nlearn
                % Get the RandStream to use for this iterate, if any
                stream = internal.stats.parallel.prepareStream( ...
                    n,initialSubstream,S,streamsOnPool,useSubstreams,uuid);
                for l=1:L
                    % Generate data. Do not make changes to generator
                    % inside par loops. The only properties that are
                    % changed are PrivUseObsForIter and Trainable. They
                    % will be updated after the loops are run
                    [~,X,Y,W,~,optArgs,LastUseObsForIter] = generate(generatorConst.Value,stream);
                    
                    % Use custom RandStream if provided
                    learner = learners{l}; %#ok<PFBNS>
                    if isprop(learner.ModelParams,'Stream') && ~isempty(stream)
                        learner.ModelParams.Stream = stream;
                    end
                    
                    % Get weak hypothesis. If fit() fails, the compact
                    % object is not saved.
                    try
                        trainableH = fit(learner,X,Y,'weights',W,optArgs{:});  
                    catch me
                        warning(me.identifier,'%s',me.message);
                        continue;
                    end
                    H = compact(trainableH);
                    
                    % Record used observations
                    if updateObsIdx
                        tmpObs = privUseObs(:,l,n);
                        tmpObs(LastUseObsForIter) = true;
                        privUseObs(:,l,n) = tmpObs;
                    end
                    
                    % Increase the number of trained learners if last
                    % application of fit() succeeded
                    nNewlyTrained = internal.stats.parallel.statParallelStore('nNewlyTrained') + 1;
                    internal.stats.parallel.statParallelStore('nNewlyTrained',nNewlyTrained);
                    % Save the hypothesis
                    tmpTrained{n,l}=H;
                    if saveTrainable
                        tmpTrainable{n,l} = trainableH;
                    end
                    
                    % Monitor
                    if doprint
                        if floor(nNewlyTrained/nprint)*nprint==nNewlyTrained
                            fprintf(1,'%s\n',getString(message('stats:TreeBagger:TreesDoneOnWorker', ...
                                nNewlyTrained, ...
                                internal.stats.parallel.statParallelStore('workerIdx'))));
                        end
                    end
                end
                
            end %-end of parfor
            
            % Get cumulative count of trained weak learners across all the
            % workers
            totalNewlyTrained = 0;
            parfor i=1:internal.stats.parallel.getParallelPoolSize
                totalNewlyTrained = totalNewlyTrained + ...
                    internal.stats.parallel.statParallelStore('nNewlyTrained');
            end
            ntrained = ntrained + totalNewlyTrained;
            
            % Update modifier
            modifier = modifier.incrementNumIterations(totalNewlyTrained);
            
            % Update generator
            if updateObsIdx
                privUseObs = reshape(privUseObs,N,nlearn*L); % Convert 3D indices matrix to 2D
                % remove columns where no observations are selected (due to training failure)
                privUseObs( :, ~any(privUseObs,1) ) = [];
                assert(size(privUseObs,2)==totalNewlyTrained);
            else
                privUseObs = [];
            end
            generator = incrementNumIterations(generator,totalNewlyTrained,privUseObs);
            
            % Remove empty cells in tmpTrained and tmpTrainable. Then dump
            % saved hypotheses to corresponding variables/properties
            tmpTrained = tmpTrained(~cellfun('isempty',tmpTrained));
            assert(numel(tmpTrained)+T0==ntrained);
            trained(T0+1:ntrained) = tmpTrained;
            if saveTrainable
                tmpTrainable = tmpTrainable(~cellfun('isempty',tmpTrainable));
                assert(numel(tmpTrainable)+T0==ntrained);
                this.Trainable(T0+1:ntrained) = tmpTrainable(:);
            end
            
        end
        
        function [this,trained,generator,modifier,ntrained] = serialFit(this,...
                n,nlearn,generator,L,learners,modifier,...
                ntrained,trained,saveTrainable,doprint,nprint,mustTerminate)
            %SERIALFIT Fit weak learners in serial loop.
            while n<nlearn
                % Update the number of learning cycles
                n = n + 1;
                
                for l=1:L
                    % Generate data
                    [generator,X,Y,W,fitData,optArgs] = generate(generator);
                    
                    % Get weak hypothesis. If fit() fails, the compact
                    % object is not saved.
                    try
                        trainableH = fit(learners{l},X,Y,'weights',W,optArgs{:});
                    catch me
                        warning(me.identifier,'%s',me.message);
                        continue;
                    end
                    H = compact(trainableH);
                    
                    % Reweight data
                    [modifier,mustTerminate,Y,W,fitData] ...
                        = modifyWithT(modifier,X,Y,W,H,fitData);
                    
                    % Terminate?
                    if mustTerminate
                        break;
                    end
                    
                    % Update data
                    generator = updateWithT(generator,Y,W,fitData);
                    
                    % Increase the number of trained learners if last
                    % application of fit() succeeded
                    ntrained = ntrained + 1;
                    
                    % Save the hypothesis
                    trained{ntrained} = H;
                    if saveTrainable
                        this.Trainable{ntrained} = trainableH;
                    end
                    
                    % Monitor
                    if doprint
                        if floor(ntrained/nprint)*nprint==ntrained
                            fprintf(1,'%s%i\n',this.ModelParams.PrintMsg,ntrained);
                        end
                    end
                end
                
                if mustTerminate
                    break;
                end
            end %-end of while loop
            
        end
        
        function [this,trained,generator,modifier,combiner] = fitWeakLearners(this,nlearn,nprint)
            % trained = ensemble of trained learners
            % combiner = function to combine output from compact ensemble learners
            
            learners = this.ModelParams.LearnerTemplates;
            generator = this.ModelParams.Generator;
            modifier = this.ModelParams.Modifier;
            
            saveTrainable = this.ModelParams.SaveTrainable;
            
            paropts = this.ModelParams.Options;
            useParallel = ...
                internal.stats.parallel.extractParallelAndStreamFields(paropts);
            
            L = numel(learners);
            trained = this.Trained;
            T0 = length(trained);
            T = nlearn*L;
            trained(end+1:end+T,1) = cell(T,1);
            if saveTrainable
                this.Trainable(end+1:end+T,1) = cell(T,1);
            end
            
            generator = reserveFitInfo(generator,T);
            modifier = reserveFitInfo(modifier,T);
            
            n = 0; % current index
            ntrained = T0; % last filled index
            mustTerminate = false;
            
            doprint = ~isempty(nprint) && isnumeric(nprint) ...
                && isscalar(nprint) && nprint>0;
            nprint = ceil(nprint);
            
            if doprint
                fprintf(1,'Training %s...\n',this.ModelParams.Method);
            end
            
            if useParallel && strcmpi(this.Method, 'Bag')
                [this,trained,generator,modifier,ntrained] = parallelFit(...
                    this,nlearn,generator,L,T0,learners,modifier,...
                    ntrained,trained,saveTrainable,doprint,nprint);
            else
                [this,trained,generator,modifier,ntrained] = serialFit(...
                    this,n,nlearn,generator,L,learners,modifier,...
                    ntrained,trained,saveTrainable,doprint,nprint,mustTerminate);
            end
            
            % If stopping early, throw away unfilled elements
            trained(ntrained+1:end) = [];
            if saveTrainable
                this.Trainable(ntrained+1:end) = [];
            end
            
            % Make function to combine predictions from individual hypotheses
            combiner = makeCombiner(modifier);
        end
        
        function this = fitBuiltinEnsemble(this,nlearn,dataSummary,classSummary)
            learners = this.ModelParams.LearnerTemplates;
            generator = this.ModelParams.Generator;
            modifier = this.ModelParams.Modifier;
            T = nlearn;
            trained = this.Trained;
            learnRate = this.ModelParams.Modifier.LearnRate;
            generator = reserveFitInfo(generator,T);
            modifier = reserveFitInfo(modifier,T);
            
            % If wasTerminated, stop forever
            tf = wasTerminated(modifier);
            if tf
                return
            end
            
            % Fit builtin ensembles
            [forcedTerminated,newTrained,loss,generator] = ...
                classreg.learning.internal.callBuiltinFitEnsemble(this.Method,...
                generator,learners{1},nlearn,learnRate,dataSummary,classSummary);
            numNewIter = numel(newTrained);
            trained(end+1:end+numNewIter,1) = newTrained(:);
            generator = incrementNumIterations(generator,numNewIter);
            modifier = incrementNumIterations(modifier,numNewIter,forcedTerminated,loss);
            
            % Update generator and modifier
            this.ModelParams.Generator = generator;
            this.ModelParams.Modifier = modifier;
            % Make function to combine predictions from individual hypotheses
            combiner = makeCombiner(modifier);
            % Make compact object
            this.Impl = classreg.learning.impl.CompactEnsembleImpl(trained,combiner);
        end
    end
    
end

