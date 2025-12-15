%==========================================================================
%  Multi-Benchmark Runner + Automated Parallel Cluster Setup
%  MATLAB R2024a
%  Muhammad Asad Lal  
%==========================================================================
clc;
clear;
close all;
addpath(genpath(fullfile(pwd, 'Algorithms')));
%% ---------------- User settings ----------------
% Parallel computing settings
parallel_on = true;  % Set to true to enable parallel processing (ON/OFF switch)
num_workers = 8;    % Set the desired number of workers for THIS RUN
max_cluster_limit = num_workers; % Set the maximum worker limit to save to the cluster profile. 
                        % This must be >= num_workers.

% Benchmarks available: 'fun_info', 'CEC2017', 'Func_eng'
benchName        = 'CEC2017';            % choose benchmark
% Select functions once using numeric ids (mix singles + ranges)
%Fun_ids          = 1:23;
Fun_ids          = [1, 3:30];
Fun_list         = arrayfun(@(k) sprintf('F%d',k), Fun_ids, 'UniformOutput', false);
%Fun_list        = {'PressureVesselDesign','SpringDesign','ThreeBarTruss','GearTrainDesign','CantileverBeam'};
Search           = 50;
Max_iterations   = 1000;
num_runs         = 30;                    % runs per function (target total)
stats_after_runs = 30;                    % add summary stats after >= this many runs
dense_rank_tol   = 0;                     % 0 strict; >0 treats near-equal as equal
% --- NEW: tail window as percentage of iterations ---
REP_TAIL_PCT     = 0.15;                  % last 10% of iterations
MIN_TAIL_K       = 30;                    % ensure at least 30 points if available

%% ---------------- Parallel Pool Setup (Integrated) ----------------
% Internal flag: initialize based on user setting
use_parallel = parallel_on; 
effective_workers = 1;

if use_parallel
    try
        % 1. Get the local cluster profile object
        cluster = parcluster();
        
        % 2. AUTOMATICALLY ADJUST THE CLUSTER LIMIT IF IT'S TOO LOW
        % This is the fix for "Too many workers requested"
        current_limit = cluster.NumWorkers;
        
        if current_limit < max_cluster_limit
            fprintf('Warning: Local cluster limit (%d) is less than required maximum (%d).\n', ...
                current_limit, max_cluster_limit);
            % Set the new maximum limit
            cluster.NumWorkers = max_cluster_limit; 
            % Save the change permanently
            cluster.saveProfile();
            fprintf('Cluster limit permanently updated to %d.\n', max_cluster_limit);
        end
        
        % 3. Check if a pool is already running
        pool = gcp('nocreate');
        if isempty(pool)
            % Start the pool with the specified number of workers
            % Since we checked and adjusted the cluster limit above, this should succeed.
            pool = parpool(cluster, num_workers); 
        end
        
        % 4. Confirm the effective worker count
        effective_workers = pool.NumWorkers;
        fprintf('Parallel execution enabled. Pool is running with %d effective workers.\n', effective_workers);
        
    catch ME
        % Handle any failure (e.g., if parpool cannot be found or started)
        warning('Custom:ParallelFail', 'Failed to start parallel pool: %s\nRunning sequentially.', ME.message);
        use_parallel = false; % Disable parallel if pool fails to start
    end
end

if ~use_parallel
    fprintf('Running sequentially (Parallel switch is OFF or setup failed).\n');
end
%% ---------- Select the function-info provider for the benchmark ----------
switch benchName
    case 'fun_info'; info_handle = @fun_info;
    case 'CEC2017';  info_handle = @CEC2017;
    case 'Func_eng'; info_handle = @Func_eng;
    otherwise; error('Unknown benchmark: %s', benchName);
end
%% ========================== Results folders (per benchmark) ==============
resultsRoot   = fullfile(pwd, 'Results and Graphs');
benchRoot     = fullfile(resultsRoot, benchName);
excelFilePath = fullfile(benchRoot, [benchName '.xlsx']);
% REMOVED: Separate decision variables file - now stored in main file
graphsRoot    = fullfile(benchRoot, 'Graphs');
if ~exist(resultsRoot, 'dir'), mkdir(resultsRoot); end
if ~exist(benchRoot  , 'dir'), mkdir(benchRoot);   end
if ~exist(graphsRoot , 'dir'), mkdir(graphsRoot);  end
warning('off', 'MATLAB:xlswrite:AddSheet');
% NEW: make sure Excel is not locking this workbook (closes if open)
ensureExcelClosedAndUnlocked(excelFilePath);
%% ========================== Header messages ==============================
fprintf('Starting benchmark: %s\n', benchName);
fprintf('Functions: %s\n', strjoin(Fun_list, ', '));
fprintf('Target runs per function: %d\n', num_runs);
if strcmp(benchName, 'Func_eng')
    fprintf('NOTE: Storing decision variables in main Excel file after Rank column.\n');
end
fprintf('%s\n\n', repmat('-',1,71));
%% ========================== Iterate functions ============================
for f = 1:numel(Fun_list)
    Fun_name  = Fun_list{f};
    sheetName = Fun_name;
    % per-function graphs folder
    funcDir = fullfile(graphsRoot, Fun_name);
    if ~exist(funcDir, 'dir'), mkdir(funcDir); end
    % keep all runs here
    runDir = fullfile(funcDir, 'AllRuns');
    if ~exist(runDir, 'dir'), mkdir(runDir); end
    
    % Arrays to decide representative AFTER all runs (pre-allocate for performance)
    max_repData_size = num_runs; 
    repData = repmat(struct('run', 0, 'FIM_best', 0, 'minBest', 0, 'leadFrac', 0, ...
                            'tailMean', 0, 'firstBestIter', 0, 'gapFinal', 0, 'tailGap', 0, ...
                            'png', '', 'eps', '', 'pdf', '', 'isWin', false), 1, max_repData_size);
    repData_count = 0; % Counter for actual executed runs

    % NEW: Store FIM positions for all runs to find best run later (only for Func_eng)
    if strcmp(benchName, 'Func_eng')
        FIM_positions = cell(num_runs, 2); % Column 1: run number, Column 2: FIM position
        FIM_best_scores = nan(num_runs, 1); % Store FIM best scores to find optimal run
    end

    % problem info for this function (must be defined OUTSIDE loop)
    [lowerbound, upperbound, dimension, fitness] = info_handle(Fun_name);
    % --------- Determine existing completed runs for this sheet ----------
    existingRuns = 0;
    if isfile(excelFilePath)
        try
            T_exist = readtable(excelFilePath, 'Sheet', sheetName);
            runColsMask = startsWith(T_exist.Properties.VariableNames, 'Run_');
            existingRuns = sum(runColsMask);
        catch
        end
    end
    if existingRuns >= num_runs
        fprintf('■ %s — requested %d runs, already completed %d. Skipping.\n', ...
            Fun_name, num_runs, existingRuns);
        fprintf('%s\n\n', repmat('-',1,80));
        continue;
    end
    startRunIdx = existingRuns + 1;
    runsToDo = num_runs - existingRuns;
    runIndices = startRunIdx:num_runs;

    fprintf('▶ %s — continuing from Run %d up to %d (%d runs to do).\n', ...
        Fun_name, startRunIdx, num_runs, runsToDo);
    
    % Pre-allocate cell arrays to collect results from all algorithms
    % For Func_eng: store FIM position for decision variables, for others: only best scores
    if strcmp(benchName, 'Func_eng')
        resultsCell = cell(runsToDo, 15); % Extra column for FIM position
    else
        resultsCell = cell(runsToDo, 14); % Original: run index + 13 algorithm results
    end
    
    % Pre-allocate FIM_pos before the parfor loop
FIM_pos = cell(runsToDo, 1);  % A cell array to store decision variables for each run

% ====================== run remaining times for this function =========
% Select loop type based on user setting
if use_parallel
    parfor k = 1:runsToDo
        r = runIndices(k);  % Get the actual run index
        
        % Store results for this run
        if strcmp(benchName, 'Func_eng')
            run_r_results = cell(1, 15); % Extra space for FIM position
        else
            run_r_results = cell(1, 14); % Original
        end

        fprintf('→ [%s] Run %d/%d ... running\n', Fun_name, r, num_runs);

        % ---------- Run all algorithms ----------
        if strcmp(benchName, 'Func_eng')
            % Capture FIM position for decision variables
            [FIM_best, FIM_pos{k}, FIM_curve] = FIM(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        else
            [FIM_best, ~, FIM_curve] = FIM(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        end

        % Run other algorithms and store results
        [GA_best, ~, GA_curve] = GA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [TBLO_best, ~, TBLO_curve] = TBLO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [WSO_best, ~, WSO_curve] = WSO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [WO_best, ~, WO_curve] = WO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [GSA_best, ~, GSA_curve] = GSA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [WOA_best, ~, WOA_curve] = WOA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [GWO_best, ~, GWO_curve] = GWO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [AVOA_best, ~, AVOA_curve] = AVOA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [RSA_best, ~, RSA_curve] = RSA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [MPA_best, ~, MPA_curve] = MPA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [TSA_best, ~, TSA_curve] = TSA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [MVO_best, ~, MVO_curve] = MVO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);

        % ---------- ENSURE CURVES ARE CONSISTENT WITH BEST SCORES ----------
        FIM_curve = standardizeCurve(FIM_curve, FIM_best, Max_iterations);
        GA_curve = standardizeCurve(GA_curve, GA_best, Max_iterations);
        TBLO_curve = standardizeCurve(TBLO_curve, TBLO_best, Max_iterations);
        WSO_curve = standardizeCurve(WSO_curve, WSO_best, Max_iterations);
        WO_curve = standardizeCurve(WO_curve, WO_best, Max_iterations);
        GSA_curve = standardizeCurve(GSA_curve, GSA_best, Max_iterations);
        WOA_curve = standardizeCurve(WOA_curve, WOA_best, Max_iterations);
        GWO_curve = standardizeCurve(GWO_curve, GWO_best, Max_iterations);
        AVOA_curve = standardizeCurve(AVOA_curve, AVOA_best, Max_iterations);
        RSA_curve = standardizeCurve(RSA_curve, RSA_best, Max_iterations);
        MPA_curve = standardizeCurve(MPA_curve, MPA_best, Max_iterations);
        TSA_curve = standardizeCurve(TSA_curve, TSA_best, Max_iterations);
        MVO_curve = standardizeCurve(MVO_curve, MVO_best, Max_iterations);

        % Store all data in the temporary cell array
        run_r_results{1} = r;
        run_r_results{2} = struct('best', FIM_best, 'curve', FIM_curve);
        run_r_results{3} = struct('best', GA_best, 'curve', GA_curve);
        run_r_results{4} = struct('best', TBLO_best, 'curve', TBLO_curve);
        run_r_results{5} = struct('best', WSO_best, 'curve', WSO_curve);
        run_r_results{6} = struct('best', WO_best, 'curve', WO_curve);
        run_r_results{7} = struct('best', GSA_best, 'curve', GSA_curve);
        run_r_results{8} = struct('best', WOA_best, 'curve', WOA_curve);
        run_r_results{9} = struct('best', GWO_best, 'curve', GWO_curve);
        run_r_results{10} = struct('best', AVOA_best, 'curve', AVOA_curve);
        run_r_results{11} = struct('best', RSA_best, 'curve', RSA_curve);
        run_r_results{12} = struct('best', MPA_best, 'curve', MPA_curve);
        run_r_results{13} = struct('best', TSA_best, 'curve', TSA_curve);
        run_r_results{14} = struct('best', MVO_best, 'curve', MVO_curve);

        % For Func_eng: Store FIM position for decision variables
        if strcmp(benchName, 'Func_eng')
            run_r_results{15} = FIM_pos{k};  % Store the decision variables for each run
        end

        resultsCell(k, :) = run_r_results;  % Store the results for this run
    end % parfor

else % Sequential execution using standard for loop
    for k = 1:runsToDo
        r = runIndices(k); 
        %rng('shuffle');
        
        fprintf('→ [%s] Run %d/%d ... running\n', Fun_name, r, num_runs);
        
        % Store results for this run
        if strcmp(benchName, 'Func_eng')
            run_r_results = cell(1, 15); % Extra space for FIM position
        else
            run_r_results = cell(1, 14); % Original
        end

        % ---------- Run all algorithms ----------
        if strcmp(benchName, 'Func_eng')
            % Capture FIM position for decision variables
            [FIM_best, FIM_pos, FIM_curve] = FIM(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        else
            [FIM_best, ~, FIM_curve] = FIM(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        end

        % Run other algorithms and store results
        [GA_best, ~, GA_curve] = GA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [TBLO_best, ~, TBLO_curve] = TBLO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [WSO_best, ~, WSO_curve] = WSO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [WO_best, ~, WO_curve] = WO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [GSA_best, ~, GSA_curve] = GSA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [WOA_best, ~, WOA_curve] = WOA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [GWO_best, ~, GWO_curve] = GWO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [AVOA_best, ~, AVOA_curve] = AVOA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [RSA_best, ~, RSA_curve] = RSA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [MPA_best, ~, MPA_curve] = MPA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [TSA_best, ~, TSA_curve] = TSA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);
        [MVO_best, ~, MVO_curve] = MVO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness);

        % ---------- ENSURE CURVES ARE CONSISTENT WITH BEST SCORES ----------
        FIM_curve = standardizeCurve(FIM_curve, FIM_best, Max_iterations);
        GA_curve = standardizeCurve(GA_curve, GA_best, Max_iterations);
        TBLO_curve = standardizeCurve(TBLO_curve, TBLO_best, Max_iterations);
        WSO_curve = standardizeCurve(WSO_curve, WSO_best, Max_iterations);
        WO_curve = standardizeCurve(WO_curve, WO_best, Max_iterations);
        GSA_curve = standardizeCurve(GSA_curve, GSA_best, Max_iterations);
        WOA_curve = standardizeCurve(WOA_curve, WOA_best, Max_iterations);
        GWO_curve = standardizeCurve(GWO_curve, GWO_best, Max_iterations);
        AVOA_curve = standardizeCurve(AVOA_curve, AVOA_best, Max_iterations);
        RSA_curve = standardizeCurve(RSA_curve, RSA_best, Max_iterations);
        MPA_curve = standardizeCurve(MPA_curve, MPA_best, Max_iterations);
        TSA_curve = standardizeCurve(TSA_curve, TSA_best, Max_iterations);
        MVO_curve = standardizeCurve(MVO_curve, MVO_best, Max_iterations);

        % Store results
        run_r_results{1} = r;
        run_r_results{2} = struct('best', FIM_best, 'curve', FIM_curve);
        run_r_results{3} = struct('best', GA_best, 'curve', GA_curve);
        run_r_results{4} = struct('best', TBLO_best, 'curve', TBLO_curve);
        run_r_results{5} = struct('best', WSO_best, 'curve', WSO_curve);
        run_r_results{6} = struct('best', WO_best, 'curve', WO_curve);
        run_r_results{7} = struct('best', GSA_best, 'curve', GSA_curve);
        run_r_results{8} = struct('best', WOA_best, 'curve', WOA_curve);
        run_r_results{9} = struct('best', GWO_best, 'curve', GWO_curve);
        run_r_results{10} = struct('best', AVOA_best, 'curve', AVOA_curve);
        run_r_results{11} = struct('best', RSA_best, 'curve', RSA_curve);
        run_r_results{12} = struct('best', MPA_best, 'curve', MPA_curve);
        run_r_results{13} = struct('best', TSA_best, 'curve', TSA_curve);
        run_r_results{14} = struct('best', MVO_best, 'curve', MVO_curve);

        % For Func_eng: Store FIM position for decision variables
        if strcmp(benchName, 'Func_eng')
            run_r_results{15} = FIM_pos;  % Store the decision variables for the run
        end

        resultsCell(k, :) = run_r_results;  % Store the results for this run
    end % for loop ends
end % if-else ends

    
    % NEW: Find the best run for decision variables storage (only for Func_eng)
    if strcmp(benchName, 'Func_eng')
        bestRun = NaN;
        bestFIMScore = Inf;
        bestFIMPosition = [];
        
        % Collect FIM scores and positions from all runs to find the best one
        for k = 1:runsToDo
            r = resultsCell{k, 1};
            FIM_best = resultsCell{k, 2}.best;
            FIM_pos = resultsCell{k, 15};
            
            % Store for tracking
            FIM_best_scores(r) = FIM_best;
            FIM_positions{r, 1} = r;
            FIM_positions{r, 2} = FIM_pos;
            
            % Update best run if current run is better
            if FIM_best < bestFIMScore
                bestFIMScore = FIM_best;
                bestRun = r;
                bestFIMPosition = FIM_pos;
            end
        end
        fprintf('→ [%s] Best FIM score across all runs: %.6g (Run %d)\n', ...
            Fun_name, bestFIMScore, bestRun);
    end
    
    % Post-execution processing (plotting, Excel writing, etc.)
    for k = 1:runsToDo
        
        r = resultsCell{k, 1}; % Actual run number
        
        % Extract results (best scores and curves for all algorithms)
        FIM_best = resultsCell{k, 2}.best; FIM_curve = resultsCell{k, 2}.curve;
        GA_best  = resultsCell{k, 3}.best; GA_curve  = resultsCell{k, 3}.curve;
        TBLO_best= resultsCell{k, 4}.best; TBLO_curve= resultsCell{k, 4}.curve;
        WSO_best = resultsCell{k, 5}.best; WSO_curve = resultsCell{k, 5}.curve;
        WO_best  = resultsCell{k, 6}.best; WO_curve  = resultsCell{k, 6}.curve;
        GSA_best = resultsCell{k, 7}.best; GSA_curve = resultsCell{k, 7}.curve;
        WOA_best = resultsCell{k, 8}.best; WOA_curve = resultsCell{k, 8}.curve;
        GWO_best = resultsCell{k, 9}.best; GWO_curve = resultsCell{k, 9}.curve;
        AVOA_best= resultsCell{k, 10}.best; AVOA_curve= resultsCell{k, 10}.curve;
        RSA_best = resultsCell{k, 11}.best; RSA_curve = resultsCell{k, 11}.curve;
        MPA_best = resultsCell{k, 12}.best; MPA_curve = resultsCell{k, 12}.curve;
        TSA_best = resultsCell{k, 13}.best; TSA_curve = resultsCell{k, 13}.curve;
        MVO_best = resultsCell{k, 14}.best; MVO_curve = resultsCell{k, 14}.curve;
        
        % For Func_eng: Extract FIM position for decision variables
        if strcmp(benchName, 'Func_eng')
            FIM_pos = resultsCell{k, 15};
        end
        
        fprintf('→ [%s] Run %d/%d ... processing results\n', Fun_name, r, num_runs);

        % ---------- plot & save (per-run) ----------
        fig = figure('Visible','off'); hold on;
        semilogx(FIM_curve,'LineWidth',3,'Color','k','Marker','x','MarkerSize',2,'MarkerFaceColor','w');
        semilogx(GA_curve ,'LineWidth',2,'Color','m');
        semilogx(TBLO_curve,'LineWidth',2,'Color','b');
        semilogx(WSO_curve,'LineWidth',2,'Color','c');
        semilogx(WO_curve ,'LineWidth',2,'Color','r');
        semilogx(GSA_curve,'LineWidth',2,'Color','y');
        semilogx(WOA_curve,'LineWidth',2,'Color',[0.5,0.5,0]);
        semilogx(GWO_curve,'LineWidth',2,'Color',[0.1,0.2,0.5]);
        semilogx(AVOA_curve,'LineWidth',2,'Color',[0.5,0,0.5]);
        semilogx(RSA_curve,'LineWidth',2,'Color',[1,0.5,0]);
        semilogx(MPA_curve,'LineWidth',2,'Color',[0.1,0.5,1]);
        semilogx(TSA_curve,'LineWidth',2,'Color',[1,0,0.5]);
        semilogx(MVO_curve,'LineWidth',2,'Color',[0,0.5,0]);
        title('Convergence Curve Comparison');
        xlabel('Iterations','FontSize',16);
        ylabel('Best Score','FontSize',16);
        legend('FIM','GA','TBLO','WSO','WO','GSA','WOA','GWO','AVOA','RSA','MPA','TSA','MVO','Location','best');
        grid on; box on;
        % ----- tight page -----
        set(fig,'Units','inches');
        pos = get(fig,'Position');
        set(fig,'PaperUnits','inches', ...
                'PaperPosition',[0 0 pos(3) pos(4)], ...
                'PaperSize',[pos(3) pos(4)], ...
                'PaperPositionMode','manual', ...
                'Renderer','painters');
        % save per-run copies under AllRuns
        pngPath = fullfile(runDir, sprintf('%s_Run_%d.png', Fun_name, r));
        epsPath = fullfile(runDir, sprintf('%s_Run_%d.eps', Fun_name, r));
        pdfPath = fullfile(runDir, sprintf('%s_Run_%d.pdf', Fun_name, r));
        print(fig, pngPath, '-dpng',  '-r300');
        print(fig, epsPath, '-depsc', '-r300');
        print(fig, pdfPath, '-dpdf',  '-r300');
        
        % ---------- console summary & per-run rank-1 (FIXED) ----------
        algonames = {'FIM','GA','TBLO','WSO','WO','GSA','WOA','GWO','AVOA','RSA','MPA','TSA','MVO'};
        scores    = [FIM_best, GA_best, TBLO_best, WSO_best, WO_best, GSA_best, WOA_best, GWO_best, AVOA_best, RSA_best, MPA_best, TSA_best, MVO_best];
        fprintf('[%s] Run %d |\n', Fun_name, r);
        printAlgoLine(algonames, scores);   % robust line: numbers or "NA"
        runRanks  = denseRankVector(scores, dense_rank_tol); % 1 = best
        rank1_idx = (runRanks == 1);
        nWinners  = sum(rank1_idx);
        if nWinners == 1
            i = find(rank1_idx, 1);
            fprintf('Rank 1 this run: %s with score: %s\n', algonames{i}, fmtScore(scores(i)));
        else
            idx   = find(rank1_idx);
            pairs = arrayfun(@(k) sprintf('%s=%s', algonames{k}, fmtScore(scores(k))), idx, 'UniformOutput', false);
            fprintf('Rank 1 this run (ties): %s\n', strjoin(pairs, ', '));
        end
        
        % ---------- write results to Excel ----------
        ensureExcelClosedAndUnlocked(excelFilePath);  % NEW: close if open + wait until unlocked
        nextColName = sprintf('Run_%d', r);
        data = { ...
            'FIM', FIM_best;  'GA', GA_best;   'TBLO', TBLO_best; 'WSO', WSO_best; ...
            'WO',  WO_best;   'GSA', GSA_best; 'WOA',  WOA_best;  'GWO',  GWO_best; ...
            'AVOA', AVOA_best;'RSA', RSA_best; 'MPA',  MPA_best;  'TSA',  TSA_best; ...
            'MVO',  MVO_best};
        runTbl = cell2table(data, 'VariableNames', {'Algorithm','Score'});
        if isfile(excelFilePath)
            try
                T = readtable(excelFilePath, 'Sheet', sheetName);
                if ~ismember(nextColName, T.Properties.VariableNames)
                    T.(nextColName) = nan(height(T), 1);
                end
                for i = 1:height(runTbl)
                    alg = runTbl.Algorithm{i};
                    if any(strcmp(T.Algorithm, alg))
                        T{strcmp(T.Algorithm, alg), nextColName} = runTbl.Score(i);
                    else
                        row = T(1,:); row{1,:} = {[]};
                        row(1,:) = array2table(nan(1, width(T)));
                        row.Algorithm     = {alg};
                        row.(nextColName) = runTbl.Score(i);
                        T = [T; row]; %#ok<AGROW>
                    end
                end
                runColsMask = startsWith(T.Properties.VariableNames, 'Run_');
                runCount    = sum(runColsMask);
                if runCount >= stats_after_runs
                    R = T{:, runColsMask};
                    avgScores    = mean(R, 2, 'omitnan');
                    stdScores    = std (R, 0, 2, 'omitnan');
                    minScores    = min (R, [], 2, 'omitnan');
                    worstScores  = max (R, [], 2, 'omitnan');
                    medianScores = median(R, 2, 'omitnan');
                    T.Best    = minScores;
                    T.Average = avgScores;
                    T.Worst   = worstScores;
                    T.Std     = stdScores;
                    T.Median  = medianScores;
                    T.Rank    = denseRankVector(avgScores, dense_rank_tol);
                end
                writeWithRetry(T, excelFilePath, sheetName);
            catch
                runTbl.Properties.VariableNames = {'Algorithm', nextColName};
                writeWithRetry(runTbl, excelFilePath, sheetName);
            end
        else
            if ~exist(benchRoot, 'dir'), mkdir(benchRoot); end
            runTbl.Properties.VariableNames = {'Algorithm', nextColName};
            writeWithRetry(runTbl, excelFilePath, sheetName);
        end
        
        % ---------- Bold best in this run & Rank==1 & Average-for-Rank1 ----------
        try excelBoldBestInRunColumn(excelFilePath, sheetName, nextColName, dense_rank_tol); catch, end
        try excelBoldRankOnes(excelFilePath, sheetName); catch, end
        try excelBoldAverageForRankOnes(excelFilePath, sheetName); catch, end
        
        % ---------- Build representative metrics (store only; save later) ----------
        try
            % Is FIM a winner by final best?
            minBest = min(scores);
            isWin   = (abs(FIM_best - minBest) <= max(dense_rank_tol, 0));
            % Lead fraction across iterations: FIM <= best of others
            othersMat = alignCurvesPadNaN({GA_curve,TBLO_curve,WSO_curve,WO_curve,GSA_curve,WOA_curve,GWO_curve,AVOA_curve,RSA_curve,MPA_curve,TSA_curve,MVO_curve});
            FIMv      = FIM_curve(:);
            othersMin = nanmin(othersMat, [], 2);
            L         = min(numel(FIMv), numel(othersMin));
            leadFrac  = mean( FIMv(1:L) <= (othersMin(1:L) + dense_rank_tol) );
            % --- NEW tail window: percentage with minimum ---
            K = max(1, round(REP_TAIL_PCT * numel(FIMv)));
            K = min(numel(FIMv), max(K, min(MIN_TAIL_K, numel(FIMv))));
            % Tail means and gaps
            tailMeanFIM = mean(FIMv(end-K+1:end));
            tailMeanOTH = mean(othersMin(end-K+1:end));
            tailGap     = tailMeanFIM - tailMeanOTH;
            % Earliest iteration reaching its best
            [~, firstBestIter] = min(FIMv);
            % Store
            repData_count = repData_count + 1;
            repData(repData_count) = struct( ...
                'run', r, ...
                'FIM_best', FIM_best, ...
                'minBest',  minBest, ...
                'leadFrac', leadFrac, ...
                'tailMean', tailMeanFIM, ...
                'firstBestIter', firstBestIter, ...
                'gapFinal', FIM_best - minBest, ...
                'tailGap',  tailGap, ...
                'png', pngPath, 'eps', epsPath, 'pdf', pdfPath, ...
                'isWin', isWin );
        catch ME
            warning('Rep metrics failed (%s Run %d): %s', Fun_name, r, ME.message);
        end
        close(fig);
        fprintf('\n');
    end % runs
    
    % Trim repData to only include executed runs
    repData = repData(1:repData_count);
    
    % ================== NEW: Store Decision Variables in MAIN FILE (Func_eng only) ==================
    if strcmp(benchName, 'Func_eng') && ~isnan(bestRun)
        fprintf('→ [%s] Storing decision variables in main file for best run (Run %d) with FIM score: %.6g\n', ...
            Fun_name, bestRun, bestFIMScore);
        
        ensureExcelClosedAndUnlocked(excelFilePath);
        
        % Read existing table
        T = readtable(excelFilePath, 'Sheet', sheetName);
        
        % Get actual variable names for this function
        varNames = getActualVariableNames(Fun_name);
        
        % Add decision variable columns after Rank column if they don't exist
        rankColIdx = find(strcmp(T.Properties.VariableNames, 'Rank'), 1);
        
        if ~isempty(rankColIdx)
            % Remove any existing decision variable columns to avoid duplicates
            dvColMask = startsWith(T.Properties.VariableNames, 'DV_') | ismember(T.Properties.VariableNames, varNames);
            if any(dvColMask)
                T(:, dvColMask) = [];
            end
            
            % Add decision variables as new columns after Rank
            for dim = 1:dimension
                if dim <= length(varNames)
                    colName = varNames{dim};
                else
                    colName = sprintf('DV_%d', dim); % Fallback if no specific name
                end
                
                % Initialize column with NaN
                T.(colName) = nan(height(T), 1);
                
                % Fill FIM row with actual decision variable value
                fimRow = find(strcmp(T.Algorithm, 'FIM'), 1);
                if ~isempty(fimRow) && length(bestFIMPosition) >= dim
                    T{fimRow, colName} = bestFIMPosition(dim);
                end
            end
            
            % Write updated table back to Excel
            writeWithRetry(T, excelFilePath, sheetName);
            fprintf('→ [%s] Decision variables saved in main file for best run (Run %d)\n', Fun_name, bestRun);
        end
    end
    
    % ================== Decide representative AFTER all runs ==================
    repIdx = selectRepresentativeRun(repData, dense_rank_tol);
    if ~isnan(repIdx)
        bestRun = repData(repIdx).run;
        % Create target folder: Graphs/<Fun>/Run_<k>
        repFolder = fullfile(funcDir, sprintf('Run_%d', bestRun));
        if ~exist(repFolder, 'dir'), mkdir(repFolder); end
        % Copy and rename to <Fun>.<ext>
        copyfile(repData(repIdx).png, fullfile(repFolder, sprintf('%s.png', Fun_name)));
        copyfile(repData(repIdx).eps, fullfile(repFolder, sprintf('%s.eps', Fun_name)));
        copyfile(repData(repIdx).pdf, fullfile(repFolder, sprintf('%s.pdf', Fun_name)));
        % Insert PNG into the sheet
        try
            ensureExcelClosedAndUnlocked(excelFilePath); % NEW: ensure not locked before COM insert
            pngForExcel = fullfile(repFolder, sprintf('%s.png', Fun_name));
            if isfile(pngForExcel)
                excel = actxserver('Excel.Application');
                excel.Visible = false; excel.DisplayAlerts = false;
                wb = excel.Workbooks.Open(excelFilePath);
                ws = wb.Sheets.Item(sheetName);
                ws.Shapes.AddPicture(pngForExcel, 'msoFalse','msoCTrue', 100, 100, 400, 300);
                wb.Save(); wb.Close(); excel.Quit(); delete(excel);
            end
        catch ME
            warning(['Could not insert representative plot into Excel: ' ME.message]);
        end
        fprintf('✓ Finished %s | Representative: Run %d (lead=%.2f, tailMean=%.3g, gap=%.3g)\n', ...
            Fun_name, bestRun, repData(repIdx).leadFrac, repData(repIdx).tailMean, repData(repIdx).gapFinal);
    else
        fprintf('✓ Finished %s | No representative saved (no runs recorded?)\n', Fun_name);
    end
    % Re-apply bolding after final stats
    try
        excelBoldRankOnes(excelFilePath, sheetName);
        excelBoldAverageForRankOnes(excelFilePath, sheetName);
    catch
    end
    fprintf('%s\n\n', repmat('-',1,80));
end % functions

%% ---------------- Parallel Pool Teardown ----------------
% Close the parallel pool if it was opened
if exist('parpool', 'builtin') && parpool('size') > 0
    delete(gcp('nocreate'));
    fprintf('Parallel pool closed.\n');
end

fprintf('ALL FUNCTIONS COMPLETED.\n');
fprintf('Saved Excel: %s\n', excelFilePath);
if strcmp(benchName, 'Func_eng')
    fprintf('NOTE: Decision variables stored in main Excel file after Rank column.\n');
end
fprintf('Saved graphs under: %s\n', fullfile(benchRoot,'Graphs'));
warning('on', 'MATLAB:xlswrite:AddSheet');
%% ============================== NEW: Function to get actual variable names ==========================
function varNames = getActualVariableNames(functionName)
% Returns the actual variable names for each engineering function
    switch functionName
        case 'PressureVesselDesign'
            varNames = {'ts', 'th', 'r', 'l'};
        case 'WeldedBeam'
            varNames = {'W', 'L', 'd', 'h'};
        case 'SpringDesign'
            varNames = {'W', 'd', 'n'};
        case 'ThreeBarTruss'
            varNames = {'a1', 'a2'};
        case 'GearTrainDesign'
            varNames = {'n1', 'n2', 'n3', 'n4'};
        case 'CantileverBeam'
            varNames = {'y1', 'y2', 'y3', 'y4', 'y5'}; % No specific names in comments
        otherwise
            % Fallback for unknown functions
            varNames = arrayfun(@(k) sprintf('DV_%d', k), 1:10, 'UniformOutput', false);
    end
end
%% ============================== Local functions (Unchanged) ==========================
function ranks = denseRankVector(v, tol)
% Dense integer ranks (ascending). Equal => same rank. No gaps.
    v = v(:); ok = ~isnan(v);
    vAdj = v;
    if nargin >= 2 && tol > 0
        vAdj(ok) = round(v(ok)./tol) .* tol;
    end
    ranks = nan(size(vAdj));
    [~,~,d] = unique(vAdj(ok), 'sorted');
    ranks(ok) = d;
end
function writeWithRetry(T, filePath, sheetName)
% Robust writer that avoids Excel COM to reduce file locks.
% NEW: proactively close Excel and wait until the file is unlocked.
    ensureExcelClosedAndUnlocked(filePath);
    tries = 6; pauseSec = 0.5;
    for t = 1:tries
        try
            writetable(T, filePath, 'Sheet', sheetName, 'UseExcel', false);
            return;
        catch ME
            if t == tries
                rethrow(ME);
            else
                pause(pauseSec);
                ensureExcelClosedAndUnlocked(filePath);
            end
        end
    end
end
function ensureExcelClosedAndUnlocked(excelFullPath)
% Close the target workbook if Excel has it open, then wait until it is unlocked.
% Safe to call even if Excel is not running or the file does not exist.
    % Try to close if open in a running Excel instance
    try
        ex = actxGetRunningServer('Excel.Application');
        try
            % scan all open workbooks in that instance
            for k = ex.Workbooks.Count:-1:1
                try
                    wb = ex.Workbooks.Item(k);
                    % match by full path (preferred) or by name fallback
                    if (isprop(wb, 'FullName') && strcmpi(wb.FullName, excelFullPath)) || ...
                       (isprop(wb, 'Name')     && strcmpi(wb.Name,     string(java.io.File(excelFullPath).getName())))
                        wb.Close(false);
                    end
                catch
                end
            end
        catch
        end
        % If nothing left open, quit that instance
        try
            if ex.Workbooks.Count == 0
                ex.Quit();
            end
        catch
        end
        try delete(ex); catch, end
    catch
        % No running Excel instance registered, continue
    end
    % Wait until the file is not locked by any process
    maxWaitSec = 5;      % small, quick wait
    t0 = tic;
    while isFileLocked(excelFullPath)
        if toc(t0) > maxWaitSec
            % one more mild attempt: short sleep then break
            pause(0.25);
            break;
        end
        pause(0.15);
    end
end
function tf = isFileLocked(filePath)
% Probe whether file is locked for write using .NET FileStream.
    tf = false;
    if ~isfile(filePath)
        return;
    end
    try
        import System.IO.*
        fs = FileStream(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        fs.Close();
    catch
        tf = true;
    end
end
function excelBoldBestInRunColumn(filePath, sheetName, runColName, tol)
% Bold the best (minimum) value(s) in a given Run_* column on a sheet.
    if nargin < 4, tol = 0; end
    T = readtable(filePath, 'Sheet', sheetName);
    varNames = T.Properties.VariableNames;
    colIdx = find(strcmp(varNames, runColName), 1);
    if isempty(colIdx), return; end
    colVals = T{:, colIdx};
    dataMask = ~isnan(colVals);
    if ~any(dataMask), return; end
    vals = colVals(dataMask);
    vmin = min(vals);
    if tol > 0
        toBold = find(dataMask & abs(colVals - vmin) <= tol);
    else
        toBold = find(dataMask & (colVals == vmin));
    end
    excel = actxserver('Excel.Application');
    excel.Visible = false; excel.DisplayAlerts = false;
    wb = excel.Workbooks.Open(filePath);
    ws = wb.Sheets.Item(sheetName);
    colLetter = xlsColumnLetter(colIdx);
    firstRow = 2; lastRow = height(T) + 1;
    fullRange = ws.Range(sprintf('%s%d:%s%d', colLetter, firstRow, colLetter, lastRow));
    fullRange.Font.Bold = false;
    for k = 1:numel(toBold)
        r = toBold(k) + 1;
        ws.Range(sprintf('%s%d', colLetter, r)).Font.Bold = true;
    end
    wb.Save(); wb.Close(); excel.Quit(); delete(excel);
end
function excelBoldRankOnes(filePath, sheetName)
% Bold all cells equal to 1 in the "Rank" column of the given sheet.
    try T = readtable(filePath, 'Sheet', sheetName); catch, return; end
    if ~ismember('Rank', T.Properties.VariableNames), return; end
    colIdx  = find(strcmp(T.Properties.VariableNames, 'Rank'), 1);
    colVals = T{:, colIdx};
    dataMask = ~isnan(colVals);
    if ~any(dataMask), return; end
    rowsToBold = find(dataMask & (colVals == 1));
    excel = actxserver('Excel.Application');
    excel.Visible = false; excel.DisplayAlerts = false;
    wb = excel.Workbooks.Open(filePath);
    ws = wb.Sheets.Item(sheetName);
    colLetter = xlsColumnLetter(colIdx);
    firstRow  = 2; lastRow = height(T) + 1;
    fullRange = ws.Range(sprintf('%s%d:%s%d', colLetter, firstRow, colLetter, lastRow));
    fullRange.Font.Bold = false;
    for k = 1:numel(rowsToBold)
        r = rowsToBold(k) + 1;
        ws.Range(sprintf('%s%d', colLetter, r)).Font.Bold = true;
    end
    wb.Save(); wb.Close(); excel.Quit(); delete(excel);
end
function excelBoldAverageForRankOnes(filePath, sheetName)
% Bold the Average column cells for rows where Rank == 1.
% Safe if Rank/Average don't exist yet.
    try
        T = readtable(filePath, 'Sheet', sheetName);
    catch
        return;
    end
    if ~ismember('Rank', T.Properties.VariableNames) || ...
       ~ismember('Average', T.Properties.VariableNames)
        return;
    end
    rankColIdx = find(strcmp(T.Properties.VariableNames, 'Rank'), 1);
    avgColIdx  = find(strcmp(T.Properties.VariableNames, 'Average'), 1);
    rankVals   = T{:, rankColIdx};
    avgVals    = T{:, avgColIdx};
    dataMask   = ~isnan(rankVals);
    rowsToBold = find(dataMask & (rankVals == 1));
    excel = actxserver('Excel.Application');
    excel.Visible = false; excel.DisplayAlerts = false;
    wb = excel.Workbooks.Open(filePath);
    ws = wb.Sheets.Item(sheetName);
    avgColLetter = xlsColumnLetter(avgColIdx);
    firstRow     = 2; lastRow = height(T) + 1;
    fullRange    = ws.Range(sprintf('%s%d:%s%d', avgColLetter, firstRow, avgColLetter, lastRow));
    fullRange.Font.Bold = false;
    for k = 1:numel(rowsToBold)
        r = rowsToBold(k);
        if r >= 1 && r <= height(T) && ~isnan(avgVals(r))
            ws.Range(sprintf('%s%d', avgColLetter, r + 1)).Font.Bold = true; % +1 for header
        end
    end
    wb.Save(); wb.Close(); excel.Quit(); delete(excel);
end
function L = xlsColumnLetter(n)
% Convert 1-based column number to Excel letters (1->A, 27->AA, etc.)
    letters = '';
    while n > 0
        r = mod(n-1, 26);
        letters = [char(65 + r) letters]; %#ok<AGROW>
        n = floor((n-1)/26);
    end
    L = letters;
end
function M = alignCurvesPadNaN(curveList)
% Align column vectors by padding with NaN.
    N = numel(curveList);
    lens = cellfun(@numel, curveList(:));
    T = max(lens);
    M = nan(T, N);
    for j = 1:N
        v = curveList{j}(:);
        M(1:numel(v), j) = v;
    end
end
function repIdx = selectRepresentativeRun(repData, tol)
% Choose representative index in repData:
% If any wins: maximize leadFrac, then min tailMean, min final best, min firstBestIter
% Else: closest to best -> min gapFinal, min tailGap, max leadFrac, min final best, min firstBestIter.
    if isempty(repData), repIdx = NaN; return; end
    wins = find([repData.isWin]);
    if ~isempty(wins)
        % Among winners
        leadFracs = [repData(wins).leadFrac];
        [~, a] = max(leadFracs);
        pool = wins(abs(leadFracs - leadFracs(a)) <= eps);
        tailMeans = [repData(pool).tailMean];
        [~, b] = min(tailMeans); pool = pool(b);
        fimBests = [repData(pool).FIM_best];
        [~, c] = min(fimBests);  pool = pool(c);
        firstIters = [repData(pool).firstBestIter];
        [~, d] = min(firstIters); repIdx = pool(d);
        return;
    else
        % No wins: closest to best
        gaps = [repData.gapFinal];
        [~, a] = min(gaps);
        pool = find(abs(gaps - gaps(a)) <= max(tol, eps));
        tailGaps = [repData(pool).tailGap];
        [~, b] = min(tailGaps); pool = pool(b);
        leadFracs = [repData(pool).leadFrac];
        [~, c] = max(leadFracs); pool = pool(c);
        fimBests = [repData(pool).FIM_best];
        [~, d] = min(fimBests);  pool = pool(d);
        firstIters = [repData(pool).firstBestIter];
        [~, e] = min(firstIters); repIdx = pool(e);
        return;
    end
end
% ====== NEW: robust console print helpers ======
function s = fmtScore(x)
% Format scalar score or return "NA" for empty/NaN/Inf/non-scalar.
    if isempty(x) || ~isscalar(x) || ~isfinite(x)
        s = 'NA';
    else
        s = sprintf('%.6g', double(x));
    end
end
function printAlgoLine(algoNames, vals)
% Prints:  || FIM: <v> || GA: <v> || ...
    parts = cell(1, numel(algoNames));
    for i = 1:numel(algoNames)
        parts{i} = sprintf('%s: %s', algoNames{i}, fmtScore(vals(i)));
    end
    fprintf('  || %s\n', strjoin(parts, ' || '));
end
% ====== Added: curve standardizer (used above) ======
function c = standardizeCurve(c, best, T)
% Make curve usable for plotting:
% - column vector
% - length T (pad with last value)
% - monotone non-increasing (best-so-far)
% - final value equals reported best
% - replace empty/zero/invalid with a flat line at 'best'
    if nargin < 3 || isempty(T), T = numel(c); end
    c = c(:);
    if isempty(c) || (~any(isfinite(c))) || all(c==0)
        if ~isfinite(best) || isempty(best), best = NaN; end
        c = repmat(best, max(T,1), 1);
        return;
    end
    for t = 2:numel(c)
        if ~isfinite(c(t)) || c(t) > c(t-1)
            c(t) = c(t-1);
        end
    end
    if numel(c) < T
        c = [c; repmat(c(end), T-numel(c), 1)];
    elseif numel(c) > T
        c = c(1:T);
    end
    if isfinite(best) && best <= c(end)
        c(end) = best;
    end
end