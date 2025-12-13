function run_ranksum()
% This script runs a Wilcoxon Rank-Sum test for a set of optimization algorithms.
% It now pre-computes and saves the results for all algorithms for improved efficiency and resume capability.
clc;
clear;
close all;

%% -------- User settings --------
bench_funinfo  = true;                % fun_info for Unimodal/Multimodal/Fixed-Dimensional
bench_cec      = true;                % CEC2017 set

% --- Parallel computing settings ---
parallel_on    = false;                % Set to true to enable parallel processing (ON/OFF switch)
num_workers    = 10;                  % Set the desired number of workers for THIS RUN
max_cluster_limit = num_workers;      

Search         = 50;                   
Max_iterations = 1000;                  
num_runs       = 30;                   

% Choose: 'unimodal' | 'multimodal' | 'fixed' | 'cec2017' | 'all'
run_set = 'all';

% Algorithms (function names in /Algorithms)
algos = {'FIM','GA','TBLO','WSO','WO','GSA','WOA','GWO','AVOA','RSA','MPA','TSA','MVO'};

% Comparator mode: 'single' or 'all'
mode = 'all';                      % when 'all', selected_comparator is ignored
selected_comparator = 'TBLO';          % used only if mode='single'

%% -------- Parallel Pool Setup (Automated Limit Adjustment) --------
use_parallel = parallel_on; 
effective_workers = 1;

if use_parallel
    try
        % 1) Local cluster profile
        cluster = parcluster();

        % 2) Lift cluster limit if needed
        current_limit = cluster.NumWorkers;
        if current_limit < max_cluster_limit
            fprintf('Warning: Local cluster limit (%d) < requested maximum (%d).\n', ...
                current_limit, max_cluster_limit);
            cluster.NumWorkers = max_cluster_limit;
            cluster.saveProfile();
            fprintf('Cluster limit permanently updated to %d.\n', max_cluster_limit);
        end

        % 3) Start pool with desired workers
        pool_idle_minutes = 24*60*7;      % ~7 days
        pool = gcp('nocreate');
        if isempty(pool)
            if isprop(cluster,'IdleTimeout')
                cluster.IdleTimeout = pool_idle_minutes;
            end
            pool = parpool(cluster, num_workers);
        end

        % 4) Confirm workers
        effective_workers = pool.NumWorkers;
        fprintf('Parallel execution enabled. Pool is running with %d effective workers.\n', effective_workers);

    catch ME
        warning('Custom:ParallelFail', 'Failed to start parallel pool: %s\nRunning sequentially.', ME.message);
        use_parallel = false;
    end
end

if ~use_parallel
    fprintf('Running sequentially (Parallel switch is OFF or setup failed).\n');
end

%% -------- Paths --------
addpath(genpath(fullfile(pwd,'Algorithms')));

% Base: Results and Graphs
base_dir = fullfile(pwd, 'Results and Graphs');
if ~exist(base_dir, 'dir'), mkdir(base_dir); end

% New parent folder under Results and Graphs
ranksum_parent = fullfile(base_dir, 'Rank-Sum Test P-value');
if ~exist(ranksum_parent, 'dir'), mkdir(ranksum_parent); end

% Children inside "Rank-Sum Test P-value"
wil_dir = fullfile(ranksum_parent, 'Wilcoxon Rank-Sum P-value');
if ~exist(wil_dir, 'dir'), mkdir(wil_dir); end

algo_results_dir = fullfile(ranksum_parent, 'Algo Results Saved');
if ~exist(algo_results_dir, 'dir'), mkdir(algo_results_dir); end

%% -------- Parallel Progress Queue Setup --------
dq = [];  
if use_parallel
    dq = parallel.pool.DataQueue;
    afterEach(dq, @(msg) fprintf('%s\n', msg));
end

%% -------- Define sets (machine key + pretty name) --------
SETS = struct();
if bench_funinfo
    SETS.unimodal   = struct('name','Unimodal',          'bench','fun_info', 'funcs',1:7);
    SETS.multimodal = struct('name','Multimodal',        'bench','fun_info', 'funcs',8:13);
    SETS.fixed      = struct('name','Fixed-Dimensional', 'bench','fun_info', 'funcs',14:23);
end
if bench_cec
    SETS.cec2017    = struct('name','CEC2017',           'bench','CEC2017',  'funcs',[1,3:30]);
end

% Which sets to run
if strcmpi(run_set,'all')
    set_keys = fieldnames(SETS);  % {'unimodal','multimodal','fixed','cec2017'}
else
    if ~isfield(SETS, lower(run_set)), error('Set "%s" not available.', run_set); end
    set_keys = {lower(run_set)};
end

% Comparators & which algorithms to simulate
FIM_name = 'FIM';
if strcmpi(mode, 'single')
    comparators   = {selected_comparator};                         % compare only against the selected one
    algos_to_run  = unique({FIM_name, selected_comparator}, 'stable');  % simulate only FIM + that comparator
else
    comparators   = setdiff(algos, FIM_name, 'stable');            % compare FIM vs all others
    algos_to_run  = unique([{FIM_name}, comparators], 'stable');   % simulate FIM + all others
end

%% -------- Start banner --------
fprintf('\n============================================================\n');
fprintf('[%s] Wilcoxon Rank-Sum: START\n', datestr(now,'yyyy-mm-dd HH:MM:SS'));
fprintf('Sets: %s | Runs/func: %d | Parallel: %d\n', ...
    strjoin(cellfun(@(k) SETS.(k).name, set_keys,'uni',0), ', '), num_runs, use_parallel);
fprintf('Comparators: %s\n', strjoin(comparators, ', '));
if strcmpi(mode,'all')
    fprintf('(selected_comparator is ignored because mode=all)\n');
end
fprintf('Output folder: %s\n', wil_dir);
fprintf('============================================================\n\n');

% For master table when run_set='all'
master_rows = struct();  % master_rows.(comp).(key) = p_two

%% -------- Loop sets --------
for sk = 1:numel(set_keys)
    key    = set_keys{sk};   % machine key
    S      = SETS.(key);
    pretty = S.name;

    % New: Pre-compute and save all algorithm results for this set
    fprintf('########## %s Simulation ##########\n', pretty);
    benchmark_dir = fullfile(algo_results_dir, pretty);
    if ~exist(benchmark_dir, 'dir'), mkdir(benchmark_dir); end

    % Prepare function data once to pass to parallel workers
    func_data = cell(1, numel(S.funcs));
    for fi = 1:numel(S.funcs)
        func_num = S.funcs(fi);
        [lb, ub, dim, fitness] = get_bench(S.bench, func_num);
        func_data{fi} = struct('lb', lb, 'ub', ub, 'dim', dim, 'fitness', fitness);
    end

    % Loop through the algorithms we need to simulate (FIM + selected comparator in 'single' mode)
    for alg_idx = 1:numel(algos_to_run)
        alg_name = algos_to_run{alg_idx};
        alg_dir = fullfile(benchmark_dir, alg_name);
        results_file = fullfile(alg_dir, 'results.mat');

        if isfile(results_file)
            fprintf('[%s] Found existing results for %s. Skipping simulation.\n', pretty, alg_name);
            continue; % Skip to next algorithm
        end

        if ~exist(alg_dir, 'dir'), mkdir(alg_dir); end

        fprintf('[%s] Running simulations for %s...\n', pretty, alg_name);

        alg_results = cell(num_runs, numel(S.funcs));

        if use_parallel
            parfor fi = 1:numel(S.funcs)
                f = S.funcs(fi);
                data = func_data{fi};
                fun_label = sprintf('F%d', f);
                for r = 1:num_runs
                    dq.send(sprintf('[%s] Running %s on function %s (Run %d)...\n', pretty, alg_name, fun_label, r));
                    alg_results{r, fi} = run_once(alg_name, Search, Max_iterations, data.lb, data.ub, data.dim, data.fitness);
                end
            end
        else
            for fi = 1:numel(S.funcs)
                f = S.funcs(fi);
                data = func_data{fi};
                fun_label = sprintf('F%d', f);
                for r = 1:num_runs
                    fprintf('[%s] Running %s on function %s (Run %d)...\n', pretty, alg_name, fun_label, r);
                    alg_results{r, fi} = run_once(alg_name, Search, Max_iterations, data.lb, data.ub, data.dim, data.fitness);
                end
            end
        end

        % Save the results for this algorithm
        save(results_file, 'alg_results');
        fprintf('[%s] Saved results for %s to %s\n', pretty, alg_name, results_file);
    end

    % --- Analysis Phase ---
    fprintf('########## %s Analysis ##########\n', pretty);

    % Load FIM results once for all comparisons in this set
    FIM_results_file = fullfile(benchmark_dir, 'FIM', 'results.mat');
    if ~isfile(FIM_results_file)
        error('FIM results file not found: %s. Cannot perform comparisons.', FIM_results_file);
    end
    load(FIM_results_file, 'alg_results');
    FIM_results = alg_results;

    out_file = fullfile(wil_dir, sprintf('Wilcoxon (%s).xlsx', pretty));

    for ci = 1:numel(comparators)
        comp = comparators{ci};
        sheet_name = sprintf('FIM vs %s (%s)', comp, pretty);

        % ===== Perform comparison using saved results =====
        fprintf('[%s] Comparing FIM vs %s...\n', pretty, comp);

        comp_results_file = fullfile(benchmark_dir, comp, 'results.mat');
        if ~isfile(comp_results_file)
            error('Comparator results file not found for %s: %s. Cannot perform comparison.', comp, comp_results_file);
        end
        load(comp_results_file, 'alg_results');
        comp_results = alg_results;

        FIM_all = [];
        COMP_all = [];
        all_rows = {};

        for r = 1:num_runs
            for fi = 1:numel(S.funcs)
                f = S.funcs(fi);
                fun_label = sprintf('F%d', f);
                FIM_result = FIM_results{r, fi};
                comp_result = comp_results{r, fi};

                FIM_all  = [FIM_all;  FIM_result];
                COMP_all = [COMP_all; comp_result];
                all_rows(end+1,:) = {size(all_rows,1) + 1, fun_label, FIM_result, comp_result};
            end
        end

        % ===== Two-sided Wilcoxon p for this set =====
        [p_two, ~] = ranksum(FIM_all, COMP_all, 'method', 'approx');
        fprintf('%s vs %s for %s: p-value = %.3g\n', FIM_name, comp, pretty, p_two);

        T_runs = cell2table(all_rows, 'VariableNames', {'Run','Function','FIM','Comparator'});
        writetable(T_runs, out_file, 'Sheet', sheet_name, 'WriteMode','overwrite');

        startRow = height(T_runs) + 2;
        Info = table({FIM_name}, {comp}, {pretty}, 'VariableNames', {'Algo1','Algo2','Set'});
        writetable(Info, out_file, 'Sheet', sheet_name, ...
            'Range', sprintf('F%d', startRow), 'WriteVariableNames', true);
        Pv = table({'Two-sided p-value'}', p_two, 'VariableNames', {'Metric','Value'});
        writetable(Pv, out_file, 'Sheet', sheet_name, ...
            'Range', sprintf('F%d', startRow+3), 'WriteVariableNames', true);

        if ~isfield(master_rows, comp), master_rows.(comp) = struct(); end
        master_rows.(comp).(key) = p_two;
    end
    fprintf('\n');
end

%% -------- If running all sets, write master table to its own file --------
if strcmpi(run_set,'all')
    col_keys  = {};
    col_names = {};
    if isfield(SETS,'unimodal'),   col_keys{end+1}='unimodal';   col_names{end+1}=SETS.unimodal.name;   end
    if isfield(SETS,'multimodal'), col_keys{end+1}='multimodal'; col_names{end+1}=SETS.multimodal.name; end
    if isfield(SETS,'fixed'),      col_keys{end+1}='fixed';      col_names{end+1}=SETS.fixed.name;      end
    if isfield(SETS,'cec2017'),    col_keys{end+1}='cec2017';    col_names{end+1}=SETS.cec2017.name;    end

    fprintf('\n============================================================\n');
    fprintf('Master Wilcoxon Rank-Sum P-Values (FIM vs Competitors)\n');
    fprintf('============================================================\n');
    fprintf('%-20s', 'Comparator');
    for j = 1:numel(col_names)
        fprintf('%-15s', col_names{j});
    end
    fprintf('\n');

    master_tbl = {};
    for ci = 1:numel(comparators)
        comp = comparators{ci};
        vals = nan(1,numel(col_keys));
        for j = 1:numel(col_keys)
            k = col_keys{j};
            if isfield(master_rows.(comp), k), vals(j) = master_rows.(comp).(k); end
        end
        fprintf('%-20s', sprintf('%s vs %s', FIM_name, comp));
        for j = 1:numel(vals)
            if isnan(vals(j))
                fprintf('%-15s', 'N/A');
            else
                fprintf('%-15.3g', vals(j));
            end
        end
        fprintf('\n');
        master_tbl(end+1,:) = [{sprintf('%s vs %s', FIM_name, comp)} num2cell(vals)];
    end

    master_file = fullfile(wil_dir, 'Wilcoxon (All Sets).xlsx');
    col_vars = regexprep(col_names, '[-\s]+', '_');
    VarNames = [{'Comparator'}, col_vars];
    Tmaster  = cell2table(master_tbl, 'VariableNames', VarNames);
    writetable(Tmaster, master_file, 'Sheet', 'P-Values (All Sets)', 'WriteMode','overwrite');
    fprintf('\nMaster summary saved: %s\n', master_file);
    fprintf('============================================================\n\n');
end

%% -------- End banner --------
fprintf('\n============================================================\n');
fprintf('[%s] Wilcoxon Rank-Sum: END\n', datestr(now,'yyyy-mm-dd HH:MM:SS'));
fprintf('Outputs saved in: %s\n', wil_dir);
fprintf('============================================================\n\n');
end

%% ================== Helpers ==================
function best = run_once(algoName, Search, Max_iterations, lb, ub, dim, fitness)
    f = str2func(algoName);
    [best, ~, ~] = f(Search, Max_iterations, lb, ub, dim, fitness);
end

function [lb, ub, dim, fitness] = get_bench(benchName, F)
    lb=[]; ub=[]; dim=[]; fitness=[];
    switch lower(benchName)
        case 'fun_info'
            [lb, ub, dim, fitness] = call_fun_info(F);
        case 'cec2017'
            [lb, ub, dim, fitness] = call_cec2017(F);
        otherwise
            error('Unknown benchName: %s', benchName);
    end
    if isempty(lb) || isempty(ub) || isempty(dim) || isempty(fitness)
        error('Benchmark "%s" F=%s returned empty outputs.', benchName, local_to_str(F));
    end
end

function [lb, ub, dim, fitness] = call_fun_info(F)
    lb=[]; ub=[]; dim=[]; fitness=[];
    try, [lb,ub,dim,fitness] = fun_info(F); if ~isempty(lb), return; end, end
    try, tag=sprintf('F%d',F); [lb,ub,dim,fitness] = fun_info(tag); if ~isempty(lb), return; end, end
    try, [lb,ub,dim,fitness,~] = fun_info(F); if ~isempty(lb), return; end, end
    try, tag=sprintf('F%d',F); [lb,ub,dim,fitness,~] = fun_info(tag); if ~isempty(lb), return; end, end
    error('fun_info failed for F=%s (tried numeric/string, 4/5 outputs).', local_to_str(F));
end

function [lb, ub, dim, fitness] = call_cec2017(F)
    lb=[]; ub=[]; dim=[]; fitness=[];
    try, [lb,ub,dim,fitness] = CEC2017(F); if ~isempty(lb), return; end, end
    try, tag=sprintf('F%d',F); [lb,ub,dim,fitness] = CEC2017(tag); if ~isempty(lb), return; end, end
    error('CEC2017 failed for F=%s (tried numeric and ''F#'').', local_to_str(F));
end

function s = local_to_str(F)
    if isnumeric(F), s = num2str(F); else, s = char(F); end
end
