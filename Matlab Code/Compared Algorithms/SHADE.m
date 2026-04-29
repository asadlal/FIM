function [Best_score, Best_pos, Optimization_curve] = SHADE(Search, MaxFEs, lowerbound, upperbound, dimension, fitness)
% SHADE (Success-History based Adaptive Differential Evolution)
% Written in the same comparison style as your other algorithms:
%   [Best_score, Best_pos, Optimization_curve] = ALG(Search, MaxFEs, lb, ub, D, fitness)
%
% Note: This is a simplified SHADE-style implementation based on the user-provided code.
% It keeps the "memory of F and CR" idea and binomial crossover + greedy selection.
% (It is NOT a full canonical SHADE/L-SHADE with archive and current-to-pbest/1.)
%
% FE-based version:
%   The second input is MaxFEs, not Max_iterations.
%   Every call to fitness() is counted as one function evaluation.
%   The algorithm stops when FEs >= MaxFEs.

MaxFEs = floor(MaxFEs);

%===============================
% Initialization
%===============================
X = lowerbound + rand(Search, dimension) .* (upperbound - lowerbound);

fit = inf(Search, 1);
Best_score = inf;
Best_pos = zeros(1, dimension);
Optimization_curve = zeros(MaxFEs, 1);
FEs = 0;

for i = 1:Search
    if FEs >= MaxFEs
        break;
    end

    fit(i) = fitness(X(i,:));
    FEs = FEs + 1;

    if fit(i) < Best_score
        Best_score = fit(i);
        Best_pos = X(i,:);
    end

    Optimization_curve(FEs) = Best_score;
end

%===============================
% Parameters (from provided code)
%===============================
CR0       = 0.5;     % initial CR
F0        = 0.5;     % initial F
F_decay   = 0.97;    % decay for memory entries
H         = 5;       % memory size
p_best_sz = 5;       % number of memory entries sampled to compute mean

% Memory arrays (each entry stores an F and CR)
MF  = F0  * ones(H, 1);
MCR = CR0 * ones(H, 1);

%===============================
% Helper
%===============================
clamp = @(Z) max(min(Z, upperbound), lowerbound);

%===============================
% Main Loop
%===============================
while FEs < MaxFEs

    V = zeros(Search, dimension);
    U = zeros(Search, dimension);

    % ----------------------------
    % Mutation + Crossover
    % ----------------------------
    for i = 1:Search

        % --- Select 3 distinct parent indices (r1,r2,r3) ---
        idx = pick_distinct_indices(Search, i, 3);
        r1 = idx(1); r2 = idx(2); r3 = idx(3);

        % --- Select memory indices and compute "p-best" mean of memory ---
        mem_idx = randperm(H, min(p_best_sz, H));
        p_best_F  = mean(MF(mem_idx));
        p_best_CR = mean(MCR(mem_idx));

        % --- Sample parameters around memory mean (as in provided code) ---
        % Keep them inside [0,1]
        F  = max(0, normrnd(p_best_F,  0.1));
        CR = max(0, normrnd(p_best_CR, 0.1));
        F  = min(F, 1);
        CR = min(CR, 1);

        % --- Mutant (DE/rand/1) ---
        V(i,:) = X(r1,:) + F * (X(r2,:) - X(r3,:));

        % --- Binomial crossover ---
        j_rand = randi(dimension);
        U(i,:) = X(i,:);
        for j = 1:dimension
            if (rand <= CR) || (j == j_rand)
                U(i,j) = V(i,j);
            end
        end

        % --- Bounds ---
        U(i,:) = clamp(U(i,:));
    end

    % ----------------------------
    % Selection
    % ----------------------------
    for i = 1:Search
        if FEs >= MaxFEs
            break;
        end

        f_new = fitness(U(i,:));
        FEs = FEs + 1;

        if f_new < fit(i)
            X(i,:) = U(i,:);
            fit(i) = f_new;

            if f_new < Best_score
                Best_score = f_new;
                Best_pos = U(i,:);
            end
        end

        Optimization_curve(FEs) = Best_score;
    end

    if FEs >= MaxFEs
        break;
    end

    % ----------------------------
    % Memory update (matches provided logic)
    % 1) decay memory
    % 2) small pull toward F0 and last CR sample (approx. behavior)
    % ----------------------------
    [~, sorted_idx] = sort(fit, 'ascend'); %#ok<ASGLU>

    m_eff = min(H, Search);
    for h = 1:m_eff
        MF(h)  = MF(h)  * F_decay;
        MCR(h) = MCR(h) * F_decay;
    end

    % Use the best few individuals as a trigger to slightly update memory.
    % (keeps "if sorted index is among best" idea without overcomplicating)
    for h = 1:m_eff
        if h <= min(p_best_sz, m_eff)
            MF(h)  = MF(h)  + 0.1 * (F0  - MF(h));
            MCR(h) = MCR(h) + 0.1 * (CR0 - MCR(h));
        end
    end
end

if FEs == 0
    Optimization_curve = Best_score;
else
    Optimization_curve = Optimization_curve(1:FEs);
end

% =====================================================================
% Helper: pick m distinct indices from 1..N excluding i
% =====================================================================
function idx = pick_distinct_indices(N, exclude_i, m)
    pool = 1:N;
    pool(pool == exclude_i) = [];
    perm = randperm(numel(pool), m);
    idx = pool(perm);
end

end
