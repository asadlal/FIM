function [Best_score, Best_pos, Optimization_curve] = PO(Search, MaxFEs, lowerbound, upperbound, dimension, fitness)
% PO in comparison style
% Original PO logic is kept the same.
% Only the function interface and final outputs are adapted to match the runner.
% FE-based version: the second input is MaxFEs, not Max_iterations.

N = Search;
MaxFEs = floor(MaxFEs);
lb = lowerbound;
ub = upperbound;
dim = dimension;
fobj = fitness;

if MaxFEs < 1
    MaxFEs = 1;
end

% BestF: Best value in a certain iteration
% WorstF: Worst value in a certain iteration
% GBestF: Global best fitness value
% AveF: Average value in each iteration

if (max(size(ub)) == 1)
    ub = ub .* ones(1, dim);
    lb = lb .* ones(1, dim);
end

% Nominal iteration count used only for the original PO time-control terms.
% The real stopping condition is FEs >= MaxFEs.
Max_iter = max(1, floor((MaxFEs - N) / max(1, N)));

%% Initialization
X0 = initialization(N, dim, ub, lb); % Initialization
X = X0;

% Compute initial fitness values
fit = inf(1, N);
FEs = 0;
lastRecordedFE = 0;

GBestF = inf; % Global best fitness value
GBestX = X(1, :); % Global best position

Optimization_curve = zeros(MaxFEs, 1);

for i = 1:N
    if FEs >= MaxFEs
        break;
    end

    fit(i) = evalFitness(X(i, :));

    if fit(i) < GBestF
        GBestF = fit(i);
        GBestX = X(i, :);
    end

    recordCurve();
end

[fit, index] = sort(fit); % sort
for i = 1:N
    X(i, :) = X0(index(i), :);
end

finiteFit = fit(isfinite(fit));
if isempty(finiteFit)
    AveF = inf;
else
    AveF = mean(finiteFit);
end

X_new = X;
Best_score = GBestF;
Best_pos = GBestX;

%% Start search
iter = 0;
while FEs < MaxFEs
    iter = iter + 1;

    progress = min(1, FEs / MaxFEs);
    alpha = rand(1) / 5;
    sita = rand(1) * pi;

    fitness_new = fit;
    X_new = X;

    for j = 1:size(X, 1)
        if FEs >= MaxFEs
            break;
        end

        St = randi([1, 4]);

        % foraging behavior
        if St == 1
            X_new(j, :) = (X(j, :) - GBestX) .* Levy(dim) + rand(1) * mean(X) * (1 - progress) ^ (2 * progress);

        % staying behavior
        elseif St == 2
            X_new(j, :) = X(j, :) + GBestX .* Levy(dim) + randn() * (1 - progress) * ones(1, dim);

        % communicating behavior
        elseif St == 3
            H = rand(1);
            if H < 0.5
                X_new(j, :) = X(j, :) + alpha * (1 - progress) * (X(j, :) - mean(X));
            else
                X_new(j, :) = X(j, :) + alpha * (1 - progress) * exp(-j / (rand(1) * Max_iter));
            end

        % fear of strangers' behavior
        else
            X_new(j, :) = X(j, :) + rand() * cos((pi * progress) / 2) * (GBestX - X(j, :)) ...
                - cos(sita) * progress ^ (2 / Max_iter) * (X(j, :) - GBestX);
        end

        % Boundary control
        for m = 1:N
            for a = 1:dim
                if (X_new(m, a) > ub(a))
                    X_new(m, a) = ub(a);
                end
                if (X_new(m, a) < lb(a))
                    X_new(m, a) = lb(a);
                end
            end
        end

        % Finding the best location so far
        fnew = evalFitness(X_new(j, :));
        fitness_new(j) = fnew;

        if fnew < GBestF
            GBestF = fnew;
            GBestX = X_new(j, :);
        end

        recordCurve();
    end

    % Update positions
    X = X_new;
    fit = fitness_new;

    % Sorting and updating
    [fit, index] = sort(fit); % sort
    for s = 1:N
        X0(s, :) = X(index(s), :);
    end
    X = X0;

    finiteFit = fit(isfinite(fit));
    if isempty(finiteFit)
        AveF = inf;
    else
        AveF = mean(finiteFit);
    end

    Best_pos = GBestX; %#ok<NASGU>
    Best_score = GBestF; %#ok<NASGU>

    recordCurve();
end

Best_score = GBestF;
Best_pos = GBestX;

if FEs == 0
    Optimization_curve = Best_score;
else
    Optimization_curve = Optimization_curve(1:FEs);
end

%% Nested FE helper functions
function f = evalFitness(x)
    if FEs >= MaxFEs
        f = inf;
        return;
    end
    f = fobj(x);
    FEs = FEs + 1;
end

function recordCurve()
    if FEs > lastRecordedFE
        Optimization_curve(lastRecordedFE+1:FEs) = GBestF;
        lastRecordedFE = FEs;
    end
end

%% Levy search strategy
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
        (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v).^(1 / beta);
    o = step;
end

% This function initialize the first population of search agents
function Positions = initialization(SearchAgents_no, dim, ub, lb)

Boundary_no = size(ub, 2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no == 1
    Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
end

% If each variable has a different lb and ub
if Boundary_no > 1
    Positions = zeros(SearchAgents_no, dim);
    for ii = 1:dim
        ub_i = ub(ii);
        lb_i = lb(ii);
        Positions(:, ii) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
    end
end

end

end
