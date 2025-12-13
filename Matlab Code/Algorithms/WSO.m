function [Best_score, Best_pos, Optimization_curve] = WSO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness)
% WSO: War Strategy Optimization (runner-compatible, logic unchanged)

% ---- Robust bounds handling (no casting) ----
lb = lowerbound; 
ub = upperbound;
lb = reshape(lb, 1, []);  
ub = reshape(ub, 1, []);
if numel(lb) == 1, lb = repmat(lb, 1, dimension); end
if numel(ub) == 1, ub = repmat(ub, 1, dimension); end
if numel(lb) ~= dimension, lb = repmat(lb(1), 1, dimension); end
if numel(ub) ~= dimension, ub = repmat(ub(1), 1, dimension); end
clamp = @(Z) max(min(Z, ub), lb);

% ---- Initialization (same distribution as original "initialization") ----
Positions = lb + rand(Search, dimension).*(ub - lb);
Positions = clamp(Positions);
pop_size  = size(Positions, 1);

Best_pos   = zeros(1, dimension);      % King
Best_score = inf;                       % King_fit
Optimization_curve = zeros(Max_iterations, 1);

Positions_new = zeros(size(Positions));
fitness_old   = inf(1, pop_size);
fitness_new   = inf(1, pop_size);
l  = 1;                                 % loop counter (original style)
W1 = 2*ones(1, pop_size);
Wg = zeros(1, pop_size);
R  = 0.1;                               % as in original

% ---- Initial evaluation (original) ----
for j = 1:pop_size
    f = fitness(Positions(j,:));
    fitness_old(j) = f;
    if f < Best_score
        Best_score = f;
        Best_pos   = Positions(j,:);
    end
end
Optimization_curve(1) = Best_score;

% ---- Main loop (original updates preserved) ----
while l < Max_iterations
    % pick Co = 2nd best (fallback if pop_size==1)
    [~, tindex] = sort(fitness_old);
    if numel(tindex) >= 2
        Co = Positions(tindex(2), :);
    else
        Co = Best_pos;
    end

    com = randperm(pop_size);
    for i = 1:pop_size
        RR = rand;
        if RR < R
            D_V = 2*RR*(Best_pos - Positions(com(i),:)) + 1*W1(i)*rand*(Co - Positions(i,:));
        else
            D_V = 2*RR*(Co - Best_pos) + 1*rand*(W1(i)*Best_pos - Positions(i,:));
        end

        % move and clamp
        Positions_new(i,:) = Positions(i,:) + D_V;
        Positions_new(i,:) = clamp(Positions_new(i,:));

        % evaluate
        f = fitness(Positions_new(i,:));
        fitness_new(i) = f;

        % update global best (King)
        if f < Best_score
            Best_score = f;
            Best_pos   = Positions_new(i,:);
        end

        % accept if improved (original)
        if f < fitness_old(i)
            Positions(i,:) = Positions_new(i,:);
            fitness_old(i) = f;
            Wg(i) = Wg(i) + 1;
            W1(i) = 1 * W1(i) * (1 - Wg(i)/Max_iterations)^2;
        end
    end

    % random reinsertion of current worst (original)
    if l < 1000
        [~, tindex1] = max(fitness_old);
        Positions(tindex1,:) = lb + rand(1,dimension).*(ub - lb);
        Positions(tindex1,:) = clamp(Positions(tindex1,:));
    end

    % advance loop, record best-so-far
    l = l + 1;
    Optimization_curve(l) = Best_score;
end

% ---- Final shaping ----
Best_pos = clamp(Best_pos);
Best_pos = Best_pos(:)';      % row vector

end
