function [Best_score, Best_pos, Optimization_curve] = HOA(Search, MaxFEs, lowerbound, upperbound, dimension, fitness)
% HOA (Hiking Optimization Algorithm) - comparison-style single-file
% Logic follows HOA_v2 (Tobler Hiking Function based) exactly
% FE-based version: the second input is MaxFEs, not Max_iterations.
% Stop: while FEs < MaxFEs

MaxFEs = floor(MaxFEs);
if MaxFEs < 1
    MaxFEs = 1;
end

% ---------- bounds as 1xD ----------
if isscalar(lowerbound), lb = repmat(lowerbound, 1, dimension); else, lb = lowerbound(:)'; end
if isscalar(upperbound), ub = repmat(upperbound, 1, dimension); else, ub = upperbound(:)'; end

nPop = Search;
D = dimension;

% ---------- initialize hikers ----------
Pop = repmat(lb, nPop, 1) + repmat((ub - lb), nPop, 1) .* rand(nPop, D);

% ---------- FE counter and convergence curve ----------
FEs = 0;
Optimization_curve = zeros(MaxFEs, 1);

% ---------- evaluate initial fitness ----------
fit = inf(nPop, 1);
Best_score = inf;
Best_pos = zeros(1, D);

for i = 1:nPop
    if FEs >= MaxFEs
        break;
    end

    fit(i) = fitness(Pop(i, :));
    FEs = FEs + 1;

    if fit(i) < Best_score
        Best_score = fit(i);
        Best_pos = Pop(i, :);
    end

    Optimization_curve(FEs) = Best_score;
end

% ========================= main loop =========================
while FEs < MaxFEs

    % current global best
    [~, ind] = min(fit);
    Xbest = Pop(ind, :);

    for j = 1:nPop
        if FEs >= MaxFEs
            break;
        end

        Xini = Pop(j, :);

        theta = randi([0 50], 1, 1);          % elevation angle
        s = tan(theta);                        % slope
        SF = randi([1 2], 1, 1);               % sweep factor (1 or 2)

        Vel = 6 .* exp(-3.5 .* abs(s + 0.05)); % Tobler hiking velocity
        newVel = Vel + rand(1, D) .* (Xbest - SF .* Xini);

        newPop = Pop(j, :) + newVel;           % update position
        newPop = min(ub, newPop);              % upper bound
        newPop = max(lb, newPop);              % lower bound

        fnew = fitness(newPop);
        FEs = FEs + 1;

        % greedy selection
        if fnew < fit(j)
            Pop(j, :) = newPop;
            fit(j) = fnew;

            if fnew < Best_score
                Best_score = fnew;
                Best_pos = newPop;
            end
        end

        Optimization_curve(FEs) = Best_score;
    end
end

% keep only evaluated FE points
if FEs == 0
    Optimization_curve = Best_score;
else
    Optimization_curve = Optimization_curve(1:FEs);
end

end
