function [Best_score, Best_pos, Optimization_curve] = FIM(Search, MaxFEs, lowerbound, upperbound, dimension, fitness)
% FIM: Father-Inspired Metaheuristic
% FE-based version.
%
% Main phases:
% Phase 1: Neighborhood-Repulsion-Based Discipline
% Phase 2: Self-Learning and Elite Guidance
% Phase 3: Reward-Correction
%
% Midpoint repair is used for generated trial solutions, as defined in Eq. (5).
% Light QN is applied only in the last 10% of the FE budget on the top 5%
% of the population.

%===============================
% Initialization
%===============================
MaxFEs = floor(MaxFEs);
if MaxFEs < 1
    MaxFEs = 1;
end

FEs = 0;
lastRecordedFE = 0;

% Eq. (2): Random population initialization
X = lowerbound + rand(Search, dimension) .* (upperbound - lowerbound);

obj_values = inf(Search,1);
Best_score = inf;
Best_pos   = zeros(1, dimension);

Optimization_curve = zeros(1, MaxFEs);

%===============================
% FIM Parameters
%===============================
beta0 = 0.5;
alpha = 2;

K = 10;
M = 4;

p_rate = 0.20;     % elite pool = top 20% population

Qlight = 2;        % light QN inner iterations
Delta  = 1e-6;     % gradient step size

gamma0    = 1.0;   % initial step size for backtracking
gamma_min = 1e-6;  % minimum step size for backtracking

Ic = 15;           % correction interval
Nc = 5;            % number of corrected children

tol_grad = 1e-20;

%===============================
% Helpers
%===============================
clamp = @(Z) max(min(Z, upperbound), lowerbound);
X = clamp(X);

I_dim = eye(dimension);
span = upperbound - lowerbound;

%===============================
% Initial fitness evaluations
% Eq. (3): Fitness evaluation
%===============================
for i = 1:Search
    if FEs >= MaxFEs
        break;
    end

    obj_values(i) = evalFitness(X(i,:));

    if obj_values(i) < Best_score
        Best_score = obj_values(i);
        Best_pos = X(i,:);
    end

    recordCurve();
end

%===============================
% Main Loop
%===============================
t = 0;

while FEs < MaxFEs

    t = t + 1;

    % Eq. (4): Normalized search progress tau = FEs / MaxFEs
    progress = min(1, FEs / MaxFEs);

    % Safety: keep all agents in bounds at loop start
    X = clamp(X);

    %===========================================================
    % Phase 1: Neighborhood-Repulsion-Based Discipline
    %===========================================================
    % Eq. (6): Compute distances between solutions
    % Eq. (7): Select K nearest neighbors
    % Eq. (8): Select M worst neighbors from the K nearest neighbors
    % Eq. (9): Compute weak-neighborhood center Wavg
    %
    % Eq. (10):
    % x_p1 = x_i + (1-tau)*beta_t*r1.*(x_i - Wavg) ...
    %            + tau*r2.*(father - x_i)
    %
    % Here:
    % r1 ~ U(0.5,1.5) controls the repulsion term
    % r2 ~ U(0,1) controls the father-guidance term
    %
    % Eq. (11):
    % beta_t = beta0 / (1 + sqrt(alpha*tau))

    beta_t = beta0 / (1 + sqrt(alpha * progress));

    X_phase1   = X;
    obj_phase1 = obj_values;

    [~, best_idx] = min(obj_phase1);
    Xfather = X_phase1(best_idx,:);

    % Eq. (6): Build squared-distance matrix for the whole population
    sumX2 = sum(X_phase1.^2, 2);
    dist2_mat = max(sumX2 + sumX2.' - 2 * (X_phase1 * X_phase1.'), 0);
    dist2_mat(1:Search+1:end) = inf;   % exclude self-distance

    Xp1_all = X_phase1;
    fp1_all = obj_phase1;

    K_eff = min(K, Search - 1);

    if K_eff > 0
        for i = 1:Search
            if FEs >= MaxFEs
                break;
            end

            % Eq. (7): K nearest neighbors
            [~, nn_idx] = mink(dist2_mat(i,:), K_eff);

            neigh   = X_phase1(nn_idx, :);
            f_neigh = obj_phase1(nn_idx);

            M_eff = min(M, numel(f_neigh));
            if M_eff <= 0
                continue;
            end

            % Eq. (8): M worst neighbors
            [~, widx] = maxk(f_neigh, M_eff);

            % Eq. (9): Weak-neighborhood center
            Wavg = mean(neigh(widx, :), 1);

            % Random factors for Eq. (10)
            r1 = 0.5 + rand(1, dimension);    % r1 in [0.5,1.5] for repulsion
            r2 = rand(1, dimension);          % r2 in [0,1] for father guidance

            % Eq. (10): Phase 1 candidate solution
            Xp1 = X_phase1(i,:) ...
                + (1 - progress) * beta_t .* r1 .* (X_phase1(i,:) - Wavg) ...
                + progress .* r2 .* (Xfather - X_phase1(i,:));

            % Eq. (5): Midpoint boundary repair
            Xp1 = midpointRepair(Xp1, X_phase1(i,:), lowerbound, upperbound);

            fp1 = evalFitness(Xp1);

            % Eq. (12): Greedy selection
            if fp1 < obj_phase1(i)
                Xp1_all(i,:) = Xp1;
                fp1_all(i) = fp1;

                if fp1 < Best_score
                    Best_score = fp1;
                    Best_pos = Xp1;
                end
            end

            recordCurve();
        end

        improve_mask = fp1_all < obj_values;
        if any(improve_mask)
            X = Xp1_all;
            obj_values = fp1_all;
        end
    end

    if FEs >= MaxFEs
        break;
    end

    %===========================================================
    % Phase 2: Self-Learning and Elite Guidance
    %===========================================================
    % Eq. (13): pL = 0.30 + 0.50*tau
    %
    % If u < pL:
    %   use Eq. (16): elite-guided learning
    % Otherwise:
    %   use Eq. (14) and Eq. (15): self-learning perturbation
    %
    % Eq. (17): Select between elite-learning and self-learning candidate
    % Eq. (18): Greedy selection

    pL = 0.30 + 0.50 * progress;

    p_num = max(2, ceil(p_rate * Search));
    p_num = min(p_num, Search);

    [~, sort_idx] = sort(obj_values, 'ascend');
    elite_pool = sort_idx(1:p_num);

    for i = 1:Search
        if FEs >= MaxFEs
            break;
        end

        useElite = (rand < pL) && (Search >= 3);

        if useElite
            %-------------------------------
            % Elite-guided learning
            % Eq. (16)
            %-------------------------------
            elite_idx = elite_pool(randi(p_num));

            idx_pool = 1:Search;
            idx_pool(idx_pool == i) = [];

            if numel(idx_pool) >= 2
                rr = idx_pool(randperm(numel(idx_pool), 2));

                % Peer indices.
                % These are named a_idx and b_idx to avoid confusion with
                % r1 and r2 in Eq. (10).
                a_idx = rr(1);
                b_idx = rr(2);

                % Eq. (16): lambda_i and mu_i
                lambda_i = 0.50 + 0.30 * rand;   % lambda_i in [0.50, 0.80]
                mu_i     = 0.30 + 0.40 * rand;   % mu_i in [0.30, 0.70]

                % Eq. (16): Elite-learning candidate
                Xcand = X(i,:) ...
                    + lambda_i .* (X(elite_idx,:) - X(i,:)) ...
                    + mu_i .* (X(a_idx,:) - X(b_idx,:));

                % Eq. (5): Midpoint boundary repair
                Xcand = midpointRepair(Xcand, X(i,:), lowerbound, upperbound);
            else
                useElite = false;
            end
        end

        if ~useElite
            %-------------------------------
            % Self-learning perturbation
            % Eq. (14) and Eq. (15)
            %-------------------------------
            Rpert = rand(1, dimension);           % random value in [0,1]
            phi   = 1 - 2 * Rpert;                % phi in [-1,1]
            Lij   = randi([1, 2], 1, dimension);  % Lij in {1,2}

            % Eq. (14): delta_i,j = phi_i,j * L_i,j * ((ub_j-lb_j)/t)
            delta = (phi .* Lij) .* (span ./ t);

            % Eq. (15): self-learning candidate
            Xcand = X(i,:) + delta;

            % Eq. (5): Midpoint boundary repair
            Xcand = midpointRepair(Xcand, X(i,:), lowerbound, upperbound);
        end

        fp = evalFitness(Xcand);

        % Eq. (18): Greedy selection
        if fp < obj_values(i)
            X(i,:) = Xcand;
            obj_values(i) = fp;

            if fp < Best_score
                Best_score = fp;
                Best_pos = Xcand;
            end
        end

        recordCurve();
    end

    if FEs >= MaxFEs
        break;
    end

    %===========================================================
    % Phase 3: Reward-Correction
    %===========================================================

    %===========================================================
    % Phase 3.1: Reward
    % Light QN refinement
    %===========================================================
    % Reward is applied only when tau >= 0.90.
    % It is applied to the top 5% of the population.
    %
    % Eq. (19): Central-difference gradient at current solution
    % Eq. (20): Search direction p = -B*g
    % Eq. (21): Backtracking trial solution
    % Eq. (22): Central-difference gradient at new solution
    % Eq. (23): s, y, and rho terms
    % Eq. (24): BFGS inverse-Hessian update
    % Eq. (25): Greedy acceptance of refined solution

    if progress >= 0.90

        top_qn_count = max(1, ceil(0.05 * Search));
        [~, qn_order] = sort(obj_values, 'ascend');
        qn_agents = qn_order(1:top_qn_count);

        for ii = 1:top_qn_count
            if FEs >= MaxFEs
                break;
            end

            i = qn_agents(ii);

            x = X(i,:);
            B = I_dim;
            fx_curr = obj_values(i);

            % Eq. (19): Initial central-difference gradient
            [g, grad_ok] = centralGradient(x, x, Delta);
            if ~grad_ok
                break;
            end

            if norm(g) >= tol_grad
                for q = 1:Qlight
                    if FEs >= MaxFEs
                        break;
                    end

                    % Eq. (20): Search direction
                    p = -B * g';

                    % Eq. (21): Backtracking trial solution
                    gamma = gamma0;
                    x_new = x + gamma * p';
                    x_new = midpointRepair(x_new, x, lowerbound, upperbound);

                    f_old = fx_curr;
                    f_new = evalFitness(x_new);
                    recordCurve();

                    while f_new >= f_old && gamma > gamma_min && FEs < MaxFEs
                        gamma = 0.5 * gamma;

                        x_new = x + gamma * p';
                        x_new = midpointRepair(x_new, x, lowerbound, upperbound);

                        f_new = evalFitness(x_new);
                        recordCurve();
                    end

                    if f_new >= f_old
                        break;
                    end

                    % If no budget remains for the new gradient, keep the improved point
                    % and leave QN for this agent.
                    if FEs >= MaxFEs
                        x = x_new;
                        fx_curr = f_new;
                        break;
                    end

                    % Eq. (22): Central-difference gradient at new solution
                    [g_new, grad_ok] = centralGradient(x_new, x_new, Delta);
                    if ~grad_ok
                        x = x_new;
                        fx_curr = f_new;
                        break;
                    end

                    % Eq. (23): s, y, and rho
                    s = x_new - x;
                    y = g_new - g;

                    if isrow(s), s = s.'; end
                    if isrow(y), y = y.'; end

                    ys = y.' * s;

                    % Eq. (24): BFGS inverse-Hessian update
                    if ys > 1e-14
                        rho = 1 / ys;
                        V = I_dim - rho * (s * y.');
                        B = V * B * V.' + rho * (s * s.');
                    end

                    % Accept QN step inside the local refinement
                    x = x_new;
                    fx_curr = f_new;
                    g = g_new;

                    if norm(g) < tol_grad
                        break;
                    end
                end
            end

            % Eq. (25): Accept refined solution if improved
            if fx_curr < obj_values(i)
                X(i,:) = x;
                obj_values(i) = fx_curr;

                if fx_curr < Best_score
                    Best_score = fx_curr;
                    Best_pos = x;
                end
            end

            recordCurve();
        end
    end

    if FEs >= MaxFEs
        break;
    end

    %===========================================================
    % Phase 3.2: Correction
    %===========================================================
    % Correction is triggered when mod(t, Ic) = 0.
    %
    % Eq. (26): x_c = x_i + eta*(b_j - x_i)
    % Eq. (27): eta = 0.4*(1-tau)^2
    %
    % Here, selected weak solutions are redirected toward randomly selected
    % boundary points to restore diversity.

    if mod(t, Ic) == 0

        [~, sorted_idx] = sort(obj_values, 'descend');

        numWorst      = max(1, round(0.2 * Search));
        worst_indices = sorted_idx(1:numWorst);

        num_to_send = min(Nc, numWorst);
        pick = worst_indices(randperm(numWorst, num_to_send));

        % Eq. (27): Correction strength eta
        eta = 0.4 * (1 - progress)^2;

        % Random boundary value b_j, either lb_j or ub_j
        boundary = lowerbound + (upperbound - lowerbound) .* randi([0,1], num_to_send, dimension);

        % Eq. (26): Corrected candidate solution
        Xcorr = X(pick,:) + eta * (boundary - X(pick,:));

        % Eq. (5): Midpoint boundary repair
        Xcorr = midpointRepair(Xcorr, X(pick,:), lowerbound, upperbound);

        new_scores = inf(num_to_send,1);
        evaluated_mask = false(num_to_send,1);

        for k = 1:num_to_send
            if FEs >= MaxFEs
                break;
            end

            new_scores(k) = evalFitness(Xcorr(k,:));
            evaluated_mask(k) = true;

            if new_scores(k) < Best_score
                Best_score = new_scores(k);
                Best_pos = Xcorr(k,:);
            end

            recordCurve();
        end

        if any(evaluated_mask)
            X(pick(evaluated_mask),:) = Xcorr(evaluated_mask,:);
            obj_values(pick(evaluated_mask)) = new_scores(evaluated_mask);
        end
    end

    % End-of-iteration safety clamp
    X = clamp(X);

    % Record convergence
    recordCurve();

end

% Fill any remaining curve entries with the final best value.
recordCurve();
if lastRecordedFE < MaxFEs
    Optimization_curve(lastRecordedFE+1:MaxFEs) = Best_score;
end

%===============================
% Nested FE helper functions
%===============================
    function f = evalFitness(x)
        if FEs >= MaxFEs
            f = inf;
            return;
        end

        f = fitness(x);
        FEs = FEs + 1;
    end

    function recordCurve()
        if FEs > lastRecordedFE
            Optimization_curve(lastRecordedFE+1:FEs) = Best_score;
            lastRecordedFE = FEs;
        end
    end

    function [g, ok] = centralGradient(x0, xbase, delta_grad)
        g = zeros(1, dimension);
        ok = true;

        for jj = 1:dimension
            if FEs >= MaxFEs
                ok = false;
                return;
            end

            Xp = x0;
            Xm = x0;

            Xp(jj) = x0(jj) + delta_grad;
            Xm(jj) = x0(jj) - delta_grad;

            Xp = midpointRepair(Xp, xbase, lowerbound, upperbound);
            Xm = midpointRepair(Xm, xbase, lowerbound, upperbound);

            fp = evalFitness(Xp);
            recordCurve();

            if FEs >= MaxFEs
                ok = false;
                return;
            end

            fm = evalFitness(Xm);
            recordCurve();

            g(jj) = (fp - fm) / (2 * delta_grad);
        end
    end

end

%===============================
% Eq. (5): Midpoint repair helper
%===============================
function Xnew = midpointRepair(Xtrial, Xbase, lb, ub)

    LB = lb + zeros(size(Xtrial));
    UB = ub + zeros(size(Xtrial));

    Xnew = Xtrial;

    lowMask = Xtrial < LB;
    upMask  = Xtrial > UB;

    Xnew(lowMask) = (Xbase(lowMask) + LB(lowMask)) / 2;
    Xnew(upMask)  = (Xbase(upMask) + UB(upMask)) / 2;

end