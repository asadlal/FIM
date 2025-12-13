function [Best_score, Best_pos, Optimization_curve] = FIM(Search, Max_iterations, lowerbound, upperbound, dimension, fitness) 
% FIM: Father-Inspired Metaheuristic
% Comments below reference the equation numbers in the paper (Algorithm 1 + Section 3).

%===============================
% Initialization
%===============================
% Eq. (1): population matrix X ∈ ℝ^{N×D}
% Eq. (2): random initialization within bounds
X = lowerbound + rand(Search, dimension) .* (upperbound - lowerbound);

% Eq. (3): fitness vector F stores F(X_i); initialize personal bests Pi
obj_values = zeros(Search,1);
personal_best_pos = X;                 % Pi ← Xi
personal_best_score = zeros(Search,1); % F(Pi)  

for i = 1:Search
    obj_values(i) = fitness(X(i,:));   % F(Xi)
    personal_best_score(i) = obj_values(i); % initial pbest F(Pi) ← F(Xi)
end

% Track best-so-far solution (father / global best X*)
[Best_score, best_idx] = min(obj_values);
Best_pos = X(best_idx,:);
Optimization_curve = zeros(1, Max_iterations);

%===============================
% FIM Parameters
%===============================
% Discipline schedule (Eq. (10): βt = β0 / (1 + sqrt(α·τ)))
beta0 = 0.5;
alpha = 4;

% Neighborhood settings for Phase 2 (Eqs. (7)-(9))
K = 10;      % K nearest neighbors (Table 1)
M = 4;       % choose worst M among K (Table 1)

% Elite archive size parameter m used in Eq. (26) (Table 1)
max_elite = 10;

% Reward (QN) parameters (Reward uses Eqs. (18)-(25))
% Light vs Strong settings come from Table 1
max_qn_iter_light  = 5;
max_qn_iter_strong = 20;
delta_grad_light   = 1e-6;  % Δlight for Eq. (18) and Eq. (21)
delta_grad_strong  = 1e-9;  % Δstrong for Eq. (18) and Eq. (21)

elite_archive = NaN(max_elite, dimension);
elite_scores  = inf(max_elite, 1);   % store F(Xe) used by Eq. (15) and for Eq. (26) archive ranking
elite_count   = 0;

%===============================
% Helpers
%===============================
% Bound handling operator (keeps solutions within [lb, ub])
clamp = @(Z) max(min(Z, upperbound), lowerbound);
X      = clamp(X);  % ensure initial in-bounds

%===============================
% Main Loop
%===============================
for t = 1:Max_iterations
    % τ = t/T used by Eq. (10), Eq. (14), and Eq. (28)
    progress = t / Max_iterations;

    % safety: keep all agents in-bounds at loop start
    X = clamp(X);

    % ---------- Phase 1: Aspiration (Inspiration in paper) ----------
    % Father (best solution) influences children.
    % In Algorithm 1, father is the best-so-far solution X* (Best_pos).
    % Here, BestNow is the best in the current population at this point.
    [~, best_idx] = min(obj_values);
    BestNow = X(best_idx,:);
    for i = 1:Search
        % Eq. (4) in paper:
        %   Xp1_i = Xi + r * (Xfather - (1+r)*Xi), r ~ U(0,1)
        % Implementation note (no code change):
        %   This line uses two independent random numbers r1 and r2:
        %   Xi + r1 * (Xfather - (1+r2)*Xi)
        Xc = X(i,:) + rand() .* (BestNow - (1 + rand()) .* X(i,:));
        Xc = clamp(Xc);

        % Eq. (5): accept Xp1_i only if improved
        fc = fitness(Xc);
        if fc < obj_values(i)
            X(i,:) = Xc;
            obj_values(i) = fc;

            % update global best (father X*)
            if fc < Best_score
                Best_score = fc; Best_pos = Xc;
            end

            % Eq. (6): update personal best Pi if improved in Phase 1
            upd_pbest(i);
        end
    end

    % ---------- Phase 2: Discipline ----------
    % Eq. (10): βt schedule
    beta_t = beta0 / (1 + (alpha * progress)^0.5);

    for i = 1:Search
        % Eq. (7): Euclidean distances to build K-NN
        D = sqrt(sum((X - X(i,:)).^2,2));
        [~, ord] = sort(D);

        % Select K nearest neighbors (exclude self)
        K_eff = min(K, Search - 1);
        neigh = X(ord(2:K_eff+1), :);
        f_neigh = obj_values(ord(2:K_eff+1));

        % Eq. (8): pick worst M neighbors; compute Wavg
        M_eff = min(M, size(neigh,1));
        [~, widx] = sort(f_neigh,'descend');
        Wavg = mean(neigh(widx(1:M_eff), :), 1);  % Wavg_i

        % Eq. (9): Xp2_i = Xi + βt·r'·(Xi − Wavg), r'~U(0.5,1.5)
        Xd = X(i,:) + beta_t * (0.5 + rand) .* (X(i,:) - Wavg);
        Xd = clamp(Xd);

        % Eq. (11): accept Xp2_i only if improved
        fd = fitness(Xd);
        if fd < obj_values(i)
            X(i,:) = Xd;
            obj_values(i) = fd;

            % update global best (father X*)
            if fd < Best_score
                Best_score = fd; Best_pos = Xd;
            end

            % Eq. (12): update personal best Pi if improved in Phase 2
            upd_pbest(i);
        end
    end

    % ---------- Phase 3: Mentorship ----------
    % Eq. (14): w1 = τ^2, w2 = 1 − w1
    w1 = progress^2;
    w2 = 1 - w1;

    % Randomly select one elite solution Xe from archive E (paper selects a random elite from E)
    % Implementation note: one random elite is used for all i in this iteration.
    if elite_count > 0
        idxE       = randi(elite_count);
        elite_rand = elite_archive(idxE, :);
        elite_f    = elite_scores(idxE);   % F(Xe) used in Eq. (15)
    else
        elite_rand = Best_pos;             % fallback if archive empty
        elite_f    = Best_score;
    end

    eps_den = 1e-20; % ε = 1e−20 in Eq. (15)

    for i = 1:Search
        % Eq. (15): protected personal best weight
        % if (F(Pi) − F(Xe)) / (|F(Xe)| + ε) > 0.1  => w2_adj = 0.2*w2 else w2_adj = w2
        rel_gap = (personal_best_score(i) - elite_f) / max(abs(elite_f), eps_den);
        if rel_gap > 0.1
            w2_adj = 0.2 * w2;
        else
            w2_adj = w2;
        end

        % Eq. (13): Xp3_i = Xi + w1*(Xe − Xi) + w2_adj*(Pi − Xi)
        Xe = X(i,:) + w1 * (elite_rand - X(i,:)) + w2_adj * (personal_best_pos(i,:) - X(i,:));
        Xe = clamp(Xe);

        % Eq. (16): accept Xp3_i only if improved
        fe = fitness(Xe);
        if fe < obj_values(i)
            X(i,:) = Xe;
            obj_values(i) = fe;

            % update global best (father X*)
            if fe < Best_score
                Best_score = fe; Best_pos = Xe;
            end
        end

        % Eq. (17): update personal best Pi if improved in Phase 3
        if obj_values(i) < personal_best_score(i)
            personal_best_pos(i,:) = X(i,:);
            personal_best_score(i) = obj_values(i);
        end
    end

    % ===================== Phase 4: Reward–Correction =====================
    % Reward: Eqs. (18)-(25)
    %   Eq. (18) gradient, Eq. (19) direction, Eq. (20) backtracking trial,
    %   Eq. (21) new gradient, Eqs. (22)-(23) BFGS update, Eq. (24) accept, Eq. (25) pbest
    % Correction: Eqs. (27)-(29)
    %   Eq. (27) boundary move, Eq. (28) η schedule, Eq. (29) pbest after correction

    % ---------------- Reward subsection (Light QN for all) ----------------
    tol_grad           = 1e-20; % practical stop if ‖g‖ is extremely small (implementation detail)
    for i = 1:Search
        x = X(i,:);
        B = eye(dimension);                % Bi initialized as identity (Algorithm 1, Step 4)
        fx_curr = obj_values(i);           % F(x)

        for k = 1:max_qn_iter_light
            % Eq. (18): central-difference gradient gi at x using Δlight
            g = zeros(1, dimension);
            for j = 1:dimension
                Xp = x; Xm = x;
                Xp(j) = x(j) + delta_grad_light;
                Xm(j) = x(j) - delta_grad_light;
                Xp = clamp(Xp); Xm = clamp(Xm);
                g(j) = (fitness(Xp) - fitness(Xm)) / (2 * delta_grad_light);
            end
            if norm(g) < tol_grad, break; end

            % Eq. (19): search direction pi = −Bi*gi
            p = -B * g';

            % Eq. (20): backtracking line search trial  x_trial = x + γ*pi
            step_size = 1.0;                 % initial γ (Table 1)
            x_new = x + step_size * p';
            x_new = clamp(x_new);
            f_old = fx_curr;
            f_new = fitness(x_new);

            % Eq. (20) backtracking: γ ← γ/2 until improvement or γ ≤ γmin
            while f_new >= f_old && step_size > 1e-6
                step_size = 0.5 * step_size;
                x_new = x + step_size * p';
                x_new = clamp(x_new);
                f_new = fitness(x_new);
            end

            % Eq. (21): gradient g_new at x_p4 (= accepted trial point)
            g_new = zeros(1, dimension);
            for j = 1:dimension
                Xp = x_new; Xm = x_new;
                Xp(j) = x_new(j) + delta_grad_light;
                Xm(j) = x_new(j) - delta_grad_light;
                Xp = clamp(Xp); Xm = clamp(Xm);
                g_new(j) = (fitness(Xp) - fitness(Xm)) / (2 * delta_grad_light);
            end

            % Eqs. (22)-(23): inverse-Hessian BFGS update
            % Eq. (22): s = x_p4 - x, y = g_new - g, rho = 1/(y^T s)
            % Eq. (23): B+ = (I - rho*s*y^T) B (I - rho*y*s^T) + rho*s*s^T
            s = x_new - x; y = g_new - g;
            if isrow(s), s = s.'; end
            if isrow(y), y = y.'; end
            ys = y.' * s;                         % scalar y^T s
            if ys > 1e-14
                rho = 1 / ys;
                I = eye(dimension);
                V = I - rho * (s * y.');          % (I - rho*s*y^T)
                B = V * B * V.' + rho * (s * s.');% B+
            end

            % Eq. (24) (inner acceptance): keep the improved trial as new x
            if f_new < f_old
                x = x_new;
                fx_curr = f_new;
            else
                break;
            end
        end

        % Eq. (24): accept improved candidate (relative to Xi before Reward)
        % Eq. (25): update personal best Pi after Reward improvement
        fx = fx_curr;
        if fx < obj_values(i)
            X(i,:) = x;
            obj_values(i) = fx;
            if fx < Best_score
                Best_score = fx; Best_pos = x;
            end
            upd_pbest(i); % Eq. (25)
        end
    end

    % -------------- Reward subsection (Strong QN for elites) --------------
    % Applies the same Reward equations (18)-(25) with stronger settings (Δstrong)
    % Strong QN is activated after half iterations (t > T/2) for elite solutions (Algorithm 1)
    tol_grad           = 1e-20; % practical stop if ‖g‖ is extremely small (implementation detail)
    if progress > 0.5 && elite_count > 0
        % refine all currently available elites in the archive
        num_refine = elite_count;
        [~, orderE] = sort(elite_scores(1:num_refine), 'ascend');
        for e = 1:num_refine
            idxE    = orderE(e);
            x       = elite_archive(idxE, :);
            fx_curr = elite_scores(idxE);
            B = eye(dimension);

            for k = 1:max_qn_iter_strong
                % Eq. (18): gradient at x using Δstrong
                g = zeros(1, dimension);
                for j = 1:dimension
                    Xp = x; Xm = x;
                    Xp(j) = x(j) + delta_grad_strong;
                    Xm(j) = x(j) - delta_grad_strong;
                    Xp = clamp(Xp); Xm = clamp(Xm);
                    g(j) = (fitness(Xp) - fitness(Xm)) / (2 * delta_grad_strong);
                end
                if norm(g) < tol_grad, break; end

                % Eq. (19): pi = −Bi*gi
                p = -B * g';

                % Eq. (20): backtracking trial
                step_size = 1.0;
                x_new = x + step_size * p';
                x_new = clamp(x_new);
                f_old = fx_curr;
                f_new = fitness(x_new);
                while f_new >= f_old && step_size > 1e-6
                    step_size = 0.5 * step_size;
                    x_new = x + step_size * p';
                    x_new = clamp(x_new);
                    f_new = fitness(x_new);
                end

                % Eq. (21): gradient at x_p4
                g_new = zeros(1, dimension);
                for j = 1:dimension
                    Xp = x_new; Xm = x_new;
                    Xp(j) = x_new(j) + delta_grad_strong;
                    Xm(j) = x_new(j) - delta_grad_strong;
                    Xp = clamp(Xp); Xm = clamp(Xm);
                    g_new(j) = (fitness(Xp) - fitness(Xm)) / (2 * delta_grad_strong);
                end

                % Eqs. (22)-(23): BFGS update
                s = x_new - x; y = g_new - g;
                if isrow(s), s = s.'; end
                if isrow(y), y = y.'; end
                ys = y.' * s;                     % y^T s
                if ys > 1e-14
                    rho = 1 / ys;
                    I = eye(dimension);
                    V = I - rho * (s * y.');
                    B = V * B * V.' + rho * (s * s.');
                end

                % Eq. (24) (inner acceptance)
                if f_new < f_old
                    x = x_new;
                    fx_curr = f_new;
                else
                    break;
                end
            end

            % Eq. (24): accept improved elite candidate (relative to archive entry)
            % Eq. (25): pbest update is not applied here because archive stores elites directly
            fx = fx_curr;
            if fx < elite_scores(idxE)
                elite_archive(idxE,:) = x;
                elite_scores(idxE)    = fx;
                if fx < Best_score
                    Best_score = fx; Best_pos = x;
                end
            end
        end
    end

    % ---------- Elite Archive Update (Eq. (26)) [INSIDE REWARD] ----------
    % Eq. (26): if |E| < m add Xi, else replace worst elite if Xi is better
    [~, elite_order] = sort(obj_values);
    top_solutions = X(elite_order(1:min(max_elite,Search)),:);

    for e = 1:size(top_solutions,1)
        cand  = top_solutions(e,:);
        fcand = fitness(cand);

        if elite_count < max_elite
            elite_count = elite_count + 1;
            elite_archive(elite_count,:) = cand;
            elite_scores(elite_count)    = fcand;   % store F(Xe)
        else
            [worst_val, worst_idx] = max(elite_scores(1:elite_count));
            if fcand < worst_val
                elite_archive(worst_idx,:) = cand;
                elite_scores(worst_idx)    = fcand;
            end
        end
    end
    % ----------------------------------------------------------------------

    % ------------------- Correction / Punishment subsection -------------------
    % Implements Eq. (27) with η from Eq. (28)
    if mod(t, 15) == 0
        [~, sorted_idx] = sort(obj_values, 'descend');
        numWorst      = max(1, round(0.2 * Search));  % bottom 20% (Algorithm 1)
        worst_indices = sorted_idx(1:numWorst);
        num_to_send   = min(5, numWorst);             % Nc = 5 (Table 1)
        pick = worst_indices(randperm(numWorst, num_to_send));

        % Eq. (28): η = 0.4*(1 − τ)^2 (adaptive correction strength)
        step_scale = 0.4 * (1 - progress)^2;

        for k = 1:num_to_send
            idx = pick(k);

            % Eq. (27): Xc = X + η*(b − X), where b is a random boundary point per dimension
            boundary = lowerbound + (upperbound - lowerbound) .* randi([0,1], 1, dimension);
            step_vector = step_scale * (boundary - X(idx,:));
            X_new = X(idx,:) + step_vector;
            X_new = clamp(X_new);

            % apply correction move and evaluate
            X(idx,:) = X_new;
            obj_values(idx) = fitness(X_new);

            % track global best if improved
            if obj_values(idx) < Best_score
                Best_score = obj_values(idx); Best_pos = X(idx,:);
            end
        end
    end

    % Eq. (29): after Correction, update personal bests for corrected improvements
    % Implementation note: this applies the same Pi update rule to all agents (superset of Eq. (29)).
    arrayfun(@upd_pbest, 1:Search);

    % end-of-iteration safety clamp
    X = clamp(X);

    % record convergence
    Optimization_curve(t) = Best_score;
end

% ========= Nested helper: update personal best for agent idx =========
% Personal-best selection rule used in:
% Phase 1: Eq. (6), Phase 2: Eq. (12), Phase 3: Eq. (17), Reward: Eq. (25), Correction: Eq. (29)
    function upd_pbest(idx)
        if obj_values(idx) < personal_best_score(idx)
            personal_best_score(idx) = obj_values(idx);
            personal_best_pos(idx,:) = X(idx,:);
        end
    end

end
