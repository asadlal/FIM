function [Best_score, Best_pos, Optimization_curve] = GSA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness)

    % Define constants
    Alpha = 20;    % Decay parameter for gravitational constant
    G_0 = 100;      % Initial gravitational constant
    R_norm = 2;     % Euclidean norm for force calculation (can use different norms like R_norm=1)

    % Initialize population (Position of agents)
    X = lowerbound + rand(Search, dimension) .* (upperbound - lowerbound);
    velocity = zeros(Search, dimension);
    fit = zeros(Search, 1);

    % Evaluate fitness of each agent (for minimization)
    for i = 1:Search
        L = X(i, :);
        fit(i) = fitness(L);
    end
    
    % Preallocate optimization curve for tracking best score per iteration
    Optimization_curve = zeros(Max_iterations, 1);
    Best_score = inf; % Initialize best score as infinity for minimization

    % Main loop of GSA
    for t = 1:Max_iterations
        % Update best solution found so far (for minimization)
        [current_best, blocation] = min(fit);
        if current_best < Best_score
            Best_score = current_best;
            Best_pos = X(blocation, :);
        end
        
        % Compute gravitational constant G and masses (G decreases over time)
        G = G_0 * exp(-Alpha * t / Max_iterations);  % Gravitational constant with exponential decay
        mass = (fit - max(fit)) ./ (min(fit) - max(fit) + 1e-8);  % Normalized mass for each agent
        mass = mass / sum(mass);

        % Calculate acceleration for each agent using R_norm
        for i = 1:Search
            F_total = zeros(1, dimension);  % Total force on agent `i`
            for j = 1:Search
                if j ~= i
                    distance = norm(X(i, :) - X(j, :), R_norm) + 1e-6;
                    F_total = F_total + G * ((mass(i) * mass(j)) / distance^2) .* (X(j, :) - X(i, :));
                end
            end
            acceleration = F_total / (mass(i) + 1e-6);
            velocity(i, :) = rand * velocity(i, :) + acceleration; % Update velocity
            X(i, :) = X(i, :) + velocity(i, :);  % Update position
        end
        
        % Boundary control to keep agents within bounds
        X = max(X, lowerbound);
        X = min(X, upperbound);

        % Re-evaluate fitness of each agent after movement
        for i = 1:Search
            L = X(i, :);
            fit(i) = fitness(L);
        end
        
        % Store the best score found in this iteration for tracking
        Optimization_curve(t) = Best_score;
    end
end