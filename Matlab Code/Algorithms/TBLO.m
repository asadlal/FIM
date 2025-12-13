function [Best_score, Best_pos, Optimization_curve] = TBLO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness)
    % Initialization
    X = lowerbound + rand(Search, dimension) .* (upperbound - lowerbound);
    fit = zeros(Search, 1);

    % Evaluate fitness for each solution
    for i = 1:Search
        L = X(i, :);
        fit(i) = fitness(L);
    end
    
    % Preallocate for speed
    best_so_far = zeros(Max_iterations, 1);

    % Main Loop
    for t = 1:Max_iterations
        % Sort population based on fitness
        [sorted_fit, sorted_idx] = sort(fit);
        X = X(sorted_idx, :);  % Sort the population
        fit = sorted_fit;  % Sort the fitness values

        % Teacher's solution is the best one
        teacher = X(1, :);
        teacher_fitness = fit(1);

        % Update each individual's position and fitness
        for i = 1:Search
            % Learning process
            learning_factor = rand;  % Random learning factor between 0 and 1
            X(i, :) = X(i, :) + learning_factor * (teacher - X(i, :));
            
            % Enforce bounds
            X(i, :) = max(X(i, :), lowerbound);
            X(i, :) = min(X(i, :), upperbound);

            % Evaluate new fitness
            fit(i) = fitness(X(i, :));
        end

        % Update the best score so far
        best_so_far(t) = min(fit);
    end

    % Find the best solution and its fitness
    [Best_score, idx] = min(fit);
    Best_pos = X(idx, :);
    Optimization_curve = best_so_far;
end