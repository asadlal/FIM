function [Best_score, Best_pos, Optimization_curve] = WO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness)
    % Initialization
    X = lowerbound + rand(Search, dimension) .* (upperbound - lowerbound);
    fit = zeros(Search, 1);

    % Evaluate fitness for each solution
    for i = 1:Search
        L = X(i, :);
        fit(i) = fitness(L);
    end

    % Preallocate for speed.
    best_so_far = zeros(Max_iterations, 1);

    % Main Loop
    for t = 1:Max_iterations
        % Find the best solution
        [best, blocation] = min(fit);
        Best_pos = X(blocation, :);

        % Update best scores
        if t == 1
            Best_score = best;
        elseif best < Best_score
            Best_score = best;
        end

        % Wolverine behavior: exploit and explore
        for i = 1:Search
            if rand() < 0.5  % Exploitation
                % Move towards the best position
                direction = rand(size(X(i, :))) .* (Best_pos - X(i, :));
                X(i, :) = X(i, :) + direction;
            else  % Exploration
                % Random walk
                X(i, :) = lowerbound + rand(size(X(i, :))) .* (upperbound - lowerbound);
            end
            
            % Ensure the position is within bounds
            X(i, :) = max(X(i, :), lowerbound);
            X(i, :) = min(X(i, :), upperbound);
            
            % Evaluate fitness
            fit(i) = fitness(X(i, :));
        end

        best_so_far(t) = Best_score; % Save best score
    end

    Optimization_curve = best_so_far;
end