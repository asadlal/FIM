function [Best_score, Best_pos, Optimization_curve] = WO(Search, MaxFEs, lowerbound, upperbound, dimension, fitness)
    % WO (Wolverine Optimization) - FE-based version
    % The second input is now MaxFEs, not Max_iterations.
    % Remaining WO movement logic is kept the same.

    % Initialization
    X = lowerbound + rand(Search, dimension) .* (upperbound - lowerbound);
    fit = inf(Search, 1);

    % FE counter
    FEs = 0;

    % Initialize Best position and score
    Best_pos = zeros(1, dimension);
    Best_score = inf;

    % Preallocate convergence curve by function evaluations
    Optimization_curve = zeros(MaxFEs, 1);

    % Evaluate fitness for each solution
    for i = 1:Search
        if FEs >= MaxFEs
            break;
        end

        L = X(i, :);
        fit(i) = fitness(L);
        FEs = FEs + 1;

        if fit(i) < Best_score
            Best_score = fit(i);
            Best_pos = X(i, :);
        end

        Optimization_curve(FEs) = Best_score;
    end

    % Main Loop: stop by MaxFEs
    while FEs < MaxFEs
        % Find the best solution
        [best, blocation] = min(fit);
        if best < Best_score
            Best_score = best;
            Best_pos = X(blocation, :);
        end

        % Wolverine behavior: exploit and explore
        for i = 1:Search
            if FEs >= MaxFEs
                break;
            end

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
            FEs = FEs + 1;

            % Update best score immediately after the new evaluation
            if fit(i) < Best_score
                Best_score = fit(i);
                Best_pos = X(i, :);
            end

            % Save best score at this FE
            Optimization_curve(FEs) = Best_score;
        end
    end

    % If any unused curve entries remain, fill them with the final best value
    if FEs < MaxFEs
        Optimization_curve(FEs+1:MaxFEs) = Best_score;
    end
end
