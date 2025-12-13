function [Best_score, Best_pos, Optimization_curve] = MVO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness)

    % Constants
    p = 6;              % Exploitation accuracy
    WEP_min = 0.2;      % Min Wormhole Existence Probability
    WEP_max = 1;        % Max Wormhole Existence Probability

    % --- make bounds robust (works for scalar or per-dimension vectors) ---
    lb = lowerbound(:)';                 % 1×D
    ub = upperbound(:)';                 % 1×D
    if isscalar(lb), lb = repmat(lb, 1, dimension); end
    if isscalar(ub), ub = repmat(ub, 1, dimension); end

    % Initialize the universe positions
    universes = rand(Search, dimension) .* (ub - lb) + lb;
    Best_pos = universes(1, :);
    Best_score = fitness(Best_pos);
    Optimization_curve = zeros(1, Max_iterations);

    % Main Loop
    for t = 1:Max_iterations
        for i = 1:Search
            % Evaluate fitness of each universe
            universe_fitness = fitness(universes(i, :));
            
            % Update the best position if the current position is better
            if universe_fitness < Best_score
                Best_score = universe_fitness;
                Best_pos = universes(i, :);
            end

            % Update WEP dynamically
            WEP = WEP_min + (WEP_max - WEP_min) * (t / Max_iterations); % Increasing WEP as iterations increase

            % Explore or exploit based on WEP
            if rand() < WEP
                % Exploration (wormhole jump)
                distance = rand(1, dimension) .* (Best_pos - universes(i, :)); 
                universes(i, :) = universes(i, :) + distance;  % Wormhole jump
            else
                % Exploitation (fine-tuning based on p)
                step_size = (Best_pos - universes(i, :)) / (1 + p);  % Smaller steps for higher p
                universes(i, :) = universes(i, :) + step_size;  % Exploitation phase
            end

            % Bound the universes' positions
            universes(i, :) = min(max(universes(i, :), lb), ub);
        end

    end

end
