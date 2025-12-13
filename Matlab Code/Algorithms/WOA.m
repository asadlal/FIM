function [Best_score, Best_pos, Optimization_curve] = WOA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness)
    % Initialize whale positions randomly within the search bounds
    Positions = lowerbound + (upperbound - lowerbound) .* rand(Search, dimension);
    Best_pos = Positions(1, :); % Initialize best position
    Best_score = fitness(Best_pos); % Initialize best fitness score
    
    % Initialize an array to track the best score for each iteration
    Optimization_curve = zeros(1, Max_iterations);

    % Main optimization loop
    for t = 1:Max_iterations
        % Linearly decrease parameter 'a' from 2 to 0
        a = 2 - (t * (2 / Max_iterations));

        % Update each whale's position
        for i = 1:Search
            % Calculate the current fitness of each whale
            current_fitness = fitness(Positions(i, :));

            % Update the best position if current position is better
            if current_fitness < Best_score
                Best_score = current_fitness;
                Best_pos = Positions(i, :);
            end

            % Generate random parameters 'r' and 'l'
            r = rand(1, dimension); % Random vector in [0, 1]
            l = -1 + (2 * rand); % Random number in [-1, 1]

            % Compute A and C coefficients
            A = 2 * a * r - a;
            C = 2 * r;

            % Exploitation phase if |A| < 1
            if abs(A) < 1
                % Move towards the best known position
                D = abs(C .* Best_pos - Positions(i, :));
                Positions(i, :) = Best_pos - A .* D;
            else
                % Exploration phase if |A| >= 1
                random_agent_index = randi([1, Search]);
                random_agent = Positions(random_agent_index, :);
                D = abs(C .* random_agent - Positions(i, :));
                Positions(i, :) = random_agent - A .* D;
            end

            % Spiral updating mechanism with probability p < 0.5
            p = rand();
            if p < 0.5
                % Update position using a logarithmic spiral
                distance_to_best = abs(Best_pos - Positions(i, :));
                Positions(i, :) = distance_to_best .* exp(1 * l) .* cos(2 * pi * l) + Best_pos;
            end
        end

        % Ensure all Positions are within the defined bounds
        Positions = max(min(Positions, upperbound), lowerbound);

        % Store the best score for this iteration in the Optimization_curve
        Optimization_curve(t) = Best_score;
    end
end