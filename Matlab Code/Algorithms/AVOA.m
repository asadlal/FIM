function [Best_score, Best_pos, Optimization_curve] = AVOA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness)
    % Initialize positions of vultures
    Positions = lowerbound + (upperbound - lowerbound) .* rand(Search, dimension);

    % Initialize Best position and score
    Best_pos = zeros(1, dimension); 
    Best_score = inf;

    % Probability parameters
    P1 = 0.6;
    P2 = 0.4;
    P3 = 0.6;

    % Coefficients
    L1 = 0.8;
    L2 = 0.2;
    w = 2.5;
    beta = 1.5;

    % Initialize optimization curve to store best score per iteration
    Optimization_curve = zeros(1, Max_iterations);

    % Main loop
    for t = 1:Max_iterations
        % Loop through each vulture
        for i = 1:Search
            % Calculate fitness of current vulture
            current_fitness = fitness(Positions(i, :));

            % Update Best position if current is better
            if current_fitness < Best_score
                Best_score = current_fitness;
                Best_pos = Positions(i, :);
            end
        end

        % Update the positions of all vultures
        for i = 1:Search
            % Generate random numbers for movement
            h = -2 + (2 + 2) * rand;      % Random number between -2 and 2
            z = -1 + (1 + 1) * rand;      % Random number between -1 and 1
            u = rand; v = rand;           % Random numbers between 0 and 1
            rand1 = rand; rand2 = rand;
            rand3 = rand; rand4 = rand;
            rand5 = rand; rand6 = rand;

            % Update position based on the probability P1, P2, P3
            if rand1 < P1
                % Exploration phase influenced by randomness
                Positions(i, :) = Best_pos + w * (L1 * h * Best_pos - L2 * Positions(i, :));
            elseif rand2 < P2
                % Exploitation phase focusing on best solutions
                Positions(i, :) = Positions(i, :) + beta * (rand3 * Best_pos - rand4 * Positions(i, :));
            elseif rand5 < P3
                % Combination of exploration and exploitation
                Positions(i, :) = Best_pos + u * (L1 * h * Best_pos - L2 * Positions(i, :)) + v * (rand6 * z);
            end

            % Enforce boundary constraints
            Positions(i, :) = max(min(Positions(i, :), upperbound), lowerbound);
        end

        % Update the optimization curve with the best score found so far
        Optimization_curve(t) = Best_score;

    % Return final results
    end
end