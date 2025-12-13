function [Best_score, Best_pos, Optimization_curve] = RSA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness)

    % Initialize positions of reptiles
    Positions = lowerbound + (upperbound - lowerbound) .* rand(Search, dimension);

    % Initialize Best position and score
    Best_pos = zeros(1, dimension); 
    Best_score = inf;

    % Sensitive parameters
    beta1 = 0.01;
    beta2 = 0.1;

    % Initialize the Evolutionary Sense (ES) for adaptive exploration-exploitation
    ES = 2 - 4 * rand;  % Randomly initialized between 2 and -2

    % Initialize optimization curve to store best score per iteration
    Optimization_curve = zeros(1, Max_iterations);

    % Main loop
    for t = 1:Max_iterations
        % Update ES as a decreasing random value within [-2, 2]
        ES = ES - (4 * rand) / Max_iterations; % Gradual reduction each iteration

        % Loop through each reptile
        for i = 1:Search
            % Calculate fitness of current reptile
            current_fitness = fitness(Positions(i, :));

            % Update Best position if current is better
            if current_fitness < Best_score
                Best_score = current_fitness;
                Best_pos = Positions(i, :);
            end
        end

        % Update the positions of all reptiles
        for i = 1:Search
            % Generate random numbers for movement
            rand1 = rand; 
            rand2 = rand;
            rand3 = rand;
            
            % Exploration/Exploitation phase controlled by ES
            if rand1 < 0.5
                % Exploration phase with a slight random influence
                Positions(i, :) = Positions(i, :) + beta1 * ES * (rand2 * Best_pos - rand3 * Positions(i, :));
            else
                % Exploitation phase, moving closer to the best solution
                Positions(i, :) = Positions(i, :) + beta2 * ES * (Best_pos - Positions(i, :));
            end

            % Enforce boundary constraints
            Positions(i, :) = max(min(Positions(i, :), upperbound), lowerbound);
        end

        % Update the optimization curve with the best score found so far
        Optimization_curve(t) = Best_score;
    end
end