function [Best_score, Best_pos, Optimization_curve] = RSA(Search, MaxFEs, lowerbound, upperbound, dimension, fitness)
% RSA
% FE-based version: the second input is MaxFEs, not Max_iterations.
% Every call to fitness() is counted as one function evaluation.
% The algorithm stops when FEs >= MaxFEs.

    MaxFEs = floor(MaxFEs);
    if MaxFEs < 1
        MaxFEs = 1;
    end

    % Nominal iteration count used only for the original ES reduction term.
    % The real stopping condition is FEs >= MaxFEs.
    Max_iterations = max(1, floor((MaxFEs - Search) / max(1, Search)));

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

    % FE counter and convergence curve
    FEs = 0;
    Optimization_curve = zeros(1, MaxFEs);

    % Main loop
    while FEs < MaxFEs
        % Update ES as a decreasing random value within [-2, 2]
        ES = ES - (4 * rand) / Max_iterations; % Gradual reduction each nominal iteration

        % Loop through each reptile and evaluate current fitness
        for i = 1:Search
            if FEs >= MaxFEs
                break;
            end

            % Calculate fitness of current reptile
            current_fitness = fitness(Positions(i, :));
            FEs = FEs + 1;

            % Update Best position if current is better
            if current_fitness < Best_score
                Best_score = current_fitness;
                Best_pos = Positions(i, :);
            end

            % Update the optimization curve with the best score found so far
            Optimization_curve(FEs) = Best_score;
        end

        if FEs >= MaxFEs
            break;
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
    end

    % Keep only evaluated FE points
    if FEs == 0
        Optimization_curve = Best_score;
    else
        Optimization_curve = Optimization_curve(1:FEs);
    end
end
