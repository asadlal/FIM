function [Best_score, Best_pos, Optimization_curve] = GWO(Search, Max_iterations, lowerbound, upperbound, dimension, fitness)

    % Initialize positions of wolves
    Positions = lowerbound + (upperbound - lowerbound) .* rand(Search, dimension);
    
    % Initialize best, beta, and delta positions and scores
    Best_pos = zeros(1, dimension); 
    Best_score = inf;
    Beta_pos = zeros(1, dimension); 
    Beta_score = inf;
    Delta_pos = zeros(1, dimension); 
    Delta_score = inf;
    
    % Initialize optimization curve to store best score per iteration
    Optimization_curve = zeros(1, Max_iterations);

    % Main loop
    for t = 1:Max_iterations
        % Linearly decrease parameter 'a' from 2 to 0 over iterations
        a = 2 - t * (2 / Max_iterations);

        % Loop through each wolf to update Best, Beta, and Delta based on fitness
        for i = 1:Search
            % Calculate fitness of current wolf
            current_fitness = fitness(Positions(i, :));
            
            % Update Best, Beta, and Delta wolves based on fitness
            if current_fitness < Best_score
                Delta_score = Beta_score; 
                Delta_pos = Beta_pos;
                Beta_score = Best_score; 
                Beta_pos = Best_pos;
                Best_score = current_fitness; 
                Best_pos = Positions(i, :);
            elseif current_fitness < Beta_score
                Delta_score = Beta_score; 
                Delta_pos = Beta_pos;
                Beta_score = current_fitness; 
                Beta_pos = Positions(i, :);
            elseif current_fitness < Delta_score
                Delta_score = current_fitness; 
                Delta_pos = Positions(i, :);
            end
        end

        % Update the positions of all wolves
        for i = 1:Search
            % Calculate distances and positions based on Best, Beta, and Delta
            r1 = rand(1, dimension); r2 = rand(1, dimension);
            A1 = 2 * a * r1 - a; 
            C1 = 2 * r2;
            D_best = abs(C1 .* Best_pos - Positions(i, :));
            X1 = Best_pos - A1 .* D_best;

            r1 = rand(1, dimension); r2 = rand(1, dimension);
            A2 = 2 * a * r1 - a; 
            C2 = 2 * r2;
            D_beta = abs(C2 .* Beta_pos - Positions(i, :));
            X2 = Beta_pos - A2 .* D_beta;

            r1 = rand(1, dimension); r2 = rand(1, dimension);
            A3 = 2 * a * r1 - a; 
            C3 = 2 * r2;
            D_delta = abs(C3 .* Delta_pos - Positions(i, :));
            X3 = Delta_pos - A3 .* D_delta;

            % Update position of wolf by averaging the positions influenced by Best, Beta, and Delta
            Positions(i, :) = (X1 + X2 + X3) / 3;
        end

        % Enforce boundary constraints
        Positions = max(min(Positions, upperbound), lowerbound);

        % Update the optimization curve with the best score found so far
        Optimization_curve(t) = Best_score;
    end
end