function [Best_score, Best_pos, Optimization_curve] = TSA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness)

    % --- make bounds robust (works for scalar or per-dimension vectors) ---
    lb = lowerbound(:)';                 % 1×D
    ub = upperbound(:)';                 % 1×D
    if isscalar(lb), lb = repmat(lb, 1, dimension); end
    if isscalar(ub), ub = repmat(ub, 1, dimension); end

    % Initialize agents' positions randomly within the search bounds
    agents = rand(Search, dimension) .* (ub - lb) + lb;
    Best_pos = agents(1, :);   % Best solution found
    Best_score = fitness(Best_pos); % Best score (fitness)
    Optimization_curve = zeros(1, Max_iterations);  % To store best fitness at each iteration
    
    % Tunicate parameters
    Pmin = 1;  % Minimum probability
    Pmax = 4;  % Maximum probability

    % Main loop for Tunicate Search Algorithm
    for t = 1:Max_iterations
        % Update random parameters c1, c2, c3 in each iteration
        c1 = rand(1, dimension);  % Random coefficient for exploration
        c2 = rand(1, dimension);  % Random coefficient for exploitation
        c3 = rand(1, dimension);  % Random coefficient for exploitation
        
        % For each agent in the population
        for i = 1:Search
            % Evaluate fitness of each agent
            agent_fitness = fitness(agents(i, :));
            
            % Update the best position if the current position is better
            if agent_fitness < Best_score
                Best_score = agent_fitness;
                Best_pos = agents(i, :);
            end
            
            % Tunicate movement strategy based on random parameters
            P = Pmin + (Pmax - Pmin) * rand(1, 1);  % Random probability in the range [Pmin, Pmax]
            
            % Generate binary vector U (Exploration vs Exploitation)
            U = rand(1, dimension) > 0.5;  % Binary vector (exploration if 1, exploitation if 0)
            
            % Exploration phase
            if rand() < P
                % Random movement strategy (exploration)
                movement = c1 .* (Best_pos - agents(i, :)) + c2 .* rand(1, dimension);
                agents(i, :) = agents(i, :) + movement;
            else
                % Exploitation phase
                movement = c3 .* (Best_pos - agents(i, :));
                agents(i, :) = agents(i, :) + movement;
            end
            
            % Apply binary vector U to switch between Lévy or Brownian update
            agents(i, :) = agents(i, :) .* U + Best_pos .* (1 - U);

            % Bound the agents' positions to stay within the search space limits
            agents(i, :) = max(min(agents(i, :), upperbound), lowerbound);
        end
    end
end
