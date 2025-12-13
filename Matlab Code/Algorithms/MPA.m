function [Best_score, Best_pos, Optimization_curve] = MPA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness)
    
    % Constant Parameters
    P = 0.5;          % Probability threshold for behavior switch
    FADs = 0.2;       % Fish aggregating devices parameter (controls exploration intensity)

    % --- make bounds robust (works for scalar or per-dimension vectors) ---
    lb = lowerbound(:)';                 % 1×D
    ub = upperbound(:)';                 % 1×D
    if isscalar(lb), lb = repmat(lb, 1, dimension); end
    if isscalar(ub), ub = repmat(ub, 1, dimension); end

    % Initialize agents' positions randomly within the search bounds
    agents = rand(Search, dimension) .* (ub - lb) + lb;
    Best_pos = agents(1, :);
    Best_score = fitness(Best_pos);
    Optimization_curve = zeros(1, Max_iterations);  % To store best fitness at each iteration

    % Main loop
    for t = 1:Max_iterations
        for i = 1:Search
            % Evaluate fitness of each agent
            agent_fitness = fitness(agents(i, :));
            
            % Update the best position if current position is better
            if agent_fitness < Best_score
                Best_score = agent_fitness;
                Best_pos = agents(i, :);
            end

            % Generate binary vector U and random vector R
            U = rand(1, dimension) > P;  % Binary vector U for behavior switch
            R = rand(1, dimension);      % Random vector in [0, 1]

            % Update positions based on MPA equations
            if rand() < FADs
                % Exploration phase using Lévy flight
                beta = 1.5;
                sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
                u = randn(1, dimension) * sigma;
                v = randn(1, dimension);
                D = (u ./ abs(v).^(1 / beta)) .* (Best_pos - agents(i, :));
                agents(i, :) = agents(i, :) + R .* D;
            else
                % Exploitation phase using Brownian motion
                D = randn(1, dimension) .* (Best_pos - agents(i, :));
                agents(i, :) = Best_pos + R .* D;
            end

            % Apply binary vector U to switch between Lévy or Brownian update
            agents(i, :) = agents(i, :) .* U + Best_pos .* (1 - U);

            % Bound the agents' positions to stay within search space limits
            agents(i, :) = min(max(agents(i, :), lb), ub);
        end
    end
end
