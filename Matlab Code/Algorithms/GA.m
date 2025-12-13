function [Best_score, Best_pos, Optimization_curve] = GA(Search, Max_iterations, lowerbound, upperbound, dimension, fitness)
    % Problem Definition
    CostFunction = fitness;  % Use the provided fitness function
    
    VarMin = lowerbound;  % Lower Bound of Variables
    VarMax = upperbound;  % Upper Bound of Variables

    %% GA Parameters
    MaxIt = Max_iterations;   % Maximum Number of Iterations
    nPop = Search;            % Population Size
    
    % GA Operators
    pc = 0.8;    % Crossover Probability
    nc = 2 * round(pc * nPop / 2);  % Number of Offsprings
    pm = 0.1;    % Mutation Probability
    nm = round(pm * nPop);  % Number of Mutants

    %% Initialization
    empty_individual.Position = [];
    empty_individual.Cost = [];

    pop = repmat(empty_individual, nPop, 1);
    GlobalBest.Cost = inf;

    % Initialize Population
    for i = 1:nPop
        pop(i).Position = unifrnd(VarMin, VarMax, [1, dimension]);
        pop(i).Cost = CostFunction(pop(i).Position);

        % Update Global Best
        if pop(i).Cost < GlobalBest.Cost
            GlobalBest = pop(i);
        end
    end

    % Optimization results
    Best_cost = zeros(MaxIt, 1);

    %% GA Main Loop
    for it = 1:MaxIt
        % Selection
        pop = sortPopulation(pop);  % Sort population based on fitness
        pop = pop(1:round(nPop / 2));  % Elitism: keep the best half
        
        % Crossover
        offspring = [];
        while size(offspring, 1) < nc
            p1 = pop(randi([1, size(pop, 1)]));  % Randomly select parents
            p2 = pop(randi([1, size(pop, 1)]));  % Ensure p1 and p2 are different
            
            [o1, o2] = crossover(p1, p2);  % Perform crossover
            
            offspring = [offspring; o1; o2];  % Add offsprings
        end
        
        % Mutation
        for i = 1:nm
            idx = randi([1, size(offspring, 1)]);  % Randomly select an individual
            offspring(idx) = mutate(offspring(idx));  % Perform mutation
        end
        
        % Create new population
        pop = [pop; offspring];
        
        % Update the Global Best
        for i = 1:size(pop, 1)
            if pop(i).Cost < GlobalBest.Cost
                GlobalBest = pop(i);
            end
        end
        
        % Store the Best Cost Value
        Best_cost(it) = GlobalBest.Cost;
    end

    % Return the final best solution and the optimization curve
    Best_pos = GlobalBest.Position;
    Best_score = GlobalBest.Cost;  % Ensure Best_score is assigned
    Optimization_curve = Best_cost;
end

% Utility Functions
function pop = sortPopulation(pop)
    % Sort population based on cost
    [~, idx] = sort([pop.Cost]);
    pop = pop(idx);
end

function [o1, o2] = crossover(p1, p2)
    % Perform crossover between two parents
    alpha = rand;  % Random crossover weight
    o1.Position = alpha * p1.Position + (1 - alpha) * p2.Position;  % Offspring 1
    o2.Position = (1 - alpha) * p1.Position + alpha * p2.Position;  % Offspring 2
    o1.Cost = p1.Cost;  % Placeholder, will be evaluated later
    o2.Cost = p2.Cost;  % Placeholder, will be evaluated later
end

function mutated_individual = mutate(individual)
    % Perform mutation on an individual
    mr = 0.1;  % Mutation rate
    mutation_vector = unifrnd(-0.5, 0.5, size(individual.Position));  % Random mutation
    mutated_individual.Position = individual.Position + mr * mutation_vector;  % Apply mutation
    mutated_individual.Cost = individual.Cost;  % Placeholder, will be evaluated later
end