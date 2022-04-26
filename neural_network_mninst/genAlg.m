%% Genetic algorithm

classdef genAlg
    % Neural Network object for training
    
    properties
        % fitness of the models
        fitness
        % errors and gradients
        grad
        % batch size
        m
        % optimizer
        optim
        %error
        error4
        
    end

    methods

        function obj = genAlg(nnMatrix, test_data, test_images, test_labels,  mutRate)
            %function to optimize the weights of h1 and h2
            % the nnMatrix is a matrix with xdim = number of models and
            % ydim = number of hyperparameters arrays: (W, b)x layers

            %set counter
            counter = 0;

            %set fitness scores
            fitness = zeros(length(nnMatrix), 1 );

            %estimate fitness of each model
            %iterate over models
            % ------> Change fitness evaluation so it can be called in the
            % class itself (new method)
            for model = nnMatrix            

                counter = counter +1;

                %calculate accuracy
                hits = 0;
                n = length(test_data);
                for i = 1:n
            
                    out = model.predict(test_images(:,i)); % model prediction vector
                    [~, num] = max(out); % Find highest prediction score
            
                    if test_labels(i) == (num-1)
                        hits = hits + 1; % Count the number of correct classifications
                    end       
                end
                %calculate accuracy
                accuracy = hits/n;
                fitness(counter) = accuracy;
                
            end
            obj.fitness = fitness;

            %get the 3 top performer

            % use sort approach to get the sorted fitness list as well as

            % choose top 3 models
            [values, index] = sort(obj.fitness)
            
            %flatten and extr
            %reshape(model.',1, [])
            
            % do cross over and create 7 more children

            % mutate all of them depending on the mutation rate
            % generate random number between -1 and 1 then divide by 100 or
            % 1000
            % repeat generation process unitl reaches 95% or max
            % generations of 100

        end



    end
    
 end