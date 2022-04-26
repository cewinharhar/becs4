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
        %sorted fitness values
        sorted_fitness
        %index of original fitness to index into nnMatrix to get best
        %performers
        index
        % an array of models
        nnMatrix
        parent1
        % new population
        new_population
        
    end

    methods

        function obj = genAlg(nnMatrix, test_data, test_images, test_labels,  mutRate)
            %function to optimize the weights of h1 and h2
            % the nnMatrix is a matrix with xdim = number of models and
            % ydim = number of hyperparameters arrays: (W, b)x layers
            obj.nnMatrix = nnMatrix;

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

            % use sort approach to get the sorted fitness list as well as
            [obj.sorted_fitness, obj.index] = sort(obj.fitness, 'descend');
            
                        %obj.new_population
            % choose top 3 models
            model_1 = obj.nnMatrix(obj.index(1));
            model_2 = obj.nnMatrix(obj.index(2));
            model_3 = obj.nnMatrix(obj.index(3));

            % Get the weights and biases of each model, flatten and store
            % for cross over section flatten and extr
            %reshape(model.',1, [])

            % Model 1
            M1W1 = reshape(model_1.grad.W1.',1, []);
            M1W2 = reshape(model_1.grad.W2.',1, []);
            M1W3 = reshape(model_1.grad.W3.',1, []);

            M1b1 = model_1.grad.b1;
            M1b2 = model_1.grad.b2;
            M1b3 = model_1.grad.b3;

            obj.parent1 = {M1W1; M1W2; M1W3; M1b1; M1b2; M1b3};
            
            % Model 2
            M2W1 = reshape(model_2.grad.W1.',1, []);
            M2W2 = reshape(model_2.grad.W2.',1, []);
            M2W3 = reshape(model_2.grad.W3.',1, []);

            M2b1 = model_2.grad.b1;
            M2b2 = model_2.grad.b2;
            M2b3 = model_2.grad.b3;
            
            parent2 = {M2W1; M2W2; M2W3; M2b1; M2b2; M2b3};
            % Model 3
            M3W1 = reshape(model_3.grad.W1.',1, []);
            M3W2 = reshape(model_3.grad.W2.',1, []);
            M3W3 = reshape(model_3.grad.W3.',1, []);

            M3b1 = model_3.grad.b1;
            M3b2 = model_3.grad.b2;
            M3b3 = model_3.grad.b3;

            parent3 = {M3W1; M3W2; M3W3; M3b1; M3b2; M3b3};
            % do cross over and create 7 more children
            



            % mutate all of them depending on the mutation rate
            % generate random number between -1 and 1 then divide by 100 or
            % 1000
            % repeat generation process unitl reaches 95% or max
            % generations of 100

        end
    end
% methods(Static)
%         function obj = newPopulation()
%             %obj.new_population
%             % choose top 3 models
%             model_1 = obj.nnMatrix(obj.index(1));
%             model_2 = obj.nnMatrix(obj.index(2));
%             model_3 = obj.nnMatrix(obj.index(3));
% 
%             % Get the weights and biases of each model, flatten and store
%             % for cross over section flatten and extr
%             %reshape(model.',1, [])
% 
%             % Model 1
%             M1W1 = reshape(model_1.grad.W1.',1, []);
%             M1W2 = reshape(model_1.grad.W2.',1, []);
%             M1W3 = reshape(model_1.grad.W3.',1, []);
% 
%             M1b1 = model_1.grad.b1;
%             M1b2 = model_1.grad.b2;
%             M1b3 = model_1.grad.b3;
% 
%             obj.parent1 = {M1W1; M1W2; M1W3; M1b1; M1b2; M1b3};
%             
%             % Model 2
%             M2W1 = reshape(model_2.grad.W1.',1, []);
%             M2W2 = reshape(model_2.grad.W2.',1, []);
%             M2W3 = reshape(model_2.grad.W3.',1, []);
% 
%             M2b1 = model_2.grad.b1;
%             M2b2 = model_2.grad.b2;
%             M2b3 = model_2.grad.b3;
%             
%             parent2 = {M2W1; M2W2; M2W3; M2b1; M2b2; M2b3};
%             % Model 3
%             M3W1 = reshape(model_3.grad.W1.',1, []);
%             M3W2 = reshape(model_3.grad.W2.',1, []);
%             M3W3 = reshape(model_3.grad.W3.',1, []);
% 
%             M3b1 = model_3.grad.b1;
%             M3b2 = model_3.grad.b2;
%             M3b3 = model_3.grad.b3;
% 
%             parent3 = {M3W1; M3W2; M3W3; M3b1; M3b2; M3b3};
%             % do cross over and create 7 more children
% 
%             for i=1:7
% 
% 
%             end
% 
%         end
%         
%     end
    
 end