%% Genetic algorithm
% I wonder if I can just average the top 3 models to produce one child? or
% average them together with different weights? In the MLP file functions
% were defined to do that.
% to make the children, I can generate a random indexing into each weight
% or bias. I'll take the first half of those and index into a parent and
% give it to the child then take the latter half and index into the second
% parent and place it into the child. Then do the same for the rest!! I
% think this will work and then we don't need to flatten it (at 
% least I don't think I will)
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
        % new population
        new_population

        model_1
        model_2
        model_3
        parent1
        parent2
        parent3
        newbias3
        genCount
        
    end

    methods

        function obj = genAlg(nnMatrix, test_data, test_images, test_labels,  mutRate, generations)
            obj.nnMatrix = nnMatrix;
            obj.genCount = 0;
            for iterations = 1:generations
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
                for model = obj.nnMatrix            
    
                    counter = counter +1;
    
                    %calculate accuracy
                    hits = 0;
                    n = length(test_data);
                    for i = 1:3 %make this n later
                
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
                
                
                % choose top 3 models
                obj.model_1 = obj.nnMatrix(obj.index(1));
                obj.model_2 = obj.nnMatrix(obj.index(2));
                obj.model_3 = obj.nnMatrix(obj.index(3));
                obj.new_population = [];
                for i=1:3
                    if (iterations == 1)
                        w1 = obj.nnMatrix(obj.index(i)).mlp.W1;
                        w2 = obj.nnMatrix(obj.index(i)).mlp.W2;
                        w3 = obj.nnMatrix(obj.index(i)).mlp.W3;
                        b1 = obj.nnMatrix(obj.index(i)).mlp.b1;
                        b2 = obj.nnMatrix(obj.index(i)).mlp.b2;
                        b3 = obj.nnMatrix(obj.index(i)).mlp.b3;
        
                        NN_to_genMod = genModel(128, 64, w1, w2, w3, b1, b2, b3);
                        obj.new_population = [obj.new_population, NN_to_genMod];
                    else
                        w1 = obj.nnMatrix(obj.index(i)).W1;
                        w2 = obj.nnMatrix(obj.index(i)).W2;
                        w3 = obj.nnMatrix(obj.index(i)).W3;
                        b1 = obj.nnMatrix(obj.index(i)).b1;
                        b2 = obj.nnMatrix(obj.index(i)).b2;
                        b3 = obj.nnMatrix(obj.index(i)).b3;
        
                        NN_to_genMod = genModel(128, 64, w1, w2, w3, b1, b2, b3);
                        obj.new_population = [obj.new_population, NN_to_genMod];
                    end

    
                end
    
                
    %             obj.new_population = [obj.new_population, obj.model_1];
    %             obj.new_population = [obj.new_population, obj.model_2];
    %             obj.new_population = [obj.new_population, obj.model_3];
    
    
                % Get the weights and biases of each model, flatten and store
                % for cross over section flatten and extr
                %reshape(model.',1, [])
                if (iterations == 1)
                    % Model 1
                    M1W1 = reshape(obj.model_1.mlp.W1.',1, []);
                    M1W2 = reshape(obj.model_1.mlp.W2.',1, []);
                    M1W3 = reshape(obj.model_1.mlp.W3.',1, []);
        
                    M1b1 = obj.model_1.mlp.b1;
                    M1b2 = obj.model_1.mlp.b2;
                    M1b3 = obj.model_1.mlp.b3;
        
                    obj.parent1 = {M1W1; M1W2; M1W3; M1b1; M1b2; M1b3};
                    
                    % Model 2
                    M2W1 = reshape(obj.model_2.mlp.W1.',1, []);
                    M2W2 = reshape(obj.model_2.mlp.W2.',1, []);
                    M2W3 = reshape(obj.model_2.mlp.W3.',1, []);
        
                    M2b1 = obj.model_2.mlp.b1;
                    M2b2 = obj.model_2.mlp.b2;
                    M2b3 = obj.model_2.mlp.b3;
                    
                    obj.parent2 = {M2W1; M2W2; M2W3; M2b1; M2b2; M2b3};
        
                    % Model 3
                    M3W1 = reshape(obj.model_3.mlp.W1.',1, []);
                    M3W2 = reshape(obj.model_3.mlp.W2.',1, []);
                    M3W3 = reshape(obj.model_3.mlp.W3.',1, []);
        
                    M3b1 = obj.model_3.mlp.b1;
                    M3b2 = obj.model_3.mlp.b2;
                    M3b3 = obj.model_3.mlp.b3;
        
                    obj.parent3 = {M3W1; M3W2; M3W3; M3b1; M3b2; M3b3};
                else
                    % Model 1
                    M1W1 = reshape(obj.model_1.W1.',1, []);
                    M1W2 = reshape(obj.model_1.W2.',1, []);
                    M1W3 = reshape(obj.model_1.W3.',1, []);
        
                    M1b1 = obj.model_1.b1;
                    M1b2 = obj.model_1.b2;
                    M1b3 = obj.model_1.b3;
        
                    obj.parent1 = {M1W1; M1W2; M1W3; M1b1; M1b2; M1b3};
                    
                    % Model 2
                    M2W1 = reshape(obj.model_2.W1.',1, []);
                    M2W2 = reshape(obj.model_2.W2.',1, []);
                    M2W3 = reshape(obj.model_2.W3.',1, []);
        
                    M2b1 = obj.model_2.b1;
                    M2b2 = obj.model_2.b2;
                    M2b3 = obj.model_2.b3;
                    
                    obj.parent2 = {M2W1; M2W2; M2W3; M2b1; M2b2; M2b3};
        
                    % Model 3
                    M3W1 = reshape(obj.model_3.W1.',1, []);
                    M3W2 = reshape(obj.model_3.W2.',1, []);
                    M3W3 = reshape(obj.model_3.W3.',1, []);
        
                    M3b1 = obj.model_3.b1;
                    M3b2 = obj.model_3.b2;
                    M3b3 = obj.model_3.b3;
        
                    obj.parent3 = {M3W1; M3W2; M3W3; M3b1; M3b2; M3b3};
                end

                
                % do cross over and create 7 more children
                
    
                for i=1:7
                    
                    indexW1 = randperm(length(obj.parent1{1})); % create random index
                    elements1 = round(length(indexW1)*0.33); %index using first 33%
                    elements2 = round(length(indexW1)*0.667); %index using second 33%
                    elements3 = length(indexW1); % last 33%
                    newweight1 = zeros(1,length(obj.parent1{1})); % create a new matrix
                    newweight1(1:elements1) = obj.parent1{1}(indexW1(1:elements1)); % radomly selecting weights from parent1
                    newweight1(elements1:elements2) = obj.parent2{1}(indexW1(elements1:elements2)); % from parent 2
                    newweight1(elements2:elements3) = obj.parent3{1}(indexW1(elements2:elements3)); % from parent 3
                    
                    
    
                    indexW2 = randperm(length(obj.parent1{2}));
                    elements1 = round(length(indexW2)*0.33);
                    elements2 = round(length(indexW2)*0.667);
                    elements3 = length(indexW2);
                    newweight2 = zeros(1,length(obj.parent1{2}));
                    newweight2(1:elements1) = obj.parent1{2}(indexW2(1:elements1));
                    newweight2(elements1:elements2) = obj.parent2{2}(indexW2(elements1:elements2));
                    newweight2(elements2:elements3) = obj.parent3{2}(indexW2(elements2:elements3));
    
                    indexW3 = randperm(length(obj.parent1{3}));
                    elements1 = round(length(indexW3)*0.33);
                    elements2 = round(length(indexW3)*0.667);
                    elements3 = length(indexW3);
                    newweight3 = zeros(1,length(obj.parent1{3}));
                    newweight3(1:elements1) = obj.parent1{3}(indexW3(1:elements1));
                    newweight3(elements1:elements2) = obj.parent2{3}(indexW3(elements1:elements2));
                    newweight3(elements2:elements3) = obj.parent3{3}(indexW3(elements2:elements3));
                    
    
                    indexb1 = randperm(length(obj.parent1{4}));
                    elements1 = round(length(indexb1)*0.33);
                    elements2 = round(length(indexb1)*0.667);
                    elements3 = length(indexb1);
                    newbias1 = zeros(1,length(obj.parent1{4}));
                    newbias1(1:elements1) = obj.parent1{4}(indexb1(1:elements1));
                    newbias1(elements1:elements2) = obj.parent2{4}(indexb1(elements1:elements2));
                    newbias1(elements2:elements3) = obj.parent3{4}(indexb1(elements2:elements3));
    
                    indexb2 = randperm(length(obj.parent1{5}));
                    elements1 = round(length(indexb2)*0.33);
                    elements2 = round(length(indexb2)*0.667);
                    elements3 = length(indexb2);
                    newbias2 = zeros(1,length(obj.parent1{5}));
                    newbias2(1:elements1) = obj.parent1{5}(indexb2(1:elements1));
                    newbias2(elements1:elements2) = obj.parent2{5}(indexb2(elements1:elements2));
                    newbias2(elements2:elements3) = obj.parent3{5}(indexb2(elements2:elements3));
    
                    indexb3 = randperm(length(obj.parent1{6}));
                    elements1 = round(length(indexb3)*0.33);
                    elements2 = round(length(indexb3)*0.667);
                    elements3 = length(indexb3);
                    newbias3 = zeros(1,length(obj.parent1{6}));
                    newbias3(1:elements1) = obj.parent1{6}(indexb3(1:elements1));
                    newbias3(elements1:elements2) = obj.parent2{6}(indexb3(elements1:elements2));
                    newbias3(elements2:elements3) = obj.parent3{6}(indexb3(elements2:elements3));
                    obj.newbias3 = newbias3;
                    disp(obj.newbias3)
    
                    % now need to reshape and create new NN object and add to a
                    % new matrix. First need to reshape into same dimensions
                    % required for weights
%                     newweight1 = reshape(newweight1, [128, 784]);
%                     newweight2 = reshape(newweight2, [64, 128]);
%                     newweight3 = reshape(newweight3, [10, 64]);
                      newweight1 = reshape(newweight1, [784, 128])';
                      newweight2 = reshape(newweight2, [128, 64])';
                      newweight3 = reshape(newweight3, [64, 10])';


                    % biases don't need to be reshaped.
                    
    
                    % get the new MLP and add it to the new population
                   
                    child = genModel(128, 64, (newweight1), (newweight2), (newweight3), (newbias1), (newbias2), (newbias3));
                    obj.new_population = [obj.new_population, child];
                end
                obj.nnMatrix = obj.new_population;
                obj.genCount = obj.genCount +1;
                
    
    
    
                % mutate all of them depending on the mutation rate
                % generate random number between -1 and 1 then divide by 100 or
                % 1000
                % repeat generation process unitl reaches 95% or max
                % generations of 100
            end

        end

        function obj = crossoverPopulation()
        end
    end
    
 end