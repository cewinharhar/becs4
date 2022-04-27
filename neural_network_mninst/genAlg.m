%% Genetic algorithm
% I wonder if I can just average the top 3 models to produce one child? or
% average them together with different weights? In the MLP file functions
% were defined to do that.
% to make the children, I can generate a random indexing into each weight
% or bias. I'll take the first half of those and index into a parent and
% give it to the child then take the latter half and index into the second
% parent and place it into the child. Then do the same for the rest!! I
% think this will work and then we don't need to flatten it (at 
% least I don't think I will) hi
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
        %
        generationCounter
        test_data
        test_images
        test_labels
        mutRate
        crossOverRate
        %sandbox
        evoSandBox
        % new population        
        model_1
        model_2
        model_3
        parent1
        parent2
        parent3
        newweight1
        new_population
    end

    methods

        function obj = load(nnMatrix, test_data, test_images, test_labels,  mutRate, crossOverRate)
            %function to optimize the weights of h1 and h2
            % the nnMatrix is a matrix with xdim = number of models and
            % ydim is the number of hyperparameters arrays: (W, b)x layers            
            
            %initialize class
            obj.nnMatrix = nnMatrix;
            obj.test_data = test_data;
            obj.test_images = test_images;
            obj.test_labels = test_labels;
            %both evolution paramters must be between 0 and 1
            % 0.05-0.1
            obj.mutRate = mutRate;
            % 0.2 - 0.3
            obj.crossOverRate = crossOverRate;


            %set fitness scores
            obj.fitness = zeros(length(nnMatrix), 1 );

            %set evolution sandbox
            %sample   = zeros(10, 6);
            %rowNames = {'parent1','parent2','parent3','child1','child2','child3','child4','child5','child6','child7'};
            %colNames = {'w1','w2','w3','b1', 'b2', 'b3'};          
            %obj.evoSandBox = array2table(sample,'RowNames',rowNames,'VariableNames',colNames);
            %use cell arrays for this (like lists)
            obj.evoSandBox = {};
            for rows = 1:10
                for cols = 1:6
                  obj.evoSandBox{rows,cols} = 1:10;
                end
            end      

            obj.generationCounter = 0;
            genAlg()

        end

         %recursive genetic algorithm function
        function finalNN = genAlg(obj)
            %exit statement
            while obj.generationCounter < 10
                obj.generationCounter = obj.generationCounter + 1;

                %------
                % Fitness evaluation
                %iterate over models

                modelCounter = 0;
                for model = nnMatrix            
    
                    modelCounter = modelCounter +1;
    
                    %calculate accuracy
                    hits = 0;
                    n = length(obj.test_data);
                    for i = 1:n                
                        out = model.predict(obj.test_images(:,i)); % model prediction vector
                        [~, num] = max(out); % Find highest prediction score                
                        if obj.test_labels(i) == (num-1)
                            hits = hits + 1; % Count the number of correct classifications
                        end       
                    end
                    %calculate accuracy
                    accuracy = hits/n;
                    obj.fitness(modelCounter) = accuracy;
                    
                end
                
                %-----
                %exit call
                %-----
                if max(obj.fitness) < 95
                    break
                end

                %-----
                % Rank the models by accuracy
                %-----
                [obj.sorted_fitness, obj.index] = sort(obj.fitness, 'descend');
                
                
                %-----
                % extract top 3 models and transfer information into
                % sandbox
                %-----                
                for parent = 1:3
                    initParent = obj.nnMatrix(obj.index(parent));
    
                %-----
                % Get the weights and biases of each model, flatten and store               
                %-----
                                    
                    MW1 = reshape(initParent.mlp.W1.',1, []);
                    MW2 = reshape(initParent.mlp.W2.',1, []);
                    MW3 = reshape(initParent.mlp.W3.',1, []);
        
                    Mb1 = initParent.mlp.b1;
                    Mb2 = initParent.mlp.b2;
                    Mb3 = initParent.mlp.b3;              
    
                    obj.evoSandBox(parent,:) = {MW1; MW2; MW3; Mb1; Mb2; Mb3};
                end
    
                %-----
                % Evolution process
                %-----               

                for hyperparameter = 1:width(obj.evoSandBox)
                    for child = 4:10
                        % Evolution or not?
                        %if obj.crossOverRate >= rand()
                           %if not go to next column
                        %   continue
                        %end                   
                    
                        %-----
                        % cross over
                        wheelOfFortune = obj.evoSandBox(1:3,hyperparameter);
                        obj.evoSandBox(child, hyperparameter) = wheelOfFortune(randi([1,3],1));
                                 
                        %-----
                        %mutation
                        chromosomLength = length(obj.evoSandBox{child, hyperparameter});
                        mutationSites = rand([1,chromosomLength], round(obj.mutationRate * chromosomLength));
    
                        for pointMutation = mutationSites
                            %differentiate between weights and bias mutation
                            if hyperparameter < 4 %only weights
                                mutant = rand([-0.01, 0.01], 1);
                            else                  %only biases
                                mutant = rand([-0.04, 0.04], 1);
                            end
                            obj.evoSandBox{child, hyperparameter}(pointMutation) = mutant;
                        end
                    end
                end

                %-----
                %transfer hyperparameters back to models

                
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
                    newbias3 = zeros(1,length(obj.parent1{4}));
                    newbias3(1:elements1) = obj.parent1{6}(indexb3(1:elements1));
                    newbias3(elements1:elements2) = obj.parent2{6}(indexb3(elements1:elements2));
                    newbias3(elements2:elements3) = obj.parent3{6}(indexb3(elements2:elements3));
    
                    % now need to reshape and create new NN object and add to a
                    % new matrix. First need to reshape into same dimensions
                    % required for weights
                    newweight1 = reshape(newweight1, [128, 784]);
                    newweight2 = reshape(newweight2, [64, 128]);
                    newweight3 = reshape(newweight3, [10, 64]);
                    % biases don't need to be reshaped.
   

            end

             % get the new MLP and add it to the new population
            % newNN(size_hl1, size_hl2, opt, lr, w1, w2, w3, b1, b2, b3)
            child = NN(128, 64, "Adam", 0.001);
            child = child.newNN(128, 64, "Adam",  0.001, newweight1, newweight2, newweight3, newbias1, newbias2, newbias3);
            obj.new_population = [obj.new_population, child];
%             end
        end
    end

%%
rowNames = {'parent1','parent2','parent3','child1','child2','child3','child4','child5','child6','child7'};
con = containers.Map();
for o = rowNames
    con(o) = rand();
end

%%
c = {};
for i = 1:10
    for o = 1:6
      c{i,o} = [1:10];
    end
end




