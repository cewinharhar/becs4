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

        function obj = genAlg(nnMatrix, test_data, test_images, test_labels,  mutRate, crossOverRate)
            %function to optimize the weights of h1 and h2
            % the nnMatrix is a matrix with xdim = number of models and
            % ydim is the number of hyperparameters arrays: (W, b)x layers            
            
            %initialize class
            obj.nnMatrix    = nnMatrix;
            obj.test_data   = test_data;
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

            obj = genAlgRecursive(obj);

        end

         %recursive genetic algorithm function
        function obj = genAlgRecursive(obj)
            %exit statement
            while obj.generationCounter < 10
                obj.generationCounter = obj.generationCounter + 1;
                disp("generation")
                disp(obj.generationCounter)
                disp("-----------------")
                %------
                % Fitness evaluation
                %iterate over models
                modelCounter = 0;
                for model = obj.nnMatrix            
                    modelCounter = modelCounter +1;
    
                    %calculate accuracy
                    hits = 0;
                    n = length(obj.test_data);
                    disp("length n")
                    disp(n)
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
                % Rank the models by accuracy
                %-----
                [obj.sorted_fitness, obj.index] = sort(obj.fitness, 'descend');

                %-----
                %exit call
                %-----
                if max(obj.fitness) > 0.93
                    break
                end                
                
                %-----
                % extract top 3 models and transfer information into
                % sandbox
                %-----                
                for parent = 1:3
                    initParent = obj.nnMatrix(obj.index(parent));
    
                %-----
                % Get the weights and biases of each model, flatten and store               
                %-----
                                    
                    MW1 = initParent.mlp.W1(:)';
                    MW2 = initParent.mlp.W2(:)';
                    MW3 = initParent.mlp.W3(:)';
        
                    Mb1 = initParent.mlp.b1';
                    Mb2 = initParent.mlp.b2';
                    Mb3 = initParent.mlp.b3';              
    
                    obj.evoSandBox(parent,:) = {MW1, MW2, MW3, Mb1, Mb2, Mb3};
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
                        mutationSites   = randi([1,chromosomLength], round(obj.mutRate * chromosomLength));
    
                        for pointMutation = mutationSites
                            %differentiate between weights and bias mutation
                            if hyperparameter < 4 %only weights
                                mutant = randi([-200, 200], 1) / 10000;
                            else                  %only biases
                                mutant = randi([-400, 400], 1) / 10000;
                            end
                            obj.evoSandBox{child, hyperparameter}(pointMutation) = mutant;
                        end
                    end
                end

                %-----
                %transfer hyperparameters back to models
                for updateModel = 1:length(obj.nnMatrix)
                
                    obj.nnMatrix(updateModel).mlp.W1 = reshape(obj.evoSandBox(updateModel,1), [128, 784]);
                    obj.nnMatrix(updateModel).mlp.W2 = reshape(obj.evoSandBox(updateModel,2), [128, 784]);
                    obj.nnMatrix(updateModel).mlp.W3 = reshape(obj.evoSandBox(updateModel,3), [128, 784]);
                    obj.nnMatrix(updateModel).mlp.b1 = obj.evoSandBox(updateModel,4)';
                    obj.nnMatrix(updateModel).mlp.b2 = obj.evoSandBox(updateModel,5)';
                    obj.nnMatrix(updateModel).mlp.b3 = obj.evoSandBox(updateModel,6)';
                end

                genAlg()

            end   

        end
    end
end


%rowNames = {'parent1','parent2','parent3','child1','child2','child3','child4','child5','child6','child7'};
%con = containers.Map();
%for o = rowNames
%    con(o) = rand();
%end


%c = {};
%for i = 1:10
%    for o = 1:6
%      c{i,o} = [1:10];
%    end
%end




