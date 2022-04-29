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
        %counts gens
        generationCounter
        %saves the test, train data in the class
        test_data
        test_images
        test_labels
        %saves the parameters in the class
        mutRate        
        generations
        no_models
        mutations
        %sandbox for evolution
        evoSandBox       
    end

    methods

        function obj = genAlg(nnMatrix, test_data, test_images, test_labels,  mutRate, generations)
            %function to initialize the genetic algorithm 
            obj.generations = generations;
            obj.mutRate = mutRate;
            %initialize class
            obj.nnMatrix    = nnMatrix;
            obj.test_data   = test_data;
            obj.test_images = test_images;
            obj.test_labels = test_labels;
            %evolution paramters must be between 0 and 1
            obj.mutRate = mutRate;
            %nr of models
            obj.no_models = length(nnMatrix);
            %initialize fitness scores
            obj.fitness = zeros(length(nnMatrix), 1 );

            %initialize evolution sandbox in a cell array format
            % This is where we store our models with their associated
            % weights and bias's
            obj.evoSandBox = {};
            for rows = 1:obj.no_models
                for cols = 1:6
                  obj.evoSandBox{rows,cols} = 1:obj.no_models;
                end
            end      

            %counts the generations
            obj.generationCounter = 0;
            
            % calling the main function in this class where the actual
            % algorthm lies.
            obj = genAlgRecursive(obj);

        end

         %recursive genetic algorithm function
        function obj = genAlgRecursive(obj)
            
            % setting up a basic visualisation to display the accuracy of
            % the most fit model

            %initialize figure
            figure('Name','comparing generations')
                grid on;
            
                % --- title
                title('best fitness for each generation');
                xlabel('Generations');
                ylabel('fitness');
                hold on

            % Create array for the fitness of each generation and each
            % generation for the x-axis
            fitArr = [];
            genArr = [];

            % this is for fun to see the total amount of mutations that
            % occured
            obj.mutations = 0;

            %exit criteria
            while obj.generationCounter < obj.generations
                obj.generationCounter = obj.generationCounter + 1;

                % appending the generations to the array for visualisation
                genArr = [genArr, obj.generationCounter];

                

                disp("generation")
                disp(obj.generationCounter)
                disp("-----------------")
                
                %------
                % Fitness evaluation
                %iterate over models
                modelCounter = 0;
                for model = obj.nnMatrix    
                    modelCounter = modelCounter +1;
    
                    %calculate accuracy (fitness) for each model
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
                % Rank the models by accuracy
                %-----
                [obj.sorted_fitness, obj.index] = sort(obj.fitness, 'descend');
                % appending the most accurate of each model for
                % visualisation
                fitArr = [fitArr, obj.sorted_fitness(1)];
                disp("mutations")
                disp(obj.mutations)
                disp("------------")
                disp("fitness")                
                disp(obj.sorted_fitness(1:5))
                
                %plot the accuracy
                plot(genArr, fitArr, 'HandleVisibility','off')

            drawnow
            hold on
              
                
                %-----
                % extract top 2 models and transfer information into
                % sandbox
                %-----    

                for parent = 1:2
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
                    
                    for child = 3:obj.no_models

                        %set the hyperparameters for each child as the ones
                        %from the best parent
                        obj.evoSandBox(child, hyperparameter) = obj.evoSandBox(1,hyperparameter);
                        
                        %-----
                        % cross over

                        %store the weights and biases of both parents                   
                        wheelOfFortune = obj.evoSandBox(1:2,hyperparameter);

                        % Create indexes into the hyperparameter of
                        % interest
                        indexHyp = randperm(length(obj.evoSandBox{1, hyperparameter}));
                        index1 = round(length(indexHyp)*0.33);
                        index2 = round(length(indexHyp)*0.66);
                        index3 = length(indexHyp);
                        indices = [index1 index2 index3];
                        
                        %swap the hyperparameteer parts
                        obj.evoSandBox{child, hyperparameter}(1:indices(1))          = wheelOfFortune{randi([1,2],1)}(1:indices(1));
                        obj.evoSandBox{child, hyperparameter}(indices(1):indices(2)) = wheelOfFortune{randi([1,2],1)}(indices(1):indices(2));
                        obj.evoSandBox{child, hyperparameter}(indices(2):indices(3)) = wheelOfFortune{randi([1,2],1)}(indices(2):indices(3));                        

                        %-----
                        %mutation

                        %get a decision condition weather to mutate or not
                        if obj.mutRate >= rand()
                            obj.mutations = obj.mutations +1;
                            %define the length of the chromosomes
                            chromosomLength = length(obj.evoSandBox{child, hyperparameter});
                            %specify where the mutations occure
                            mutationSites   = randi([1,chromosomLength], round(0.01 * chromosomLength));
        
                            for pointMutation = mutationSites
                                %differentiate between weights and bias mutation
                                if hyperparameter < 4 %only weights
                                    mutant = randi([-100, 100], 1) / 10000;
                                else                  %only biases
                                    mutant = randi([-150, 150], 1) / 10000;
                                end
                                obj.evoSandBox{child, hyperparameter}(pointMutation) = mutant;
                            end
                        end
                    end
                end

                %-----
                %transfer hyperparameters back to models
                for updateModel = 1:length(obj.nnMatrix)
                
                    obj.nnMatrix(updateModel).mlp.W1 = reshape(obj.evoSandBox{updateModel,1}, [128, 784]);
                    obj.nnMatrix(updateModel).mlp.W2 = reshape(obj.evoSandBox{updateModel,2}, [64, 128]);
                    obj.nnMatrix(updateModel).mlp.W3 = reshape(obj.evoSandBox{updateModel,3}, [10, 64]);
                    obj.nnMatrix(updateModel).mlp.b1 = obj.evoSandBox{updateModel,4}';
                    obj.nnMatrix(updateModel).mlp.b2 = obj.evoSandBox{updateModel,5}';
                    obj.nnMatrix(updateModel).mlp.b3 = obj.evoSandBox{updateModel,6}';
                end
            end   
        end
    end
end




