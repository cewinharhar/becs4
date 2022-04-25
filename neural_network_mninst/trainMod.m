%% trainMod
%function to train multiple models with different random values
function modelArray = trainMod(size_hl1, size_hl2, optimizer, lr, epochs, batch_size, images, y, nrOfModels)
    %create container (dictionair equivalent) to store models   
    modelArray = [];

    for modelIter = 1:nrOfModels
        
        %init model
        model = NN(size_hl1, size_hl2, optimizer, lr);
         %train the model
        for e = 1:epochs             
            samples = 1;            
            % Loop over gradient descent steps
            for j = 1:length(train_data)/batch_size                 
                for i = samples:samples+batch_size-1 % Loop over each minibatch        
                    % Calculate gradients with backpropagation
                    model = model.backpropagate(images(:,i), y(:,i));                    
                end            
                % Gradient descent (optimizer step)
                model = model.step();                  
                % Number of data points covered (+1)
                samples = samples + batch_size;            
            end
        end
        %append models to model array
        modelArray = [modelArray, model];

    end

end