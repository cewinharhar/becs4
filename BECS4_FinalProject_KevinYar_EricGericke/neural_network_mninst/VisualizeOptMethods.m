function trained_models = VisualizeOptMethods(epochs, batch_size, size_hl1, size_hl2, lr, train_data, images, y, test_data, test_images, test_labels)

    %create container (dictionair equivalent) to store models
    nnAdam = NN(size_hl1, size_hl2, 'Adam', lr);
    nnAda = NN(size_hl1, size_hl2, 'Adagrad', lr);
    nnAdaD = NN(size_hl1, size_hl2, 'Adadelta', lr);
    %nnSDG = NN(size_hl1, size_hl2, 'SDG', lr);
    
    
    %initialize figure
    figure('Name','Comparison optimization methods')
        grid on;
    
        % --- labeling
        title('Comparison optimization methods');
        xlabel('Epochs');
    
        yyaxis left
        ylabel('Accuracy [%]');
    
        yyaxis right 
        ylabel("Error [RMSE]")

        hold on
    
    %create accuracy and error arrays
    acc = [];
    err = [];
    trained_models = [];
    
    cou = 0;
    plotCol = ["-o", "-s", "-h"];
    %make for loop to iterate over all optimization methods

    %progress bar
    f = waitbar(0, 'Starting');
    
    for nn = [nnAdam, nnAda, nnAdaD]
        %for plot color
        cou = cou + 1;
        %progress bar
        waitbar(cou/4, f, sprintf('Training Model %d: %%',  floor(cou/4*100)));
        %waitbar(cou, f, sprintf('Evaluating optimizer: %d: %d %%', nn.optim.opt, cou));
        
        %set learning rate
        lr = lr;
    
        %specify parameters
        epochs = epochs; % Number of training epochs
        batch_size = batch_size; % Mini-batch size
    
        %iterate over epochs
        for e = 1:epochs 
            samples = 1;
            
            % Loop over gradient descent steps
            for j = 1:length(train_data)/batch_size 
                
                for i = samples:samples+batch_size-1 % Loop over each minibatch
        
                    % Calculate gradients with backpropagation
                    nn = nn.backpropagate(images(:,i), y(:,i));
                    
                end
            
                % Gradient descent (optimizer step)
                nn = nn.step();
                  
                % Number of data points covered (+1)
                samples = samples + batch_size;
            
            end            
            
            % Evaluate model accuracy
            hits = 0;
            n = length(test_data);
            for i = 1:n
        
                out = nn.predict(test_images(:,i)); % model prediction vector
                [~, num] = max(out); % Find highest prediction score
        
                if test_labels(i) == (num-1)
                    hits = hits + 1; % Count the number of correct classifications
                end
        
            end
            
            %calculate accuracy and error
            acc = [acc, hits/n*100];
            err = [err, mean(nn.rmse)];
    
            % Shuffle order of the images for next epoch
            [images,y] = shuffle(images,y); 
        end
    
        %append models
        trained_models = [trained_models, nn];
        
        %plot the accuracy        
        yyaxis left
        plot(1:length(acc), acc ,plotCol(cou), "DisplayName", nn.optim.opt)
    
        yyaxis right
        plot(1:length(acc), err, plotCol(cou), 'HandleVisibility','off')

        drawnow
        %reset accuracy and error
        acc = [];
        err = [];
        hold on
    
    end
    
    legend show
    hold off
    close(f)
end



