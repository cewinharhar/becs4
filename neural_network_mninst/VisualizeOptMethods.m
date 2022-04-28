%%
%we dont need random is the genetic algorithm is used   
%rng(0); % seed for the random number generator
%%cd '/home/cewinharhar/GITHUB/becs4'/neural_network_mninst/
%cd 'C:\Users\kevin yar\OneDrive - ZHAW\KEVIN STUFF\ZHAW\_PYTHON_R\_GITHUB\becs4\neural_network_mninst'
function trained_models = VisualizeOptMethods(epochs, batch_size, size_hl1, size_hl2, lr, train_data, images, y, test_data, test_images, test_labels)

    %create container (dictionair equivalent) to store models
    nnAdam = NN(size_hl1, size_hl2, 'Adam', lr);
    nnAda = NN(size_hl1, size_hl2, 'Adagrad', lr);
    nnAdaD = NN(size_hl1, size_hl2, 'Adadelta', lr);
    %nnSDG = NN(size_hl1, size_hl2, 'SDG', lr);
    
    
    %initialize figure
    figure('Name','Comparison optimization methods')
        grid on;
    
        % --- Titelei
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
            
            %disp(nn.optim.opt);
            %fprintf('Epochs:');
            %disp(e) % Track number of epochs
            
            
            % Evaluate model accuracy
            %disp('Evaluating model')
            hits = 0;
            n = length(test_data);
            for i = 1:n
        
                out = nn.predict(test_images(:,i)); % model prediction vector
                [~, num] = max(out); % Find highest prediction score
        
                if test_labels(i) == (num-1)
                    hits = hits + 1; % Count the number of correct classifications
                end
        
            end
            
            acc = [acc, hits/n*100];
            err = [err, mean(nn.rmse)];
    
            %fprintf('Accuracy: ');
            %fprintf('%f',hits/n*100)
            
            [images,y] = shuffle(images,y); % Shuffle order of the images for next epoch
        end
    
        trained_models = [trained_models, nn];
        
        %plot the accuracy
        
        yyaxis left
        plot(1:length(acc), acc ,plotCol(cou), "DisplayName", nn.optim.opt)
        %plot([1:length(acc)], acc, 'HandleVisibility','off')
    
        yyaxis right
        plot(1:length(acc), err, plotCol(cou), 'HandleVisibility','off')
        %plot([1:length(acc)], err, 'HandleVisibility','off')
        drawnow
    
        acc = [];
        err = [];
        hold on
    
    end
    
    legend show
    hold off
    close(f)
end



