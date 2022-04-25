%%
%we dont need random is the genetic algorithm is used   
%rng(0); % seed for the random number generator
%%cd '/home/cewinharhar/GITHUB/becs4'/neural_network_mninst/
%cd 'C:\Users\kevin yar\OneDrive - ZHAW\KEVIN STUFF\ZHAW\_PYTHON_R\_GITHUB\becs4\neural_network_mninst'


%% training data
disp('Loading data...')

% load train data
train_data = load('mnist_train.csv');

%reduce data volume to increase difficulty
train_data = train_data(1:10000,:);

labels = train_data(:,1);
y = zeros(10,length(train_data)); % output labels for training
for i = 1:length(train_data)
    %one hot encoding
    y(labels(i)+1,i) = 1;
end

% preprocessing
images = train_data(:,2:785);
images = images/255;
images = images'; % Input vectors for training

%% test data
    
% load test data
test_data = load('mnist_test.csv');
%reduce test data
test_data = test_data(1:5000, :);

test_labels = test_data(:,1);
test_y = zeros(10 ,length(test_data));
for i = 1:length(test_data)
    test_y(test_labels(i)+1,i) = 1;
end

% preproessing
test_images = test_data(:,2:785);
% normalize data
test_images = test_images/255;
% transposing
test_images = test_images';

%%

%set NN size
size_hl1 = 128; % Number of neurons in the first hidden layer
size_hl2 = 64; % Number of neurons in the second hidden layer

%create container (dictionair equivalent) to store models
nnAdam = NN(size_hl1, size_hl2, 'Adam', lr);
nnAda = NN(size_hl1, size_hl2, 'Adagrad', lr);
nnSDG = NN(size_hl1, size_hl2, 'SDG', lr);

%%


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
    %legend('Methane','Location','best');
    hold on

%create accuracy and error arrays
acc = [];
err = [];
trained_models = [];

cou = 0;
plotCol = ["o", "s"];
%make for loop to iterate over all optimization methods
for nn = [nnAdam, nnAda]

     %for plot color
    cou = cou + 1;
    
    if strcmp(nn.optim.opt,'SDG')
        lr = 0.1;
    else
        lr = 0.001;
    end

    %specify parameters
    epochs = 10; % Number of training epochs
    batch_size = 100; % Mini-batch size

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
        
        disp(nn.optim.opt);
        fprintf('Epochs:');
        disp(e) % Track number of epochs
        
        
        % Evaluate model accuracy
        disp('Evaluating model')
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
        err = [err, nn.error4];

        fprintf('Accuracy: ');
        fprintf('%f',hits/n*100);
        disp(' %');
        disp(' ');
       
        
        [images,y] = shuffle(images,y); % Shuffle order of the images for next epoch
    end

    trained_models = [trained_models, nn];
    
    %plot the accuracy
    
    yyaxis left
    scatter([1:length(acc)], acc, 'filled',plotCol(cou), "DisplayName", nn.optim.opt)
    plot([1:length(acc)], acc, 'HandleVisibility','off')

    yyaxis right
    scatter([1:length(acc)], err, 'filled', plotCol(cou), 'HandleVisibility','off')
    plot([1:length(acc)], err, 'HandleVisibility','off')
    drawnow

    acc = [];
    err = [];
    hold on

end

legend show
hold off




