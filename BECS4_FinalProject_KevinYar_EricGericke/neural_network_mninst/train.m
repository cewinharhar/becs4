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
%%
    
% preprocessing
images = train_data(:,2:785);
images = images/255;
images = images'; % Input vectors for training

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

size_hl1 = 128; % Number of neurons in the first hidden layer
size_hl2 = 64; % Number of neurons in the second hidden layer


% Construct model with specifed optimizer for training
lr = 0.001; % learning rate
model = NN(size_hl1, size_hl2, 'Adagrad', lr);

% Alternatively with SGD optimizer
% lr = 0.1;
% model = NN(hn1, hn2, 'SGD', lr);

epochs = 2; % Number of training epochs

batch_size = 100; % Mini-batch size

disp('Starting epoch 1');
disp(' ');

% Epoch loop
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
    
    fprintf('Epochs:');
    disp(e) % Track number of epochs
    
    
    % Evaluate model accuracy
    disp('Evaluating model')
    hits = 0;
    n = length(test_data);
    for i = 1:n

        out = model.predict(test_images(:,i)); % model prediction vector
        [~, num] = max(out); % Find highest prediction score

        if test_labels(i) == (num-1)
            hits = hits + 1; % Count the number of correct classifications
        end

    end
    fprintf('Accuracy: ');
    fprintf('%f',hits/n*100);
    disp(' %');
    disp(' ');
    
    
    [images,y] = shuffle(images,y); % Shuffle order of the images for next epoch
end

disp('Done');



