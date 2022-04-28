classdef NN
    % Neural Network object for training
    
    properties
        % MLP model
        mlp
        % errors and gradients
        grad
        % batch size
        m
        % optimizer
        optim
        %error
        error4
        %rmse
        rmse
        %cross entropy
        crossEnt
        
    end
    methods
        
        function obj = NN(size_hl1, size_hl2, opt, lr)
            obj.m = 0;
            
            % Initialize mlp model
            obj.mlp = MLP(size_hl1, size_hl2);
            
            % Initialize gradients
            obj.grad = MLP(size_hl1, size_hl2);
            obj.grad = obj.grad*0;
            
            % Initialize optimizer
            obj.optim = optimizer(opt, lr, obj.mlp);

            %init rmse
            obj.rmse = [];
            
        end
        function obj = newNN(size_hl1, size_hl2, opt, lr, w1, w2, w3, b1, b2, b3)

            % new mlp model using weights from genetic alg
            obj.mlp = GA_MLP(size_hl1, size_hl2, opt, lr, w1, w2, w3, b1, b2, b3);
            obj.grad = GA_MLP(size_hl1, size_hl2, opt, lr, w1, w2, w3, b1, b2, b3);
            obj.grad = obj.grad*0;
            obj.optim = optimizer(opt, lr, obj.mlp);

        end

       
        function obj = backpropagate(obj, a1, y)
            
            % Feed forward
            z2 = obj.mlp.W1*a1 + obj.mlp.b1;
            a2 = ReLU(z2, 'forward');
            z3 = obj.mlp.W2*a2 + obj.mlp.b2;
            a3 = ReLU(z3, 'forward');
            z4 = obj.mlp.W3*a3 + obj.mlp.b3;
            a4 = sigmoid(z4); % Output
            
            % Backpropagation minimizing the cross-entropy loss
            error4 = (a4-y);
            error3 = (obj.mlp.W3'*error4).*ReLU(z3, 'backward');
            error2 = (obj.mlp.W2'*error3).*ReLU(z2, 'backward');
            
            % Keep the sum of the gradients
            obj.grad.b3 = obj.grad.b3 + error4;
            obj.grad.b2 = obj.grad.b2 + error3;
            obj.grad.b1 = obj.grad.b1 + error2;
            
            obj.grad.W3 = obj.grad.W3 + error4*a3';
            obj.grad.W2 = obj.grad.W2 + error3*a2';
            obj.grad.W1 = obj.grad.W1 + error2*a1';
            
            %add error 
            obj.error4 = mean(error4 .^2);
            %obj.crossEnt = crossentropy(y, a4);
            obj.rmse = [obj.rmse, obj.error4];

            obj.m = obj.m + 1; % Count number of samples in batch
        end
        
        function obj = step(obj)
            
            obj.grad = obj.grad*(1/obj.m); % Mean of the gradient
            
            % Step the optimizer (Gradient descent)
            [obj, obj.optim] = obj.optim.step(obj);
            
            % Reset gradients counter
            obj.m = 0;
            
            % reset gradient
            %obj.grad = obj.grad*0;
        end
        
        function score = predict(obj, input)
            % Feed forward
            %disp("inside NN")
            %disp(size(obj.mlp.W1))
            %disp(size(input))
            %disp(size(obj.mlp.b1))
            z2 = obj.mlp.W1*input + obj.mlp.b1;
            a2 = ReLU(z2, 'forward');
            z3 = obj.mlp.W2*a2 + obj.mlp.b2;
            a3 = ReLU(z3, 'forward');
            z4 = obj.mlp.W3*a3 + obj.mlp.b3;
            score = sigmoid(z4); % Output vector of prediction scores
        end
     
    end
end