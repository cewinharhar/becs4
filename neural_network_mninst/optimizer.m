classdef optimizer
    properties
        opt % optimizer name
        lr % learning rate
        
        % For the Adam optimizer:
        m % Moving avarage of the gradients
        v % Moving avarage of the squares of the gradients
        beta1 % Moving avarage parameter of m
        beta2 % Moving avarage parameter of v
        t % Number of steps taken
    end
    
    methods
        function obj = optimizer(opt, lr, mlp)
            if strcmp(opt, 'SGD')
                % Initialize settings for Stochastic Gradient Descent
                obj.opt = 'SGD';
                obj.lr = lr;
            
            elseif strcmp(opt, 'Adam')
                % Initialize settings for the Adam optimizer
                obj.opt = 'Adam';
                obj.lr = lr;
                obj.m = MLP(mlp.size_hl1, mlp.size_hl2)*0;
                obj.v = MLP(mlp.size_hl1, mlp.size_hl2)*0;
                obj.beta1 = 0.9;
                obj.beta2 = 0.999;
                obj.t = 0;
            end
            
        end
            
        function [nn, obj] = step(obj, nn)
            
            if strcmp(obj.opt, 'SGD')
                % Gradient descent step
                nn.mlp = nn.mlp - obj.lr*nn.grad;
            
            elseif strcmp(obj.opt, 'Adam')
                obj.t = obj.t+1; % Count number of steps
                
                % Moving avarages
                obj.m = obj.beta1*obj.m + (1-obj.beta1)*nn.grad;
                obj.v = obj.beta2*obj.v + (1-obj.beta2)*(nn.grad^2);
                
                % Step the model parameters
                nn.mlp = nn.mlp - obj.lr * ((obj.m*(1/(1-obj.beta1^obj.t))) / ((obj.v*(1/(1-obj.beta2^obj.t)))^0.5 + 10^-8)); 
                
            end
        end

    end
    
end