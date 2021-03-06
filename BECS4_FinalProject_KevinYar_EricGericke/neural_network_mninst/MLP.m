classdef MLP
    % Two layer Multi-Layer Perceptron object
    
    properties
        % Number of hidden neurons
        size_hl1
        size_hl2
        % Weighting matrices
        W1
        W2
        W3
        % bias vectors
        b1
        b2
        b3
    end
    
    methods
        function obj = MLP(size_hl1, size_hl2)
            
            obj.size_hl1 = size_hl1;
            obj.size_hl2 = size_hl2;
            
            % (Initializations are the same as for example PyTorch's default settings)
            
            % Initialize weights
            obj.W1 = randn(size_hl1, 784) * sqrt(1/784);
            obj.W2 = randn(size_hl2, size_hl1) * sqrt(1/size_hl1);
            obj.W3 = randn(10, size_hl2) * sqrt(1/size_hl2);
            
            % Initialize biases
            obj.b1 = randn(size_hl1, 1) * sqrt(1/784);
            obj.b2 = randn(size_hl2, 1) * sqrt(1/size_hl1);
            obj.b3 = randn(10, 1) * sqrt(1/size_hl2);
        end
        
        % Creates a new MLP with weights from genetic algorthmn selection
        function obj = GA_MLP(size_hl1, size_hl2, w1, w2, w3, b1, b2, b3)
            
            % Size of hiddenlayer 1&2
            obj.size_hl1 = size_hl1;
            obj.size_hl2 = size_hl2;
            % Initialize weights
            obj.W1 = w1;
            obj.W2 = w2;
            obj.W3 = w3;
            % Initialize bias
            obj.b1 = b1;
            obj.b2 = b2;
            obj.b3 = b3;
        end


        % Enables addition of MLP models
        function obj = plus(obj1, obj2)
            
            obj = obj1;
            if isnumeric(obj2) % if a scalar is added
                obj.W3 = obj1.W3 + obj2;
                obj.W2 = obj1.W2 + obj2;
                obj.W1 = obj1.W1 + obj2;

                obj.b3 = obj1.b3 + obj2;
                obj.b2 = obj1.b2 + obj2;
                obj.b1 = obj1.b1 + obj2;
            
            else
            
                obj.W3 = obj1.W3 + obj2.W3;
                obj.W2 = obj1.W2 + obj2.W2;
                obj.W1 = obj1.W1 + obj2.W1;

                obj.b3 = obj1.b3 + obj2.b3;
                obj.b2 = obj1.b2 + obj2.b2;
                obj.b1 = obj1.b1 + obj2.b1;
            end
        end
        
        % Enables substraction of MLP models
        function obj = minus(obj1, obj2)
            obj = obj1;
            
            obj.W3 = obj1.W3 - obj2.W3;
            obj.W2 = obj1.W2 - obj2.W2;
            obj.W1 = obj1.W1 - obj2.W1;
            
            obj.b3 = obj1.b3 - obj2.b3;
            obj.b2 = obj1.b2 - obj2.b2;
            obj.b1 = obj1.b1 - obj2.b1;
        end
        
        % Enables multiplication of MLP models
        function obj = mtimes(obj1, obj2)
            
            if isnumeric(obj1) % if multiplied with scalar
                obj = obj2;
                obj.W3 = obj1 .* obj2.W3;
                obj.W2 = obj1 .* obj2.W2;
                obj.W1 = obj1 .* obj2.W1;

                obj.b3 = obj1 .* obj2.b3;
                obj.b2 = obj1 .* obj2.b2;
                obj.b1 = obj1 .* obj2.b1;
            
            elseif isnumeric(obj2) % if multiplied with scalar
                obj = obj1; 
                obj.W3 = obj2 .* obj1.W3;
                obj.W2 = obj2 .* obj1.W2;
                obj.W1 = obj2 .* obj1.W1;

                obj.b3 = obj2 .* obj1.b3;
                obj.b2 = obj2 .* obj1.b2;
                obj.b1 = obj2 .* obj1.b1;
            else
                obj = obj1;
                obj.W3 = obj1.W3 .* obj2.W3;
                obj.W2 = obj1.W2 .* obj2.W2;
                obj.W1 = obj1.W1 .* obj2.W1;

                obj.b3 = obj1.b3 .* obj2.b3;
                obj.b2 = obj1.b2 .* obj2.b2;
                obj.b1 = obj1.b1 .* obj2.b1;
            end
        end
        
        % Enables division of MLP models
        function obj = mrdivide(obj1, obj2)
            obj = obj2;
            obj.W3 = obj1.W3 ./ obj2.W3;
            obj.W2 = obj1.W2 ./ obj2.W2;
            obj.W1 = obj1.W1 ./ obj2.W1;

            obj.b3 = obj1.b3 ./ obj2.b3;
            obj.b2 = obj1.b2 ./ obj2.b2;
            obj.b1 = obj1.b1 ./ obj2.b1;
        end
        
        % Enables to the power of x of MLP models
        function obj = mpower(obj1, x)
            obj = obj1;
            obj.W3 = obj1.W3 .^ x;
            obj.W2 = obj1.W2 .^ x;
            obj.W1 = obj1.W1 .^ x;

            obj.b3 = obj1.b3 .^ x;
            obj.b2 = obj1.b2 .^ x;
            obj.b1 = obj1.b1 .^ x;
        end
    end
end