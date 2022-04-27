classdef genModel

    % new MLP with weights and bias from genAlg
    properties

        % num neurons in hidden layers
        size_hl1
        size_hl2
        
        % weights
        W1
        W2
        W3
        
        % biases
        b1
        b2
        b3
    end

    methods
         % Creates a new MLP with weights from genetic algorthmn selection
         function obj = genModel(size_hl1, size_hl2, w1, w2, w3, b1, b2, b3)
            
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

        function score = predict(obj, input)
            % Feed forward
            z2 = obj.W1*input + obj.b1;
            a2 = ReLU(z2, 'forward');
            z3 = obj.W2*a2 + obj.b2;
            a3 = ReLU(z3, 'forward');
            z4 = obj.W3*a3 + obj.b3;
            score = sigmoid(z4); % Output vector of prediction scores
        end
    end










end