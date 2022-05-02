function out = ReLU(x, direction)
        %Relu function
    if strcmp(direction, 'forward')
        out = max(0,x);
    else
        %derivative of ReLU
        out = double(x>0);
    end
end