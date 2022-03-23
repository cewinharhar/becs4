function out = ReLU(x, direction)
    if strcmp(direction, 'forward')
        out = max(0,x);
    else
        out = double(x>0);
    end
end