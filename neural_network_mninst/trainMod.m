%% trainMod

function mods = trainMod(size_hl1, size_hl2, 'Adam', lr)

       
    %create container (dictionair equivalent) to store models
    nnAdam = NN(size_hl1, size_hl2, 'Adam', lr);


end