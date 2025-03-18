% LSTM forward pass over an entire sequence 
function [h_final, c_final, caches] = lstmForward(X, parameters) % X is [inputDim x T] where T is the number of time steps 

[~, T] = size(X); 
hiddenDim = size(parameters.Wf, 1);
h_prev = zeros(hiddenDim, 1);
c_prev = zeros(hiddenDim, 1);
caches = cell(1, T);

for t = 1:T
    x_t = X(:, t);
    [h_next, c_next, cache] = lstmCellForward(x_t, h_prev, c_prev, parameters);
    caches{t} = cache;
    h_prev = h_next;
    c_prev = c_next;
end
h_final = h_prev;
c_final = c_prev;

end