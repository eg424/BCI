% LSTM cell forward computation for a single time step 
function [h_next, c_next, cache] = lstmCellForward(x, h_prev, c_prev, parameters) % Concatenate previous hidden state and current input 
z = [h_prev; x]; % size: [(hiddenDim+inputDim) x 1]

% Forget gate
f = sigmoid(parameters.Wf * z + parameters.bf);
% Input gate
i = sigmoid(parameters.Wi * z + parameters.bi);
% Candidate cell state
c_bar = tanh(parameters.Wc * z + parameters.bc);
% New cell state
c_next = f .* c_prev + i .* c_bar;
% Output gate
o = sigmoid(parameters.Wo * z + parameters.bo);
% New hidden state
h_next = o .* tanh(c_next);

% Store computed values for use in backpropagation
cache.x = x;
cache.h_prev = h_prev;
cache.c_prev = c_prev;
cache.z = z;
cache.f = f;
cache.i = i;
cache.c_bar = c_bar;
cache.c_next = c_next;
cache.o = o;

end