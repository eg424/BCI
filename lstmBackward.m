
function grads = lstmBackward(dh_final, caches, parameters)
    % LSTM backward pass through time
    % dh_final: gradient of loss w.r.t. final hidden state
    % caches: stored forward pass variables for each time step
    % parameters: LSTM model parameters
    
    T = length(caches);  % Number of time steps
    hiddenDim = size(parameters.Wf, 1);
    inputDim = length(caches{1}.x);
    
    % Initialize gradient storage
    grads.dWf = zeros(size(parameters.Wf));
    grads.dbf = zeros(size(parameters.bf));
    grads.dWi = zeros(size(parameters.Wi));
    grads.dbi = zeros(size(parameters.bi));
    grads.dWo = zeros(size(parameters.Wo));
    grads.dbo = zeros(size(parameters.bo));
    grads.dWc = zeros(size(parameters.Wc));
    grads.dbc = zeros(size(parameters.bc));
    
    % Initialize backpropagated gradients to zero
    dnext_h = dh_final;
    dnext_c = zeros(hiddenDim, 1);  % Cell state gradient
    
    % Backpropagate through time (BPTT)
    for t = T:-1:1
        cache = caches{t};
        
        % Compute gradients for this time step
        [dprev_h, dprev_c, dWf, dbf, dWi, dbi, dWo, dbo, dWc, dbc] = lstmCellBackward(dnext_h, dnext_c, cache, parameters);
        
        % Accumulate gradients
        grads.dWf = grads.dWf + dWf;
        grads.dbf = grads.dbf + dbf;
        grads.dWi = grads.dWi + dWi;
        grads.dbi = grads.dbi + dbi;
        grads.dWo = grads.dWo + dWo;
        grads.dbo = grads.dbo + dbo;
        grads.dWc = grads.dWc + dWc;
        grads.dbc = grads.dbc + dbc;
        
        % Pass gradients to previous time step
        dnext_h = dprev_h;
        dnext_c = dprev_c;
    end
end
