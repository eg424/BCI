function [dprev_h, dprev_c, dWf, dbf, dWi, dbi, dWo, dbo, dWc, dbc] = lstmCellBackward(dnext_h, dnext_c, cache, parameters)
    % Backprop for a single LSTM cell
    % dnext_h: gradient of loss w.r.t. current hidden state
    % dnext_c: gradient of loss w.r.t. current cell state
    % cache: cached forward pass variables
    % parameters: LSTM model parameters
    
    % Retrieve cache variables
    z = cache.z;
    f = cache.f;
    i = cache.i;
    c_bar = cache.c_bar;
    c_next = cache.c_next;
    o = cache.o;
    c_prev = cache.c_prev;
    h_prev = cache.h_prev;
    
    % Compute gradients w.r.t. output gate
    do = dnext_h .* tanh(c_next);
    do = do .* o .* (1 - o);  % Sigmoid derivative
    
    % Compute gradients w.r.t. cell state
    dc_next = dnext_h .* o .* (1 - tanh(c_next).^2) + dnext_c;
    
    % Compute gradients w.r.t. forget gate
    df = dc_next .* c_prev;
    df = df .* f .* (1 - f);  % Sigmoid derivative
    
    % Compute gradients w.r.t. input gate
    di = dc_next .* c_bar;
    di = di .* i .* (1 - i);  % Sigmoid derivative
    
    % Compute gradients w.r.t. candidate cell state
    dc_bar = dc_next .* i;
    dc_bar = dc_bar .* (1 - c_bar.^2);  % Tanh derivative
    
    % Compute gradients for weight matrices
    dz = [df; di; do; dc_bar];  % Stack all gate gradients
    dWf = df * z';
    dWi = di * z';
    dWo = do * z';
    dWc = dc_bar * z';
    
    % Compute gradients for biases
    dbf = sum(df, 2);
    dbi = sum(di, 2);
    dbo = sum(do, 2);
    dbc = sum(dc_bar, 2);
    
    % Compute gradients for previous hidden state
    dprev_h = parameters.Wf(:, 1:end-1)' * df + ...
              parameters.Wi(:, 1:end-1)' * di + ...
              parameters.Wo(:, 1:end-1)' * do + ...
              parameters.Wc(:, 1:end-1)' * dc_bar;
    
    % Compute gradients for previous cell state
    dprev_c = dc_next .* f;
end