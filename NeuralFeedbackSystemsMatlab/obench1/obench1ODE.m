function [ret]=obench1ODE(T,X)

x = X(1);
y = X(2);

load controller1;

layers = network.number_of_layers;

last_layer_output = X;
for idx = 1:layers-1
    W_mat = network.W{1,idx};
    Wx_plus_bias = mtimes(W_mat, last_layer_output) + network.b{1,idx};
    last_layer_output = poslin(Wx_plus_bias);
end

W_mat = network.W{1, layers};
Wx_plus_bias = mtimes(W_mat, last_layer_output) + network.b{1, layers};
controller_output = purelin(Wx_plus_bias);

% controller_output

w = 0.01;

xdot = y - x * x * x + w;
ydot = controller_output;

ret = [xdot; ydot];

