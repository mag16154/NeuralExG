function [ret]=obench6ODE(T,X)

x = X(1);
y = X(2);
z = X(3);

load controller6;

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

w = 0.01;

xdot = -1 * x * x * x + y;
ydot = y * y * y + z;
zdot = controller_output + w;

ret = [xdot; ydot; zdot];

