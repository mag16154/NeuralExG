function [ret]=obench8ODE(T,X)

x = X(1);
y = X(2);
z = X(3);
w = X(4);

load controller8;

layers = controller.number_of_layers;

last_layer_output = X;
for idx = 1:layers-1
    W_mat = controller.W{1,idx};
    Wx_plus_bias = mtimes(W_mat, last_layer_output) + controller.b{1,idx};
    last_layer_output = poslin(Wx_plus_bias);
end

W_mat = controller.W{1, layers};
Wx_plus_bias = mtimes(W_mat, last_layer_output) + controller.b{1, layers};
controller_output = poslin(Wx_plus_bias);

xdot = y;
ydot = -9.8 * z + 1.6 * z * z * z + x * w * w;
zdot = w;
wdot = controller_output -10;

ret = [xdot; ydot; zdot; wdot];

