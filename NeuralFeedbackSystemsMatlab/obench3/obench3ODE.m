function [ret]=obench3ODE(T,X)

x = X(1);
y = X(2);

load controller3;

layers = controller.number_of_layers;

last_layer_output = X;
for idx = 1:layers
    W_mat = controller.W{1,idx};
    Wx_plus_bias = mtimes(W_mat, last_layer_output) + controller.b{1,idx};
    last_layer_output = poslin(Wx_plus_bias);
end

controller_output = last_layer_output - 2;

w = 0.01;

xdot = -x * (0.1 + (x + y)^2);
ydot = (controller_output + x + w) * (0.1 + (x + y)^2);

ret = [xdot; ydot];

