function [ret]=obench7ODE(T,X)

x = X(1);
y = X(2);
z = X(3);

load controller7;

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

u_inp = (controller_output - 100) * 0.1;

xdot = z^3 -y;
ydot = z;
zdot = u_inp;

ret = [xdot; ydot; zdot];

