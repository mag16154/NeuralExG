function [ret]=obench10UnicycleODE(T,X)

x = X(1);
y = X(2);
z = X(3);
w = X(4);

load controller10_unicycle;

layers = 2;

last_layer_output = X;
for idx = 1:layers-1
    W_mat = W{1,idx};
    Wx_plus_bias = mtimes(W_mat, last_layer_output) + b{1,idx};
    last_layer_output = poslin(Wx_plus_bias);
end

W_mat = W{1, layers};
Wx_plus_bias = mtimes(W_mat, last_layer_output) + b{1, layers};
controller_output = poslin(Wx_plus_bias);

xdot = w * cos(z);
ydot = w * sin(z);
zdot = controller_output(2) - 20;
wdot = controller_output(1) - 20;

ret = [xdot; ydot; zdot; wdot];

