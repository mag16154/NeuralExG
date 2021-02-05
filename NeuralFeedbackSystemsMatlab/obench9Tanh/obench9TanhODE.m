function [ret]=obench9TanhODE(T,X)

x = X(1);
y = X(2);
z = X(3);
w = X(4);

load controller9tanh;

layers = 4;

last_layer_output = X;
for idx = 1:layers-1
    W_mat = W{1,idx};
    Wx_plus_bias = mtimes(W_mat, last_layer_output) + b{1,idx};
    last_layer_output = poslin(Wx_plus_bias);
end

W_mat = W{1, layers};
Wx_plus_bias = mtimes(W_mat, last_layer_output) + b{1, layers};
controller_output = tansig(Wx_plus_bias);

xdot = y;
ydot = - x + 0.1*sin(z);
zdot = w;
wdot = controller_output;

ret = [xdot; ydot; zdot; wdot];

