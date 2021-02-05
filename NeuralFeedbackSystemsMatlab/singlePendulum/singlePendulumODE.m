function [ret]=singlePendulumODE(T,X)

x = X(1);
y = X(2);

load controller_single_pendulum;
	
l = 0.5;
m = 0.5;
g = 1;
c = 0;

layers = 3;

last_layer_output = X;
for idx = 1:layers-1
    W_mat = W{1,idx};
    Wx_plus_bias = mtimes(W_mat, last_layer_output) + b{1,idx};
    last_layer_output = poslin(Wx_plus_bias);
end

W_mat = W{1, layers};
Wx_plus_bias = mtimes(W_mat, last_layer_output) + b{1, layers};
controller_output = purelin(Wx_plus_bias);

% controller_output


xdot = y;
ydot = g/l * sin(x) + (controller_output - c*y)/(m*l^2);

ret = [xdot; ydot];

