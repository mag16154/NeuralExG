function [ret]=invertedPendulumODE(T,X)

x = X(1);
y = X(2);
z = X(3);
w = X(4);

load controller_invertedPendulum;

layers = 2;

last_layer_output = X;
for idx = 1:layers-1
    W_mat = nnetwork.W{1,idx};
    Wx_plus_bias = mtimes(W_mat, last_layer_output) + nnetwork.b{1,idx};
    last_layer_output = poslin(Wx_plus_bias);
end

W_mat = nnetwork.W{1, layers};
Wx_plus_bias = mtimes(W_mat, last_layer_output) + nnetwork.b{1, layers};
controller_output = purelin(Wx_plus_bias);

xdot = y;
ydot = 0.004300000000000637 * w - 2.75 * z + 1.9399999999986903 * controller_output - 10.950000000011642 * y;
zdot = w;
wdot = 28.580000000016298 * z - 0.04399999999998272 * w - 4.440000000002328 * controller_output + 24.919999999983702 * y;

ret = [xdot; ydot; zdot; wdot];

