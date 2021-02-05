function [ret]=accnonlinear3LODE(T,X)

x1 = X(1);
v1 = X(2);
a1 = X(3);
x2 = X(4);
v2 = X(5);
a2 = X(6);

load controller_3_20;

layers = 4;

x_rel = x1 - x2;
v_rel = v1 - v2;
v_ego = v2;
v_set = 30;
t_gap = 1.4;
ac1 = -2;

control_input = [v_set; t_gap; v_ego; x_rel; v_rel];

last_layer_output = control_input;
for idx = 1:layers-1
    W_mat = W{1,idx};
    Wx_plus_bias = mtimes(W_mat, last_layer_output) + b{idx, 1};
    last_layer_output = poslin(Wx_plus_bias);
end

W_mat = W{1, layers};
Wx_plus_bias = mtimes(W_mat, last_layer_output) + b{layers, 1};
controller_output = purelin(Wx_plus_bias);
ac2 = controller_output;

x1dot = v1;
v1dot = a1;
a1dot = -2 * a1 + 2 * ac1 - 0.0001 * v1 * v1;
x2dot = v2;
v2dot = a2;
a2dot = -2 * a2 + 2 * ac2 - 0.0001 * v2 * v2;


ret = [x1dot; v1dot; a1dot; x2dot; v2dot; a2dot];

