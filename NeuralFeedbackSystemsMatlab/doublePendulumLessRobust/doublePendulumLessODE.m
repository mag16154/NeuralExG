function [ret]=doublePendulumLessODE(T,X)

th1 = X(1);
th2 = X(2);
u1 = X(3);
u2 = X(4);

load controller_double_pendulum_less_robust;

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

T1 = controller_output(1);
T2 = controller_output(2);

xdot = u1;
ydot = u2;
zdot = 4*T1 + 2*sin(th1) - (u2^2*sin(th1 - th2))/2 + (cos(th1 - th2)*(sin(th1 - th2)*u1^2 + 8*T2 + 2*sin(th2) - cos(th1 - th2)*(- (sin(th1 - th2)*u2^2)/2 + 4*T1 + 2*sin(th1))))/(2*(cos(th1 - th2)^2/2 - 1));
wdot = -(sin(th1 - th2)*u1^2 + 8*T2 + 2*sin(th2) - cos(th1 - th2)*(- (sin(th1 - th2)*u2^2)/2 + 4*T1 + 2*sin(th1)))/(cos(th1 - th2)^2/2 - 1);

ret = [xdot; ydot; zdot; wdot];

