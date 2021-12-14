import numpy as np

def rhsODE(state, time, dynamics,plant):
	# return the value that is given by the following function
	# change this function to anything else you want.

	if dynamics is 'None':
		return vanderpol(state, time)
	if dynamics is 'Gravity':
		return gravity(state, time)
	if dynamics is 'Vanderpol':
		return vanderpol(state, time)
	if dynamics is 'Brussellator':
		return brussellator(state, time)
	if dynamics is 'Jetengine':
		return jetengine(state, time)
	if dynamics is 'Lorentz':
		return lorentz(state, time)
	if dynamics is 'Buckling':
		return buckling(state, time)
	if dynamics is 'Lotka':
		return lotkavolterra(state, time)
	if dynamics is 'Lacoperon':
		return lacoperon(state, time)
	if dynamics is 'Roesseler':
		return roesseler(state, time)
	if dynamics is 'Steam':
		return steam(state, time)
	if dynamics is 'SpringPendulum':
		return springpendulum(state, time)
	if dynamics is 'CoupledVanderpol':
		return coupledVanderpol(state, time)
	if dynamics is 'HybridLinearOscillator':
		return hybridLinearOscillator(state, time)
	if dynamics is 'HybridLinearOscillator1':
		return hybridLinearOscillator1(state, time)
	if dynamics is 'HybridLinearOscillator2':
		return hybridLinearOscillator2(state, time)
	if dynamics is 'SmoothHybridLinearOscillator':
		return smoothHybridLinearOscillator(state, time)
	if dynamics is 'SmoothHybridLinearOscillator1':
		return smoothHybridLinearOscillator1(state, time)
	if dynamics is 'SmoothHybridLinearOscillator2':
		return smoothHybridLinearOscillator2(state, time)
	if dynamics is 'RegularOscillator':
		return regularOscillator(state, time)
	if dynamics is 'BiologicalModel1':
		return biologicalModel_1(state, time)
	if dynamics is 'BiologicalModel2':
		return biologicalModel_2(state, time)
	if dynamics is 'BouncingBall':
		return bouncingball(state, time, plant)
	if dynamics is 'Rober':
		return rober(state, time)
	if dynamics is 'E5':
		return e5(state, time)
	if dynamics is 'Orego':
		return orego(state, time)
	if dynamics is 'LaubLoomis':
		return laubLoomis(state, time)
	if dynamics is 'SA_Nonlinear':
		return sa_nonlinear(state, time)
	if dynamics is 'DampedOsc':
		return dampedOscillator(state, time)
	if dynamics is 'OscParticle':
		return oscParticle(state, time)
	if dynamics is 'AdaptiveCruiseControl':
		return adaptiveCruise(state, time)
	if dynamics is 'PurePursuit':
		return purepursuit(state, time)
	if dynamics is 'InputSCancel':
		return inputSCancel(state, time)
	if dynamics is 'FiveDBenchmark':
		return fiveDbenchmark(state, time)
	if dynamics is 'OtherBenchC1':
		return oBenchC1(state, time, plant)
	if dynamics is 'OtherBenchC2':
		return oBenchC2(state, time, plant)
	if dynamics is 'OtherBenchC3':
		return oBenchC3(state, time, plant)
	if dynamics is 'OtherBenchC4':
		return oBenchC4(state, time, plant)
	if dynamics is 'OtherBenchC5':
		return oBenchC5(state, time, plant)
	if dynamics is 'OtherBenchC6':
		return oBenchC6(state, time, plant)
	if dynamics is 'OtherBenchC7':
		return oBenchC7(state, time, plant)
	if dynamics is 'OtherBenchC8':
		return oBenchC8(state, time, plant)
	if dynamics is 'ACCNonLinear3L' or dynamics is 'ACCNonLinear5L' or dynamics is 'ACCNonLinear7L' or dynamics is 'ACCNonLinear10L':
		return accNonlinear(state, time, plant)
	if dynamics is 'InvPendulumC':
		return iPendulumC(state, time, plant)
	if dynamics is 'CartPole' or dynamics is 'CartPoleTanh':
		return cartPole(state, time, plant)
	if dynamics is 'RobotArm':
		return robotArm(state, time)
	if dynamics is 'CartPoleLinControl':
		return cartPolewLinearControl(state, time, plant)
	if dynamics is 'CartPoleLinControl2':
		return cartPolewLinearControl(state, time, plant)
	if dynamics is 'OtherBenchC9':
		return oBenchC9(state, time, plant)
	if dynamics is 'OtherBenchC9Tanh' or dynamics is 'OtherBenchC9Sigmoid':
		return oBenchC9hetero(state, time, plant)
	if dynamics is 'SinglePendulum':
		return singlePendulum(state, time, plant)
	if dynamics is 'DoublePendulumLess' or dynamics is 'DoublePendulumMore':
		return doublePendulum(state, time, plant)
	if dynamics is 'OtherBenchC10':
		return oBenchC10(state, time, plant)
	if dynamics is 'Airplane':
		return airplane(state, time, plant)
	if dynamics is 'Scalable5D':
		return scalable5D(state, time)
	if dynamics is 'DuffingOsc':
		return duffingOsc(state, time)
	if dynamics is 'AcuteInflame':
		return acuteInflame(state, time)


def vanderpol(state, time):

	# Vanderpol oscilltor
	# dxdt = y; dydt = mu*(1-x*x)*y -x;
	# typical value of mu = 1
	# state = [x,y]

	x = state[0]
	y = state[1]

	mu = 1
	dxdt = y
	dydt = mu*(1-x*x)*y - x

	rhs = [dxdt, dydt]
	return rhs


def bouncingball(state, time, plant):

	x = state[0]
	v = state[1]

	print("x is " + str(x) + " and v is " + str(v))

	# vel = plant.get_vel(state)
	mode = plant.get_mode(state)

	# if x == 0 and v <= 0:
	# 	dxdt = v
	# 	dvdt = -9.81
	# else:
	# 	dxdt = v
	# 	dvdt = -100*x - 4*v - 9.81

	if mode == 1:
		dxdt = v
		dvdt = -9.81
	else:
		dxdt = v
		dvdt = -100*x - 4*v - 9.81

	rhs = [dxdt, dvdt]
	return rhs


def gravity(state, time):
	# gravity
	# dxdt = y; dydt = -9.8;
	# typical value of A = 1, B = 1.5
	# state = [x,y]

	x = state[0]
	y = state[1]

	dxdt = y
	dydt = -9.8 - 0.1 * y

	rhs = [dxdt, dydt]
	return rhs


def jetengine(state, time):
	# jet-engine dynamics
	# dxdt = -1*y - 1.5*x*x - 0.5*x*x*x - 0.5; dydt = 3*x - y;
	# state = [x,y]

	x = state[0]
	y = state[1]

	dxdt = -1 * y - 1.5 * x * x - 0.5 * x * x * x - 0.5
	dydt = 3 * x - y

	rhs = [dxdt, dydt]
	return rhs


def brussellator(state, time):
	# Brussellator
	# dxdt = A + x^2y -Bx - x; dydt = Bx - x^2y;
	# typical value of A = 1, B = 1.5
	# state = [x,y]

	x = state[0]
	y = state[1]

	A = 1
	B = 1.5
	dxdt = A + x * x * y - B * x - x
	dydt = B * x - x * x * y

	rhs = [dxdt, dydt]
	return rhs


def buckling(state, time):
	# Buckling Column
	# dxdt = y; dydt = 2x - x*x*x - 0.2*y + 0.1;
	# state = [x,y]

	x = state[0]
	y = state[1]

	dxdt = y
	dydt = 2*x - x*x*x - 0.2*y + 0.1

	rhs = [dxdt, dydt]
	return rhs


def lotkavolterra(state, time):
	# Predator prey also known as Lotka-Volterra
	# dxdt = x*(alpha - beta*y); dydt = -1*y*(gamma - delta*x)
	# state = [x,y]
	# typical values of alpha = 1.5, beta = 1, gamma = 3, delta = 1

	x = state[0]
	y = state[1]

	alpha = 1.5
	beta = 1
	gamma = 3
	delta = 1

	dxdt = x*(alpha - beta*y)
	dydt = -1*y*(gamma - delta*x)

	rhs = [dxdt, dydt]
	return rhs


def lacoperon(state, time):
	# Lac-operon model
	# dIidt = -0.4*Ii*Ii*((0.0003*G*G + 0.008) / (0.2*Ii*Ii + 2.00001) ) + 0.012 + (0.0000003 * (54660 - 5000.006*Ii) *
	# (0.2*Ii*Ii + 2.00001)) / (0.00036*G*G + 0.00960018 + 0.000000018*Ii*Ii)
	# DGdt = -0.0006*G*G + (0.000000006*G*G + 0.00000016) / (0.2*Ii*Ii + 2.00001) +
	# (0.0015015*Ii*(0.2*Ii*Ii + 2.00001)) / (0.00036*G*G + 0.00960018 + 0.000000018*Ii*Ii)
	# state = [Ii, G]

	Ii = state[0]
	G = state[1]

	dIidt = -0.4*Ii*Ii*((0.0003*G*G + 0.008) / (0.2*Ii*Ii + 2.00001)) + 0.012 + (0.0000003 * (54660 - 5000.006*Ii) *
												(0.2*Ii*Ii + 2.00001)) / (0.00036*G*G + 0.00960018 + 0.000000018*Ii*Ii)
	dGdt = -0.0006*G*G + (0.000000006*G*G + 0.00000016) / (0.2*Ii*Ii + 2.00001) + (0.0015015*Ii*(0.2*Ii*Ii + 2.00001)) / \
												(0.00036*G*G + 0.00960018 + 0.000000018*Ii*Ii)

	rhs = [dIidt, dGdt]
	return rhs


def roesseler(state, time):
	# Roesseler attractor dynamics
	# dxdt = -y-z; dydt = x+a*y; dzdt = b + z*(x-c)
	# typical values of a = 0.2, b = 0.2, c = 5.7
	# state = [x, y, z]

	a = 0.2
	b = 0.2
	c = 5.7

	x = state[0]
	y = state[1]
	z = state[2]

	dxdt = -y - z
	dydt = x + a*y
	dzdt = b + z*(x-c)

	rhs = [dxdt, dydt, dzdt]
	return rhs


def steam(state, time):
	# Steam Governer system
	# dxdt = y; dydt = z*z*sin(x)*cos(x) - sin(x) - epsilon*y; dzdt = alpha*(cos(x) - beta)
	# asymptotic stability when epsilon > 2*alpha*(beta^(3/2))
	# typical values of epsilon = 3, alpha = 1, beta = 1

	epsilon = 3
	alpha = 1
	beta = 1

	x = state[0]
	y = state[1]
	z = state[2]

	dxdt = y
	dydt = z*z*np.sin(x)*np.cos(x) - np.sin(x) - epsilon*y
	dzdt = alpha*(np.cos(x) - beta)

	rhs = [dxdt, dydt, dzdt]
	return rhs


def lorentz(state, time):
	# lorenz attractor dynamics
	# dxdt = sigma*(y - x); dydt = x*(rho-z) - y; dzdt = x*y - beta*z;
	# state = [x,y,z]
	# typically sigma = 10; rho = 8.0/3.0; beta = 28;

	sigma = 10.0
	rho = 28.0
	beta = 8.0 / 3.0

	x = state[0]
	y = state[1]
	z = state[2]

	dxdt = sigma * (y - x)
	dydt = x * (rho - z) - y
	dzdt = x * y - beta * z

	rhs = [dxdt, dydt, dzdt]
	return rhs


def springpendulum(state, time):
	# Spring pendulum system
	# drdt = vr; dthetadt = omega;
	# dvrdt = r*omega*omega + g*np.cos(theta) - k*(r-L)
	# domegadt = (-1.0/r)*(2*vr*omega+g*np.sin(theta))
	# state = [r, theta, vr, omega]
	# k = 2; L = 1; g = 9.8

	k = 2.0
	L = 1.0
	g = 9.8

	r = state[0]
	theta = state[1]
	vr = state[2]
	omega = state[3]

	drdt = vr
	dthetadt = omega
	dvrdt = r*omega*omega + g*np.cos(theta) - k*(r - L)
	domegadt = (-1.0/r) * (2*vr*omega + g*np.sin(theta))

	rhs = [drdt, dthetadt, dvrdt, domegadt]
	return rhs


def coupledVanderpol(state, time):
	# Coupled Vanderpol oscillator
	# dx1dt = y1; dy1dt = (1 - x1*x1)*y1 - x1 + (x2-x1)
	# dx2dt = y2; dy2dt = (1 - x2*x2)*y2 - x2 + (x1 -x2)
	# state = [x1, y1, x2, y2]

	x1 = state[0]
	y1 = state[1]
	x2 = state[2]
	y2 = state[3]

	dx1dt = y1
	dy1dt = (1 - x1*x1)*y1 - x1 + (x2 - x1)
	dx2dt = y2
	dy2dt = (1 - x2*x2)*y2 - x2 + (x1 - x2)

	rhs = [dx1dt, dy1dt, dx2dt, dy2dt]
	return rhs


def hybridLinearOscillator1(state, time):
	# when x > 0, the speed 2x,
	# when x <= 0, then speed 1x
	# dx1dt = y1; dy1dt = -x1;
	# dx1dt = 2y1; dy1dt = -2*x1;

	x1 = state[0]
	y1 = state[1]

	dx1dt = 0
	dy1dt = 0

	dx1dt = y1
	dy1dt = -1*x1

	rhs = [dx1dt, dy1dt]

	return rhs


def hybridLinearOscillator2(state, time):
	# when x > 0, the speed 2x,
	# when x <= 0, then speed 1x
	# dx1dt = y1; dy1dt = -x1;
	# dx1dt = 2y1; dy1dt = -2*x1;

	x1 = state[0]
	y1 = state[1]

	dx1dt = 0
	dy1dt = 0

	dx1dt = 2*y1
	dy1dt = -2*x1

	rhs = [dx1dt, dy1dt]

	return rhs


def hybridLinearOscillator(state, time):
	# when x > 0, the speed 2x,
	# when x <= 0, then speed 1x
	# dx1dt = y1; dy1dt = -x1;
	# dx1dt = 2y1; dy1dt = -2*x1;

	x1 = state[0]
	y1 = state[1]

	dx1dt = 0
	dy1dt = 0

	if x1 <= 0:
		dx1dt = y1
		dy1dt = -1*x1
	else:
		dx1dt = 2*y1
		dy1dt = -2*x1

	rhs = [dx1dt, dy1dt]

	return rhs


def smoothHybridLinearOscillator1(state, time):
	# smoothen the dynamics by performing a linear interpolation.
	# when x > 0, the speed 2x,
	# when x <= 0, then speed 1x
	# dx1dt = y1; dy1dt = -x1;
	# dx1dt = 2y1; dy1dt = -2*x1;

	x1 = state[0]
	y1 = state[1]
	scale = 1.0

	g1 = 1.0/(1.0 + np.exp(scale*x1))
	g2 = 1.0/(1.0 + np.exp(scale*-1*x1))

	lambda1 = g1/(g1+g2)

	dx2dt = 0
	dy2dt = 0

	dx1dt = y1
	dy1dt = -1*x1

	fdx1dt = lambda1*dx1dt + (1-lambda1)*dx2dt
	fdy1dt = lambda1*dy1dt + (1-lambda1)*dy2dt

	rhs = [fdx1dt, fdy1dt]
	# print rhs

	return rhs


def smoothHybridLinearOscillator2(state, time):
	# smoothen the dynamics by performing a linear interpolation.
	# when x > 0, the speed 2x,
	# when x <= 0, then speed 1x
	# dx1dt = y1; dy1dt = -x1;
	# dx1dt = 2y1; dy1dt = -2*x1;

	x1 = state[0]
	y1 = state[1]
	scale = 1.0

	g1 = 1.0/(1.0 + np.exp(scale*x1))
	g2 = 1.0/(1.0 + np.exp(scale*-1*x1))

	lambda1 = g1/(g1+g2)

	dx1dt = 0
	dy1dt = 0
	dx2dt = 2*y1
	dy2dt = -2*x1

	fdx1dt = lambda1*dx1dt + (1-lambda1)*dx2dt
	fdy1dt = lambda1*dy1dt + (1-lambda1)*dy2dt

	rhs = [fdx1dt, fdy1dt]

	# print rhs

	return rhs


def smoothHybridLinearOscillator(state, time):
	# smoothen the dynamics by performing a linear interpolation.
	# when x > 0, the speed 2x,
	# when x <= 0, then speed 1x
	# dx1dt = y1; dy1dt = -x1;
	# dx1dt = 2y1; dy1dt = -2*x1;

	x1 = state[0]
	y1 = state[1]
	scale = 1.0

	g1 = 1.0/(1.0 + np.exp(scale*x1))
	g2 = 1.0/(1.0 + np.exp(scale*-1*x1))

	lambda1 = g1/(g1+g2)

	dx1dt = 0
	dy1dt = 0
	dx2dt = 0
	dy2dt = 0

	if x1 <= 0 :
		dx1dt = y1
		dy1dt = -1*x1
	if x1 > 0 :
		dx2dt = 2*y1
		dy2dt = -2*x1

	fdx1dt = lambda1*dx1dt + (1-lambda1)*dx2dt
	fdy1dt = lambda1*dy1dt + (1-lambda1)*dy2dt

	rhs = [fdx1dt, fdy1dt]

	# print rhs

	return rhs


def regularOscillator(state, time):

	x1 = state[0]
	y1 = state[1]

	dx1dt = y1
	dy1dt = -1*x1

	rhs = [dx1dt, dy1dt]

	return rhs


def biologicalModel_1(state, time):

	x1 = state[0]
	x2 = state[1]
	x3 = state[2]
	x4 = state[3]
	x5 = state[4]
	x6 = state[5]
	x7 = state[6]

	dx1dt = -0.4*x1 + 5*x3*x4
	dx2dt = 0.4*x1 - x2
	dx3dt = x2 - 5*x3*x4
	dx4dt = 5*x5*x6 - 5*x3*x4
	dx5dt = -5*x5*x6 + 5*x3*x4
	dx6dt = 0.5*x7 - 5*x5*x6
	dx7dt = -0.5*x7 + 5*x5*x6

	rhs = [dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt]

	return rhs


def laubLoomis(state, time):

	x1 = state[0]
	x2 = state[1]
	x3 = state[2]
	x4 = state[3]
	x5 = state[4]
	x6 = state[5]
	x7 = state[6]
	# x8 = state[7]

	dx1dt = 1.4*x3 - 0.9*x1
	dx2dt = 2.5*x5 - 1.5*x2
	dx3dt = 0.6*x7 - 0.8*x2*x3
	dx4dt = 2.0 - 1.3*x3*x4
	dx5dt = 0.7*x1 - x4*x5
	dx6dt = 0.3*x1 - 3.1*x6
	dx7dt = 1.8*x6 - 1.5*x2*x7
	# dx8dt = 1

	rhs = [dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt]

	return rhs


def biologicalModel_2(state, time):

	x1 = state[0]
	x2 = state[1]
	x3 = state[2]
	x4 = state[3]
	x5 = state[4]
	x6 = state[5]
	x7 = state[6]
	x8 = state[7]
	x9 = state[8]

	dx1dt = 50*x3 - x1*x6
	dx2dt = 100*x4 - x2*x6
	dx3dt = x1*x6 - 50*x3
	dx4dt = x2*x6 - 100*x4
	dx5dt = 500*x3 + 50*x1 - 10*x5
	dx6dt = 50*x5 + 50*x3 + 100*x4 - x6*(x1 + x2 + 2*x8 + 1)
	dx7dt = 50*x4 + 0.01*x2 - 0.5*x7
	dx8dt = 5*x7 - 2*x6*x8 + x9 - 0.2*x8
	dx9dt = 2*x6*x8 - x9

	rhs = [dx1dt, dx2dt, dx3dt, dx4dt, dx5dt, dx6dt, dx7dt, dx8dt, dx9dt]

	return rhs


'''https://github.com/schillic/HA2Stateflow'''


def rober(state, time):
	k1 = 0.04
	k2 = 30000000
	k3 = 10000

	x = state[0]
	y = state[1]
	z = state[2]

	dxdt = -k1*x + k3*y*z
	dydt = k1*x - k2*y*y - k3*y*z
	dzdt = k2*y*y

	rhs = [dxdt, dydt, dzdt]
	return rhs


def orego(state, time):
	s = 77.27
	w = 0.161
	q = 0.08375

	x = state[0]
	y = state[1]
	z = state[2]

	dxdt = s*(y-x*y+x-q*x*x)
	dydt = (1/s)*(-y-x*y+z)
	dzdt = w*(y-z)

	rhs = [dxdt, dydt, dzdt]
	return rhs


def e5(state, time):
	A = 0.00000000789
	B = 11000000
	C = 1130
	M = 1000000

	x = state[0]
	y = state[1]
	z = state[2]
	w = state[3]

	dxdt = -A*x - B*x*z
	dydt = A*x - M*C*y*z
	dzdt = A*x - B*x*z - M*C*y*z + C*w
	dwdt = B*x*z - C*w

	rhs = [dxdt, dydt, dzdt, dwdt]
	return rhs


def sa_nonlinear(state, time):
	pi = 22/7
	x = state[0]
	y = state[1]

	dxdt = x - y + 0.1*time
	dydt = y * np.cos(2 * pi * y) - x * np.sin(2 * pi * x) + 0.1*time
	rhs = [dxdt, dydt]
	return rhs


def dampedOscillator(state, time):

	x = state[0]
	y = state[1]

	dxdt = -0.1 * x + y
	dydt = -1 * x - 0.1 * y

	rhs = [dxdt, dydt]

	return rhs


def oscParticle(state, time):

	x = state[0]
	y = state[1]
	z = state[2]

	dxdt = -0.1 * x - y
	dydt = x - 0.1 * y
	dzdt = -0.15 * z

	rhs = [dxdt, dydt, dzdt]

	return rhs


def adaptiveCruise(state, time):

	s = state[0]
	v = state[1]
	a = state[2]
	vf = 20

	dsdt = -1 * v + vf
	dvdt = a
	dadt = s - 4 * v + 3 * vf - 3 * a - 10

	rhs = [dsdt, dvdt, dadt]

	return rhs


# A Linear Programming-based Iterative Approach to Stabilizing Polynomial Dynamics
# Control Lyapunov Function Design by Cancelling Input
def inputSCancel(state, time):
	x = state[0]
	y = state[1]
	z = state[2]

	dxdt = -x + y - z
	dydt = -x*z -x - y
	dzdt = -x

	rhs = [dxdt, dydt, dzdt]

	return rhs

# Simulation Based Computation of
# Certificates for Safety of Hybrid Dynamical Systems
# Stefan Ratschan Oct 2018


def scalable5D(state, time):
	x1 = state[0]
	x2 = state[1]
	x3 = state[2]
	x4 = state[3]
	x5 = state[4]

	dx1dt = 1 + 0.5 * (x2 + 2 * x3 + x4)
	dx2dt = x3
	dx3dt = -10 * np.sin(x2) - x2
	dx4dt = x5
	dx5dt = -10 * np.sin(x4) - x2

	rhs = [dx1dt, dx2dt, dx3dt, dx4dt, dx5dt]

	return rhs

# https://arxiv.org/pdf/2104.13902.pdf
# dataObject.setLowerBound([0.95, -0.05])
# dataObject.setUpperBound([1.05, 0.05])


def duffingOsc(state, time):
	alpha = 0.05
	gamma = 0.4
	omega = 1.3
	x1 = state[0]
	x2 = state[1]

	dx1dt = x2
	dx2dt = -alpha * x2 + x1 - x1*x1*x1 + gamma * np.cos(omega * time)

	rhs = [dx1dt, dx2dt]

	return rhs

# Breach
# https://www.mathematics.pitt.edu/sites/default/files/research-pdfs/reynolds.pdf


def acuteInflame(state, time):
	k_pm = 0.6
	k_mp = 0.01
	s_m = 0.005
	u_m = 0.002
	k_pg = 0.3
	p_inf = 20 * 1000**2
	k_pn = 1.8
	k_np = 0.1
	k_nn = 0.01
	s_nr = 0.08
	u_nr = 0.12
	u_n = 0.05
	k_nd = 0.02
	k_dn = 0.35
	x_dn = 0.06
	u_d = 0.02
	c_inf = 0.28
	s_c = 0.0125
	k_cn = 0.04
	k_cnd = 48
	u_c = 0.1

	P = state[0]
	Na = state[1]
	D = state[2]
	Ca = state[3]

	def f1(arg1):
		return arg1/(1+(Ca/c_inf)**2)

	def fs(arg2):
		return (arg2**6)/(x_dn**6 + arg2**6)

	dPdt = k_pg * P * (1 - (P/p_inf)) - (k_pm * s_m * P)/(u_m + k_mp * P) - k_pn * f1(Na) * P
	dNadt = (s_nr * f1(k_nn*Na + k_np*P + k_nd*D))/(u_nr + f1(k_nn*Na + k_np*P + k_nd*D)) - u_n * Na
	dDdt = k_dn * fs(f1(Na)) - u_d * D
	dCadt = s_c + (k_cn * f1(Na + k_cnd * D))/(1 + f1(Na + k_cnd*D)) - u_c * Ca

	rhs = [dPdt, dNadt, dDdt, dCadt]

	return rhs


def fiveDbenchmark(state, time):

	x1 = state[0]
	x2 = state[1]
	x3 = state[2]
	x4 = state[3]
	x5 = state[4]

	# dx1dt = 0.8*x2 - 2.5*x3 - x4
	# dx2dt = - x1*x4 - x3
	# dx3dt = - 0.5*x5
	# dx4dt = 2*x3*x5
	# dx5dt = x3 - 0.1*x4

	dx1dt = 1.8*x2 - 2.2*x3 + 0.5*x2*x5
	dx2dt = - x1*x4 - x3
	dx3dt = - 0.5*x5
	dx4dt = 3.0*x3*x5
	dx5dt = 1.5*x3 - 0.1*x4

	# dx1dt = -0.1*x1*x1 - 0.4*x1*x4 - x1+x2 + 3*x3 + 0.5*x4
	# dx2dt = x2*x2 - 0.5*x2*x5 + x1 + x3
	# dx3dt = 0.5*x3*x3 + x1 - x2 + 2*x3+ 0.1*x4 - 0.5*x5
	# dx4dt = x2 + 2*x3 + 0.1*x4 - 0.2*x5 - 3.467*x1 + 1.11* x2 - 2.013*x3 + 0.6515*x4 + 2.3716*x5
	# dx5dt = x3 - 0.1*x4 + 5 * x1 - 2.9971*x2 - 0.78656*x3 - 1.0308*x4 - 2.0555*x5

	rhs = [dx1dt, dx2dt, dx3dt, dx4dt, dx5dt]

	return rhs

# https://github.com/verivital/ARCH-2019
# https://easychair.org/publications/open/BFKs


def oBenchC1(state, time, plant):
	x = state[0]
	y = state[1]

	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	u_inp = controller_output[-1]
	# print("Hi inside diffEq")
	# print(state, controller_output)

	dxdt = y - x*x*x + 0.01
	dydt = u_inp

	rhs = [dxdt, dydt]
	# print(controller_output)

	return rhs


def oBenchC2(state, time, plant):
	x = state[0]
	y = state[1]

	w = 0.01
	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	u_inp = controller_output[-1]

	dxdt = y
	dydt = u_inp * y * y - x + w

	rhs = [dxdt, dydt]
	# print(controller_output)

	return rhs


def oBenchC3(state, time, plant):
	x = state[0]
	y = state[1]

	w = 0.01
	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	u_inp = controller_output[-1] - 2

	dxdt = -x * (0.1 + (x + y)**2)
	dydt = (u_inp + x + w) * (0.1 + (x + y)**2)

	rhs = [dxdt, dydt]
	# print(controller_output)

	return rhs


def oBenchC4(state, time, plant):
	x = state[0]
	y = state[1]
	z = state[2]

	w = 0.01
	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	u_inp = controller_output[-1]

	dxdt = y + 0.5 * z * z
	dydt = z + w
	dzdt = u_inp

	rhs = [dxdt, dydt, dzdt]
	# print(controller_output)

	return rhs


def oBenchC5(state, time, plant):
	x = state[0]
	y = state[1]
	z = state[2]

	w = 0.01
	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	u_inp = controller_output[-1]

	dxdt = -x + y - z + w
	dydt = -x * (z + 1) - y
	dzdt = -x + u_inp

	rhs = [dxdt, dydt, dzdt]
	# print(controller_output)

	return rhs


def oBenchC6(state, time, plant):
	x = state[0]
	y = state[1]
	z = state[2]

	w = 0.01
	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	u_inp = controller_output[-1]

	dxdt = -x * x * x + y
	dydt = y * y * y + z
	dzdt = u_inp + w

	rhs = [dxdt, dydt, dzdt]
	# print(controller_output)

	return rhs


def oBenchC7(state, time, plant):
	x = state[0]
	y = state[1]
	z = state[2]

	w = 0.0
	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	u_inp = (controller_output[-1] - 100) * 0.1

	dxdt = z * z * z - y + w
	dydt = z
	dzdt = u_inp

	rhs = [dxdt, dydt, dzdt]
	# print(controller_output)

	return rhs


def oBenchC8(state, time, plant):
	x1 = state[0]
	x2 = state[1]
	x3 = state[2]
	x4 = state[3]

	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	u_inp = controller_output[-1] - 10

	dx1dt = x2
	dx2dt = -9.8 * x3 + 1.6 * x3 * x3 * x3 + x1 * x4 * x4
	dx3dt = x4
	dx4dt = u_inp

	rhs = [dx1dt, dx2dt, dx3dt, dx4dt]
	# print(controller_output)

	return rhs


# https://github.com/verivital/ARCH-2019
# https://easychair.org/publications/open/BFKs

def iPendulumC(state, time, plant):
	x = state[0]  # position
	v = state[1]  # velocity
	z = state[2]  # theta
	w = state[3]  # omega

	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	u_inp = controller_output[-1]

	dxdt = v
	dydt = 0.004300000000000637 * w - 2.75 * z + 1.9399999999986903 * u_inp - 10.950000000011642 * v
	dzdt = w
	dwdt = 28.580000000016298 * z - 0.04399999999998272 * w - 4.440000000002328 * u_inp + 24.919999999983702 * v

	rhs = [dxdt, dydt, dzdt, dwdt]
	# print(controller_output)

	return rhs

# https://gitlab.com/goranf/ARCH-COMP/-/tree/master/2019/AINNCS/verisig/benchmarks/Cartpole


def cartPole(state, time, plant):
	x1 = state[0]
	x2 = state[1]
	x3 = state[2]
	x4 = state[3]

	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	if controller_output[0] > controller_output[1]:
		u_inp = 1.3
	else:
		u_inp = 1.1

	dx1dt = x2
	dx2dt = ((u_inp + 0.05 * x4 * x4 * np.sin(x3))/1.1) - 0.05 * (9.8 * np.sin(x3) - np.cos(x3) *
			((u_inp + 0.05 * x4 * x4 * np.sin(x3))/1.1)) / (0.5 * (4/3 - 0.1 * np.cos(x3) * np.cos(x3)/1.1)) * \
			np.cos(x3)/1.1
	# dx2dt = 10.0 * u_inp / 11.0 + x4**2 * np.sin(x3) / 22.0 + 10.0 * np.cos(x3) * (49.0 * np.sin(x3) / 100.0 - np.cos(x3) * (10.0 * u_inp / 11.0 + x4**2.0 * np.sin(x3) / 22.0) / 20.0) / (11.0 * (np.cos(x3)**2.0 / 22.0 - 0.6666666666666666))
	dx3dt = x4
	dx4dt = (9.8 * np.sin(x3) - np.cos(x3) * ((u_inp + 0.05 * x4 * x4 * np.sin(x3)) / 1.1)) / (
				0.5 * (4 / 3 - 0.1 * np.cos(x3) * np.cos(x3) / 1.1))
	# dx4dt = -(49.0 * np.sin(x3) / 5.0 - np.cos(x3) * (10.0 * u_inp / 11.0 + x4**2.0 * np.sin(x3) / 22.0)) / (np.cos(x3)**2.0 / 22.0 - 0.6666666666666666)

	rhs = [dx1dt, dx2dt, dx3dt, dx4dt]

	return rhs


def accNonlinear(state, time, plant):
	x1 = state[0]
	v1 = state[1]
	a1 = state[2]
	x2 = state[3]
	v2 = state[4]
	a2 = state[5]

	x_rel = x1 - x2
	v_rel = v1 - v2
	v_ego = v2
	v_set = 30
	t_gap = 1.4
	ac1 = -2

	control_input = [v_set, t_gap, v_ego, x_rel, v_rel]
	controller_output = plant.dnn_controllers[0].performForwardPass(control_input)
	ac2 = controller_output[-1]

	dx1dt = v1
	dv1dt = a1
	da1dt = -2 * a1 + 2 * ac1 - 0.0001 * v1 * v1
	dx2dt = v2
	dv2dt = a2
	da2dt = -2 * a2 + 2 * ac2 - 0.0001 * v2 * v2

	rhs = [dx1dt, dv1dt, da1dt, dx2dt, dv2dt, da2dt]

	return rhs


def purepursuit(state, time):

	x = state[0]
	y = state[1]
	z = state[2]

	s = 1.0
	l = 1.2
	L = 0.33

	a = 0
	b = 0
	c = 12
	d = 0

	xa = x - a
	yb = y - b
	ca = c - a
	db = d - b
	ca2 = ca**2
	db2 = db**2
	l2 = l**2

	Delta = (xa * ca + yb * db)**2 - (ca2 + db2) * (xa**2 + yb**2 - l2)
	T = (xa * ca + yb * db + np.sqrt(Delta)) / (ca2 + db2)
	xdot = s * np.cos(z)
	ydot = s * np.sin(z)
	x0 = a + T * ca
	y0 = b + T * db
	y0p = -(x0 - x) * np.sin(z) + (y0 - y) * np.cos(z)

	dxdt = xdot
	dydt = ydot
	dzdt = 2 * L * y0p / l2

	rhs = [dxdt, dydt, dzdt]

	return rhs


# http://publish.illinois.edu/c2e2-tool/example/robot/

def robotArm(state, time):
	x = state[0]
	y = state[1]
	z = state[2]
	w = state[3]

	dxdt = z
	dydt = w
	dzdt = (-2 * y * z * w - 2 * x - 2 * z + 4)/(y ** 2 + 1)
	dwdt = y * z * z - y - w + 1

	rhs = [dxdt, dydt, dzdt, dwdt]

	return rhs


# https://arxiv.org/pdf/2012.07458.pdf

def cartPolewLinearControl(state, time, plant):

	f = plant.get_controller_input(state)

	M = 1
	m = 0.001
	g = 9.81
	l = 1

	sigma = state[0]
	w = state[1]
	x = state[2]
	theta = state[3]

	dsigmadt = (f * np.cos(theta) - m * l * sigma * sigma * np.cos(theta) * np.sin(theta) + (m + M) * g * np.sin(theta))/(l * (M + m * np.sin(theta) * np.sin(theta)))
	dwdt = (f + m * np.sin(theta) * (-l * sigma * sigma + g * np.cos(theta)))/(M + m * np.sin(theta) * np.sin(theta))
	dxdt = w
	dthetadt = sigma

	rhs = [dsigmadt, dwdt, dxdt, dthetadt]

	return rhs


# http://www.roboticsproceedings.org/rss07/p41.pdf

def cartPoleLinearControl2(state, time, plant):
	f = plant.get_controller_input(state)

	pos = state[0]
	theta = state[1]
	vel = state[2]
	avel = state[3]  # ang velocity

	dposdt = vel
	dthetadt = avel
	dveldt = -0.75 * theta * theta * theta - 0.01 * theta * theta * f - 0.05 * theta * avel * avel + 0.98 * theta + 0.1 * f
	daveldt = -5.75 * theta * theta * theta - 0.12 * theta * theta * f - 0.1 * theta * avel * avel + 21.56 * theta + 0.2 * f

	rhs = [dposdt, dthetadt, dveldt, daveldt]

	return rhs

# https://easychair.org/publications/open/Jvwg


def oBenchC9(state, time, plant):
	x1 = state[0]
	x2 = state[1]
	x3 = state[2]
	x4 = state[3]

	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	u_inp = controller_output[-1] - 10

	dx1dt = x2
	dx2dt = - x1 + 0.1 * np.sin(x3)
	dx3dt = x4
	dx4dt = u_inp

	rhs = [dx1dt, dx2dt, dx3dt, dx4dt]
	# print(controller_output)

	return rhs


def oBenchC9hetero(state, time, plant):
	x1 = state[0]
	x2 = state[1]
	x3 = state[2]
	x4 = state[3]

	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	u_inp = controller_output[-1]

	dx1dt = x2
	dx2dt = - x1 + 0.1 * np.sin(x3)
	dx3dt = x4
	dx4dt = u_inp

	rhs = [dx1dt, dx2dt, dx3dt, dx4dt]

	return rhs


def singlePendulum(state, time, plant):
	x = state[0]
	y = state[1]
	# z = state[2]
	l = 0.5
	m = 0.5
	g = 1
	c = 0

	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	u_inp = controller_output[-1]

	dxdt = y
	dydt = (g / l) * np.sin(x) + (u_inp - c * y) / (m * l**2)
	# dzdt = 20

	rhs = [dxdt, dydt]

	return rhs

# https://github.com/amaleki2/benchmark_closedloop_verification


def doublePendulum(state, time, plant):
	th1 = state[0]
	th2 = state[1]
	u1 = state[2]
	u2 = state[3]

	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	# u_inp = controller_output
	# print(controller_output, u_inp)
	T1 = controller_output[0]
	T2 = controller_output[1]

	dx1dt = u1
	dx2dt = u2
	dx3dt = 4 * T1 + 2*np.sin(th1) - (u2 * u2 * np.sin(th1 - th2))/2 + (np.cos(th1 - th2)*(np.sin(th1 - th2) * u1 * u1 +
			8 * T2 + 2 * np.sin(th2) - np.cos(th1 - th2) * (- (np.sin(th1 - th2)*u2 * u2)/2 + 4 * T1 + 2 * np.sin(th1))))/(2 * (np.cos(th1 - th2)**2/2 - 1))
	dx4dt = -(np.sin(th1 - th2)*u1**2 + 8 * T2 + 2 * np.sin(th2) - np.cos(th1 - th2) * (- (np.sin(th1 - th2)*u2**2)/2 +
															4 * T1 + 2 * np.sin(th1)))/(np.cos(th1 - th2)**2/2 - 1)

	rhs = [dx1dt, dx2dt, dx3dt, dx4dt]

	return rhs


def oBenchC10(state, time, plant):
	x1 = state[0]
	x2 = state[1]
	x3 = state[2]
	x4 = state[3]

	controller_output = plant.dnn_controllers[0].performForwardPass(state)
	u1 = controller_output[0] - 20
	u2 = controller_output[1] - 20

	dx1dt = x4 * np.cos(x3)
	dx2dt = x4 * np.sin(x3)
	dx3dt = u2
	dx4dt = u1

	rhs = [dx1dt, dx2dt, dx3dt, dx4dt]

	return rhs


def airplane_old(state, time, plant):
	x = state[0]
	y = state[1]
	z = state[2]
	u = state[3]
	v = state[4]
	w = state[5]
	phi = state[6]
	theta = state[7]
	psi = state[8]
	r = state[9]
	p = state[10]
	q = state[11]

	actions = plant.dnn_controller.performForwardPass(state)

	Fx = actions[0]
	Fy = actions[1]
	Fz = actions[2]
	Mx = actions[3]
	My = actions[4]
	Mz = actions[5]

	T_psi = np.array([[np.cos(psi), -np.sin(psi), 0.0], [np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]], dtype=float)
	T_theta = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]], dtype=float)
	T_phi = np.array([[1.0, 0.0, 0.0], [0., np.cos(phi), -np.sin(phi)], [0., np.sin(phi), np.cos(phi)]], dtype=float)

	mat_1 = np.matmul(np.matmul(T_psi, T_theta), T_phi)

	mat_2 = np.array([[np.cos(theta), np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi)],
					[0.0, np.cos(theta) * np.cos(phi), -np.cos(theta) * np.sin(phi)],
					[0.0, np.sin(phi), np.cos(phi)]], dtype=float)
	# mat_2 = np.divide(mat_2, np.cos(theta))
	mat_2 = 1 / np.cos(theta) * mat_2

	# a1 = np.array([[u], [v], [w]], dtype=float)
	# a2 = np.matmul(mat_1, a1)
	a1 = np.array([u, v, w]).T
	a2 = mat_1.dot(a1)

	# a2 = a2.flatten()
	dxdt = a2[0]
	dydt = a2[1]
	dzdt = a2[2]

	# a3 = np.array([[p], [q], [r]], dtype=float)
	# a4 = np.matmul(mat_2, a3)
	a3 = np.array([p, q, r]).T
	a4 = mat_2.dot(a3)

	# a4 = a4.flatten()
	dphidt = a4[0]
	dthetadt = a4[1]
	dpsidt = a4[2]

	dudt = -np.sin(theta) + Fx - q * w + r * v
	dvdt = np.cos(theta) * np.sin(phi) + Fy - r * u + p * w
	dwdt = np.cos(theta) * np.cos(phi) + Fz - p * v + q * u

	dpdt = Mx
	dqdt = My
	drdt = Mz

	rhs = [dxdt, dydt, dzdt, dudt, dvdt, dwdt, dphidt, dthetadt, dpsidt, dpdt, dqdt, drdt]

	return rhs


def airplane(state, time, plant):
	m = 1.
	Ix = 1.
	Iy = 1.
	Iz = 1.
	Ixz = 0.
	g = 1.

	x = state[0]
	y = state[1]
	z = state[2]
	u = state[3]
	v = state[4]
	w = state[5]
	phi = state[6]
	theta = state[7]
	psi = state[8]
	r = state[9]
	p = state[10]
	q = state[11]

	actions = plant.dnn_controllers[0].performForwardPass(state)

	X = actions[0]
	Y = actions[1]
	Z = actions[2]
	L = actions[3]
	M = actions[4]
	N = actions[5]

	T_psi = np.array([[np.cos(psi), -np.sin(psi), 0.0], [np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]], dtype=float)
	T_theta = np.array([[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]], dtype=float)
	T_phi = np.array([[1.0, 0.0, 0.0], [0., np.cos(phi), -np.sin(phi)], [0., np.sin(phi), np.cos(phi)]], dtype=float)

	mat_1 = np.matmul(np.matmul(T_psi, T_theta), T_phi)

	mat_2 = np.array([[np.cos(theta), np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi)],
					[0.0, np.cos(theta) * np.cos(phi), -np.cos(theta) * np.sin(phi)],
					[0.0, np.sin(phi), np.cos(phi)]], dtype=float)
	mat_2 = 1 / np.cos(theta) * mat_2

	a1 = np.array([u, v, w]).T
	a2 = mat_1.dot(a1)

	dxdt = a2[0]
	dydt = a2[1]
	dzdt = a2[2]

	a3 = np.array([p, q, r]).T
	a4 = mat_2.dot(a3)

	dphidt = a4[0]
	dthetadt = a4[1]
	dpsidt = a4[2]

	a5 = np.array([
		[Ix, Ixz],
		[Ixz, Iz]
	])
	a6 = np.array([
		[L - (Iz - Iy) * q * r - Ixz * q * p],
		[N - (Iy - Ix) * q * p + Ixz * q * r]
	])
	a7 = np.linalg.inv(a5).dot(a6)

	dudt = -g * np.sin(theta) + X / m - q * w + r * v
	dvdt = g * np.cos(theta) * np.sin(phi) + Y / m - r * u + p * w
	dwdt = g * np.cos(theta) * np.cos(phi) + Z / m - p * v + q * u

	dpdt = a7[0][0]
	dqdt = 1. / Iy * (M - Ixz * (r ** 2 - p ** 2) - (Ix - Iz) * p * r)
	drdt = a7[1][0]

	rhs = [dxdt, dydt, dzdt, dudt, dvdt, dwdt, dphidt, dthetadt, dpsidt, dpdt, dqdt, drdt]

	return rhs
