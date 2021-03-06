from dolfin import *
import numpy
#import pylab

# Create mesh and define function space
nx = 100
mesh = UnitInterval(nx)
V = FunctionSpace(mesh, 'CG', 1)

# All constants are in SI units
zs = 1500 							# surface elevation (m)
zb = 0.0# base elevation (m)
g = 9.81							# acceleration due to gravity (m/s^2)
spy = 31556926.						# seconds per year (s/a)
rho = 911.							# density of ice (kg/m^3)
Cp = 2009.							# heat capacity of ice (J/(kg*K))
beta = 9.8*10**-8					# pressure dependence of melting point (K/Pa)
Tpmp = beta*rho*g*(zs - zb)			# pressure melting point of ice at bed (C)
k = 2.1								# thermal diffusivity of ice
dzsdx = 0.5*10**-5					# 0.1 - 1 surface slope of ice
dTdx = 3*10**-4						# 1 - 5 horizontal temperature gradient (C/m)
Qgeo = 50*10**-3					# 30 - 70 geothermal heat flow (W/m^2)
sigma = '((x[0] - {zb})/({zs}- {zb}))'.format(zs=zs, zb=zb)# string to rescale vertical coordinate
print sigma
K = k/(rho*Cp)

# horizontal ice velocity
u = Expression('(0*pow('+sigma+', 4))/spy', spy=spy)

w = Expression(('-0.0*'+sigma+'/spy',), spy=spy) 		# -0.1 to -0.5

phi = Expression('0*-rho*g*(zs - x[0])', rho=rho, g=g, zs=zs)

Ts = Expression('-10+5*sin(2*pi*(t/spy))', pi=3.14159, spy=spy, t=0)

g = Expression('(1-'+sigma+')*Qgeo/rho/Cp', rho=rho, Cp=Cp, Qgeo=Qgeo)

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


boundary = Boundary()
bc = DirichletBC(V, Ts, boundary)

# Initial condition
T0 = interpolate(Ts, V)
#u_1 = project(u0, V)  # will not result in exact solution!

dt = 0.1*spy      # time step

# Define variational problem
T = TrialFunction(V)
v = TestFunction(V)
a = (T * v + K * dt*dot(nabla_grad(T), nabla_grad(v)))*dx
#a = (T * v + K * v * dt*dot(nabla_grad(T), nabla_grad(v)) + dot(w, nabla_grad(T))*v*dt)*dx
L = (phi/rho/Cp * dt - u*dTdx * dt + T0)*v*dx + g * dt * v * ds

A = assemble(a)   # assemble only once, before the time stepping
b = None          # necessary for memory saving assemeble call

# Compute solution
T = Function(V)   # the unknown at a new time level
totalTime = 100. * spy       # total simulation time
t = dt
out_file = File("results/temp.pvd")
while t <= totalTime:
    print 'time =', t/spy
    b = assemble(L, tensor=b)
    Ts.t = t
    bc.apply(A, b)
    solve(A, T.vector(), b)

    #pylab.plot(T.vector().array())
    plot(T)
    print T.vector().array()
    out_file << (T, t)
    # Verify
    u_e = interpolate(Ts, V)
    maxdiff = numpy.abs(u_e.vector().array() - T.vector().array()).max()
    print 'Max error, t=%.2f: %-10.3f' % (t, maxdiff)

    t += dt
    T0.assign(T)
