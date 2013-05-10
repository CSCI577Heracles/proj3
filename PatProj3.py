"""
Project III Temperature flow in ice
Started from:
FEniCS tutorial demo program: Diffusion equation with Dirichlet
conditions and a solution that will be exact at all nodes.
"""
from __future__ import division
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.pyplot as pl

spy = 31557600.0  # sec/year = 60*60*24*365.25
dt = 1e2 * spy
total_sim_time = 1e3*dt#1e-2 * spy
ice_thickness = 1500.0
temperature_0 = -10.0
theta0_multiplier = 5.0
u_mult = 20.0  # TODO np.linspace(20, 100, 10)  # h_coeff
w_mult = 0.5  # TODO np.linspace(0.1, 0.5, 10)  # v_coeff
d_theta_dx = 1.01e-4  # TODO np.linspace(1, 5, 10)*10**-4 # deg C/m, horizontal temperature gradient
surface_slope = 0.1e-5  # TODO np.linspace(0.1, 1, 10)*10**-5  # dzs_dx
Qgeo = 42e-3  # TODO np.linspace(30, 70, 10)*10**-3, W/m^2, Geothermal heat flow
z_b = 0.0  # bottom
z_s = 1500.0  # surface
glo_warm = 3000. # time when global warming begins
mesh_node_count = 100

k = 2.1  # W/(m*K), thermal diffusivity of ice
rho = 600.0  # kg/m^3, density of firn
rho = 911.0  # kg/m^3, Density of ice
Cp = 2009.0  # J/(kg*K), Heat capacity
g = 9.81  # m/s^2 accel due to gravity

beta = 9.8e-8  # K/Pa, Pressure dependence of melting point

# Create mesh and define function space
mesh = IntervalMesh(mesh_node_count, z_b, z_s)
# mesh.coordinates[:] *= ice_thickness
z = mesh.coordinates()[:]

print 'z: ', z
        
def prints(strName, t):
    plt.clf()
    plt.xlim((-15, 5))
    plt.ylim((z_b, z_s))
    plt.plot(theta.vector().array(), z_bar, color = 'blue', label = 'Firn Column Temperature')
    plt.plot(theta_pmp, z_bar, color = 'red', label = 'Pressure Melting Point')
    plt.xlabel('Temperature (C)')
    plt.ylabel('Distance (m)')
    plt.title('Temperature Distribution After %d Years' % t)
    plt.legend(loc = 3)
    plt.savefig(strName)
    return
    
# inputs: 
#       x:  The array of nodes of the mesh, e.g. x = mesh.coordinates()[:]
#       nx: Number of nodes in the mesh
#       ps: Percentage of nodes to place at the surface, value on the interval [0., 1.]
#       pb: Percentage of nodes to place at the base, value on the interval [0., 1.]
#       ds: Distance from the surface to place the ps nodes, value on the interval [0., s-b]
#       db: Distance from the bed to place the pb nodes, value on the interval [0., s-b]
#
# Example:
#       Suppose we have 100 nodes, and our ice sheet is 1000m tall.
#       We wish to have 40 nodes in the top ds = 200m, and 30 nodes in the bottom db = 300m.
#       ps = 40 / 100 = 0.4
#       pb = 30 / 100 = 0.3
#       These parameters will set 40 nodes in a linspace between the surface and 200m below the 
#       surface, 30 nodes in a linspace between the base and 300m above the base, and places
#       the remaining 30 nodes in the space between.
def denser(x, nx, ps = 0.4, pb = 0.1, ds = 150, db = 200):
    b = x[0]
    s = x[-1]
    ns = ps*nx
    nb = pb*nx
    
    if ds + db >= s - b:
        print 'ds + db must be less than s - b!'
        return x
    if ps + pb > 1:
        print 'ps + pb must be less than 1!'
        return x
    
    print 'NUMBER: ', ds/ns
    xs = np.linspace(s-ds, s, ns + 1)
    xb = np.linspace(b, db, nb + 1)
    xm = np.linspace(db, s-ds, nx-nb-ns + 1)
    
    return np.concatenate([xb, xm[1:-1], xs])
    
z_bar = denser(z, mesh_node_count)
z_bar_coor = np.array([z_bar]).transpose()

print 'z_bar_coor: ', z_bar_coor

theta_pmp = beta * rho * g * (z_bar - z_s)  # deg C, Pressure melting point of ice at z
print theta_pmp

mesh.coordinates()[:] = z_bar_coor

func_space = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
theta_0 = Expression('t0 + m*sin(2.0*pi*(t/spy))',  # sin and pi are keywords
                     t0=temperature_0, m=theta0_multiplier, t=0, spy=spy)


def boundary(x, on_boundary):
    epsilon = 1e-14
    return on_boundary and abs(x[0] - z_s) < epsilon

dirichlet_bc = DirichletBC(func_space, theta_0, boundary)

# Initial condition
theta_1 = interpolate(theta_0, func_space) 
#theta_1 = project(theta_0, func_space)  # will not result in exact solution!

# Define variational problem
theta = TrialFunction(func_space)
v = TestFunction(func_space)

sigma = '((x[0] - {z_b})/({z_s} - {z_b}))'.format(
        z_s=z_s, z_b=z_b)
print 'sigma:', sigma
u_prime = '-4*{m}*pow({s}, 3)/({zs}-{zb})/{spy}'.format(
          s=sigma, m=u_mult, zs=z_s, zb=z_b, spy=spy)
print 'u_prime:', u_prime
u = Expression('m * pow({s}, 4)/spy'.format(s=sigma),
               m=u_mult, spy=spy)  # m/annum, horiz ice velocity
w_str = '-m*'+sigma+'/spy'
#if dolfin.__version__ > '1.0.0':
#    w_str = tuple(w_str,)
w = Expression((w_str,), m=w_mult, spy=spy)  # m/annum, vertical ice velocity (needs to be a tuple to work with `inner`)
phi = Expression('-p*g*(zs-x[0])*{du_dz}*dzs_dx'.format(du_dz=u_prime),
                 p=rho, g=g, zs=z_s, dzs_dx=surface_slope)  # W/m^3, heat sources from deformation of ice

bed_boundary = Expression('1 - {s}'.format(s=sigma))
neumann_bc = Qgeo/rho/Cp * dt * bed_boundary * v * ds
# neumann_bc = g * dt * v * ds  # Will's code

diffusion_term = theta*v + \
    (k/rho/Cp) * inner(nabla_grad(theta), nabla_grad(v))*dt
# Either of the following seem to work
# NOTE: Seems that we shouldn't negate w here
vertical_advection = inner(w, nabla_grad(theta)) * v * dt
# vertical_advection = w * nabla_grad(theta) * v * dt
horiz_advection = -u * d_theta_dx * v * dt
strain_heating = phi/rho/Cp * Expression(u_prime) * v * dt

a = (diffusion_term + vertical_advection)*dx
#a = (diffusion_term)*dx
# L = (theta_1 - u*d_theta_dx*dt + phi/rho/Cp * dt)*v*dx + neumann_bc
L = (theta_1*v + horiz_advection + strain_heating)*dx + neumann_bc
#L = (theta_1*v)*dx + neumann_bc

A = assemble(a)   # assemble only once, before the time stepping
b = None          # necessary for memory saving assemeble call
# Compute solution
theta = Function(func_space)   # the unknown at a new time level
t = 0
out_file = File("results/theta.pvd")
plt.figure(1)
plt.clf()
plt.ion()
plt.xlim((-20, 0))
plt.ylim((z_b, z_s))
plt.grid()
plt.xlabel('Temperature (C)')
plt.ylabel('Distance (m)')
plt.legend()
ax = plt.gca()
plt.show()
state = 0
cDepth = 0.6*len(z_bar)
minTemp = theta.vector().array()[cDepth:]
maxTemp = theta.vector().array()[cDepth:] - 20
while True:
    theta_prev = theta.vector().array()
    #print 'time = {t:.1f} years'.format(t=t / spy)
    b = assemble(L, tensor=b)
    theta_0.t = t
    dirichlet_bc.apply(A, b)
    
    solve(A, theta.vector(), b)
    
    temp = theta.vector().array()
    temp[temp > theta_pmp] = theta_pmp[temp > theta_pmp]
    theta.vector()[:] = temp

    #print '\n----------------------------------------'
    #print theta.vector().array()
    #print '----------------------------------------\n'
    #print 'z_coords: ', z_coords
    #print 'theta: ', theta.vector().array()
    plt.cla()
    plt.xlim((-15, 5))
    plt.ylim((z_b, z_s))
    plt.plot(theta.vector().array(), z_bar)
    plt.title('Temperature Distribution After %d Years' % (t/spy))
    plt.plot(theta_pmp, z_bar, color = 'red')
    #plt.grid(which = 'major', axis = 'both')
    #plt.legend(loc = 0)
    plt.draw()
    # plot(theta)  # transpose of the plot we want
    #out_file << (theta, t, )

    # Verify
    #u_e = interpolate(theta_0, func_space)
    #maxdiff = np.abs(u_e.vector().array() - theta.vector().array()).max()
    #print 'Max error, t=%.2f: %-10.3f' % (t, maxdiff)
    
    #err = max(abs((theta_prev - theta.vector().array())/theta.vector().array()))
    #print err
    #if err < 0.001:
    #    break
    
    # TODO: part 1
    #for i in range(theta.vector().array()[cDepth:].size):
    #   if minTemp[i] > theta.vector().array()[cDepth + i]:
    #        minTemp[i] = theta.vector().array()[cDepth + i]
    #    if maxTemp[i] < theta.vector().array()[cDepth + i]:
    #        maxTemp[i] = theta.vector().array()[cDepth + i]
    
    # TODO: part 2
    err = max(abs((theta_prev - theta.vector().array())/theta.vector().array()))
    print 'err = ', err
    #if state == 1:
    #    plt.clf()
    #    plt.xlim((-15, 5))
    #    plt.ylim((z_b, z_s))
    #    plt.plot(theta.vector().array(), z_bar, color = 'blue', label = 'Ice Column Temperature')
    #    plt.plot(theta_pmp, z_bar, color = 'red', label = 'Pressure Melting Point')
    #    plt.xlabel('Temperature (C)')
    #    plt.ylabel('Distance (m)')
    #    plt.title('Temperature Distribution After %d Years' % (t/spy))
    #    plt.legend(loc = 3)
    #    plt.savefig('TempGWInit2.png')
    #    state = 2
        
    if err < 0.001:
    #    plt.clf()
    #    plt.xlim((-15, 5))
    #    plt.ylim((z_b, z_s))
    #    plt.plot(theta.vector().array(), z_bar, color = 'blue', label = 'Ice Column Temperature')
    #    plt.plot(theta_pmp, z_bar, color = 'red', label = 'Pressure Melting Point')
    #    plt.xlabel('Temperature (C)')
    #    plt.ylabel('Distance (m)')
    #    plt.title('Temperature Distribution After %d Years' % (t/spy))
    #    plt.legend(loc = 3)
    #    plt.savefig('TempStable2.png')
        theta_0 = Constant(0.)
        dirichlet_bc = DirichletBC(func_space, theta_0, boundary)
    #    state = 1
        
    #if theta.vector().array()[0] == theta_pmp[0]:
    #    plt.clf()
    #    plt.xlim((-15, 5))
    #    plt.ylim((z_b, z_s))
    #    plt.plot(theta.vector().array(), z_bar, color = 'blue', label = 'Ice Column Temperature')
    #    plt.plot(theta_pmp, z_bar, color = 'red', label = 'Pressure Melting Point')
    #    plt.xlabel('Temperature (C)')
    #    plt.ylabel('Distance (m)')
    #    plt.title('Temperature Distribution After %d Years' % (t/spy))
    #    plt.legend(loc = 3)
    #    plt.savefig('TempGWFin2.png')
    #    state = 3
    #    break
    
    t += dt
    theta_1.assign(theta)

#dif = maxTemp - minTemp
#print dif

#plt.title('Seasonal Temperature')
#plt.savefig('TempSeason2.png')

#plt.plot(theta.vector().array(), z_bar, color = 'blue')
#plt.xlabel('Temperature (C)')
#plt.ylabel('Distance (m)')
#plt.title('Temperature Distribution After %d Years' % (t/spy))
#plt.savefig('TempDiff.png')

