import itertools
import numpy
import math
import hoomd
import gsd.hoomd
#import matplotlib

#matplotlib.style.use('ggplot')

###INITIALIZATION
m = 4
N_particles = 4 * m**3

spacing = 1.3
K = math.ceil(N_particles**(1 / 3))
L = K * spacing
x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))

snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = N_particles
snapshot.particles.position = position[0:N_particles]
snapshot.particles.typeid = [0] * N_particles
snapshot.configuration.box = [L, L, L, 0, 0, 0]

snapshot.particles.types = ['A']

with gsd.hoomd.open(name='lattice.gsd', mode='xb') as f:
    f.append(snapshot)

cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=1)
sim.create_state_from_gsd(filename='lattice.gsd')

integrator = hoomd.md.Integrator(dt=0.005)
cell = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
lj.r_cut[('A', 'A')] = 2.5
integrator.forces.append(lj)
nvt = hoomd.md.methods.NVT(kT=1.5, filter=hoomd.filter.All(), tau=1.0)
integrator.methods.append(nvt)

sim.operations.integrator = integrator

snapshot = sim.state.get_snapshot()
print(snapshot.particles.velocity[0:5])

sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)

thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
    filter=hoomd.filter.All())

sim.operations.computes.append(thermodynamic_properties)
sim.run(0)


print(thermodynamic_properties.degrees_of_freedom)
1 / 2 * 1.5 * thermodynamic_properties.degrees_of_freedom
print(thermodynamic_properties.kinetic_temperature)

sim.run(10000)

print(sim.state.get_snapshot().particles.velocity[0:5])
print(thermodynamic_properties.kinetic_energy)

hoomd.write.GSD.write(state=sim.state, filename='random.gsd', mode='xb')


###COMPRESSION

cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu)
sim.create_state_from_gsd(filename='random.gsd')

integrator = hoomd.md.Integrator(dt=0.005)
cell = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
lj.r_cut[('A', 'A')] = 2.5
integrator.forces.append(lj)
nvt = hoomd.md.methods.NVT(kT=1.5, filter=hoomd.filter.All(), tau=1.0)
integrator.methods.append(nvt)
sim.operations.integrator = integrator

print(sim.timestep)
ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=20000)

steps = range(0, 40000, 20)
y = [ramp(step) for step in steps]

"""
fig = matplotlib.figure.Figure(figsize=(10, 6.18))
ax = fig.add_subplot()
ax.plot(steps, y)
ax.set_xlabel('timestep')
ax.set_ylabel('ramp')
fig
"""

rho = sim.state.N_particles / sim.state.box.volume
print(rho)

initial_box = sim.state.box
final_box = hoomd.Box.from_box(initial_box)  # make a copy of initial_box
final_rho = 1.2
final_box.volume = sim.state.N_particles / final_rho

box_resize_trigger = hoomd.trigger.Periodic(10)
box_resize = hoomd.update.BoxResize(box1=initial_box,
                                    box2=final_box,
                                    variant=ramp,
                                    trigger=box_resize_trigger)
sim.operations.updaters.append(box_resize)
sim.run(10001)
ramp(sim.timestep - 1)
current_box = sim.state.box
(current_box.Lx - initial_box.Lx) / (final_box.Lx - initial_box.Lx)
sim.run(10000)
print(sim.state.box == final_box)
sim.operations.updaters.remove(box_resize)


###EQUILIBRATION
sim.run(500)