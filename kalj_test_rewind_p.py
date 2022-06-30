import itertools
import numpy
import math
import hoomd
import gsd.hoomd
import sys
import random

N_particles = int(sys.argv[1])
temp = float(sys.argv[2])
delta_t = float(sys.argv[3])
t_eq = int(sys.argv[4])
t_run = int(sys.argv[5])
t_write = int(sys.argv[6]) #THIS NEEDS TO STAY THE SAME FOR ALL RUNS
#final_rho = float(sys.argv[3])

cpu = hoomd.device.CPU()
if cpu.communicator.rank == 0:
    print(N_particles)
    print(temp)
    print(delta_t)
    print(t_run)
    print(t_write)

random_seed = int(random.randrange(0,65535))

###SYSTEM SETUP
spacing = 1.3
K = math.ceil(N_particles**(1 / 3))
L = K * spacing
x = numpy.linspace(-L / 2, L / 2, K, endpoint=False)
position = list(itertools.product(x, repeat=3))

snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = N_particles
snapshot.particles.position = position[0:N_particles]
snapshot.particles.typeid = [0] * math.floor(0.8*N_particles) + [1] * math.floor(0.2*N_particles) #particle 'B' is 20%
snapshot.configuration.box = [L, L, L, 0, 0, 0]
snapshot.particles.types = ['A','B']

if cpu.communicator.rank == 0:
    with gsd.hoomd.open(name='lattice.gsd', mode='xb') as f:
        f.append(snapshot)

###RANDOMIZE
## Initialize state
cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=random_seed)
sim.create_state_from_gsd(filename='lattice.gsd')

## Setup LJ Integrator
integrator = hoomd.md.Integrator(dt=0.005)
cell = hoomd.md.nlist.Cell(buffer=0.1)
lj = hoomd.md.pair.LJ(nlist=cell)
#   Define pair potential
epsilon_AA = 1.0
sigma_AA = 1.0
lj.params[('A', 'A')] = dict(epsilon=epsilon_AA, sigma=sigma_AA)
lj.r_cut[('A', 'A')] = 2.5*sigma_AA
if cpu.communicator.rank == 0: print("r_cut_AA=",lj.r_cut[('A', 'A')])
epsilon_AB = 1.5
sigma_AB = 0.8
lj.params[('A', 'B')] = dict(epsilon=epsilon_AB, sigma=sigma_AB)
lj.r_cut[('A', 'B')] = 2.5*sigma_AB
epsilon_BB = 0.5
sigma_BB = 0.88
lj.params[('B', 'B')] = dict(epsilon=epsilon_BB, sigma=epsilon_BB)
lj.r_cut[('B', 'B')] = 2.5*sigma_BB
r_buf = 0.3*max(lj.r_cut[('A', 'A')],lj.r_cut[('A', 'B')],lj.r_cut[('B', 'B')])
cell.buffer = r_buf
#   Assign force to integrator and integrator to simulation
integrator.forces.append(lj)
nvt = hoomd.md.methods.NVT(kT=temp, filter=hoomd.filter.All(), tau=100*delta_t)
integrator.methods.append(nvt)
sim.operations.integrator = integrator
snapshot = sim.state.get_snapshot()

## Randomize velocities
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=temp)
#   Prints
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)
sim.run(0)
#if cpu.communicator.rank == 0: print("RANDOMIZED")
if cpu.communicator.rank == 0: print(sim.state.get_snapshot().particles.velocity[0:5])
if cpu.communicator.rank == 0: print("DOF=",thermodynamic_properties.degrees_of_freedom)
if cpu.communicator.rank == 0: print("Particles=",thermodynamic_properties.num_particles)
if cpu.communicator.rank == 0: print("KE=",thermodynamic_properties.kinetic_energy)
if cpu.communicator.rank == 0: print("kT=",thermodynamic_properties.kinetic_temperature)
if cpu.communicator.rank == 0: print("V=",thermodynamic_properties.volume**(1.0/3.0))

## Randomize positions
sim.run(100000)

## Write randomized state to file
hoomd.write.GSD.write(state=sim.state, filename='random.gsd', mode='xb')

###COMPRESS
## Initialize state
cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=random_seed)
sim.create_state_from_gsd(filename='lattice.gsd')

## Setup LJ Integrator
integrator = hoomd.md.Integrator(dt=0.005)
cell = hoomd.md.nlist.Cell(buffer=r_buf)
lj = hoomd.md.pair.LJ(nlist=cell)
#   Define pair potential
epsilon_AA = 1.0
sigma_AA = 1.0
lj.params[('A', 'A')] = dict(epsilon=epsilon_AA, sigma=sigma_AA)
lj.r_cut[('A', 'A')] = 2.5*sigma_AA
epsilon_AB = 1.5
sigma_AB = 0.8
lj.params[('A', 'B')] = dict(epsilon=epsilon_AB, sigma=sigma_AB)
lj.r_cut[('A', 'B')] = 2.5*sigma_AB
epsilon_BB = 0.5
sigma_BB = 0.88
lj.params[('B', 'B')] = dict(epsilon=epsilon_BB, sigma=epsilon_BB)
lj.r_cut[('B', 'B')] = 2.5*sigma_BB
#   Assign force to integrator and integrator to simulation
integrator.forces.append(lj)
nvt = hoomd.md.methods.NVT(kT=temp, filter=hoomd.filter.All(), tau=100*delta_t)
integrator.methods.append(nvt)
sim.operations.integrator = integrator

## Setup compressor
if cpu.communicator.rank == 0: print(sim.timestep)
ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=10000)
initial_rho = sim.state.N_particles / sim.state.box.volume
if cpu.communicator.rank == 0: print("rho_i=",initial_rho)
initial_box = sim.state.box
if cpu.communicator.rank == 0: print("V_i=",initial_box.volume**(1.0/3.0))
final_box = hoomd.Box.from_box(initial_box)  # make a copy of initial_box
final_rho = 1.2 #same as Flenner & Co
if cpu.communicator.rank == 0: print("rho_f=",final_rho)
final_box.volume = sim.state.N_particles / final_rho
if cpu.communicator.rank == 0: print("V_f=",final_box.volume**(1.0/3.0)) #Kob & Andersen's is 9.4 in units of sigma_AA
t_r = int(10e4*final_rho/initial_rho)
ramp.t_ramp = t_r
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)
box_resize_trigger = hoomd.trigger.Periodic(10)
box_resize = hoomd.update.BoxResize(box1=initial_box, box2=final_box, variant=ramp, trigger=box_resize_trigger)
sim.operations.updaters.append(box_resize)

## Run compressor
sim.run(t_r)
if cpu.communicator.rank == 0: print("COMPRESSED")
if cpu.communicator.rank == 0: print("Compression successful?",sim.state.box == final_box) #check compression success
if cpu.communicator.rank == 0: print(initial_box)
if cpu.communicator.rank == 0: print(final_box)
if cpu.communicator.rank == 0: print(sim.state.box)
#sim.operations.updaters.remove(box_resize) #remove compressor, don't need it anymore
if cpu.communicator.rank == 0: print("DOF=",thermodynamic_properties.degrees_of_freedom)
if cpu.communicator.rank == 0: print("Particles=",thermodynamic_properties.num_particles)
if cpu.communicator.rank == 0: print("KE=",thermodynamic_properties.kinetic_energy)
if cpu.communicator.rank == 0: print("kT=",thermodynamic_properties.kinetic_temperature)
if cpu.communicator.rank == 0: print("V_therm=",thermodynamic_properties.volume**(1.0/3.0))
if cpu.communicator.rank == 0: print("V_box=",sim.state.box.volume**(1.0/3.0))

###EQUILIBRATION
sim.run(100*t_eq)
hoomd.write.GSD.write(state=sim.state, filename='equilibrated.gsd', mode='xb')
#   Prints
if cpu.communicator.rank == 0: print("EQUILIBRATED")
if cpu.communicator.rank == 0: print(sim.state.get_snapshot().particles.velocity[0:5])
if cpu.communicator.rank == 0: print("DOF=",thermodynamic_properties.degrees_of_freedom)
if cpu.communicator.rank == 0: print("Particles=",thermodynamic_properties.num_particles)
if cpu.communicator.rank == 0: print("KE=",thermodynamic_properties.kinetic_energy)
if cpu.communicator.rank == 0: print("kT=",thermodynamic_properties.kinetic_temperature)
if cpu.communicator.rank == 0: print("V=",thermodynamic_properties.volume**(1.0/3.0))

###PRODUCTION
cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=random_seed)
sim.create_state_from_gsd(filename='equilibrated.gsd')

## Setup LJ Integrator
integrator = hoomd.md.Integrator(dt=delta_t)
cell = hoomd.md.nlist.Cell(buffer=r_buf)
lj = hoomd.md.pair.LJ(nlist=cell)
#   Define pair potential
epsilon_AA = 1.0
sigma_AA = 1.0
lj.params[('A', 'A')] = dict(epsilon=epsilon_AA, sigma=sigma_AA)
lj.r_cut[('A', 'A')] = 2.5*sigma_AA
epsilon_AB = 1.5
sigma_AB = 0.8
lj.params[('A', 'B')] = dict(epsilon=epsilon_AB, sigma=sigma_AB)
lj.r_cut[('A', 'B')] = 2.5*sigma_AB
epsilon_BB = 0.5
sigma_BB = 0.88
lj.params[('B', 'B')] = dict(epsilon=epsilon_BB, sigma=epsilon_BB)
lj.r_cut[('B', 'B')] = 2.5*sigma_BB
#   Assign force to integrator and integrator to simulation
integrator.forces.append(lj)
nvt = hoomd.md.methods.NVT(kT=temp, filter=hoomd.filter.All(), tau=100*delta_t)
integrator.methods.append(nvt)
sim.operations.integrator = integrator

## Setup writer
#traj_writer = hoomd.write.DCD(filename='trajectory.dcd', trigger=hoomd.trigger.Periodic(t_write), unwrap_full=True)
traj_writer = hoomd.write.GSD(filename='trajectory.gsd', trigger=hoomd.trigger.Periodic(t_write))
sim.operations.writers.append(traj_writer)

thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)

## Run
sim.run(100*t_run)
if cpu.communicator.rank == 0: print("DONE")
if cpu.communicator.rank == 0: print(sim.state.get_snapshot().particles.velocity[0:5])
if cpu.communicator.rank == 0: print("DOF=",thermodynamic_properties.degrees_of_freedom)
if cpu.communicator.rank == 0: print("Particles=",thermodynamic_properties.num_particles)
if cpu.communicator.rank == 0: print("KE=",thermodynamic_properties.kinetic_energy)
if cpu.communicator.rank == 0: print("kT=",thermodynamic_properties.kinetic_temperature)
if cpu.communicator.rank == 0: print("V=",thermodynamic_properties.volume**(1.0/3.0))