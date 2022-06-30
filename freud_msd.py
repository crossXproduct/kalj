import freud
import gsd.hoomd
import scipy
import numpy as np
import sys
import matplotlib.pyplot as plt

path = str(sys.argv[1])
t_write = int(sys.argv[2])
dt = float(sys.argv[3])
traj = gsd.hoomd.open(path,'rb')

print('frames: ',len(traj))
print('particles: ',len(traj[0].particles.position))
print('first snapshot: ',traj[0].particles.position[0:10])

print(traj[0].configuration.box,traj[1].configuration.box)
msd = freud.msd.MSD(traj[0].configuration.box)

#create 3D array of particle positions over whole trajectory (N_frames,N_particles,3)
#positions = np.empty([len(traj),traj[0].particles.N,3])
positionslist = list()
positionslist.append(traj[0].particles.position[:])
for i in range(1,len(traj)-1):
    positionslist.append(traj[i].particles.position[:])

positions = np.array(positionslist)
msd.compute(positions,reset=False)

print('msd size',len(msd.msd))
print('msd[0:10]',msd.msd[0:11])
print('msd[last 10]',msd.msd[len(msd.msd)-11:len(msd.msd)-1])

traj.close()

msdfile = open('msd.dat','w')
for i in range(0,len(msd.msd)):
    msdfile.write("%4d,%7.5f\n"%(i*t_write*dt,msd.msd[i]))
plt.plot(range(0,len(msd.msd)*i*dt,dt),msd.msd)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('frames')
plt.xlim(left=0.1)
plt.ylim(bottom=1)
plt.ylabel('msd')
plt.savefig('msd.png')
