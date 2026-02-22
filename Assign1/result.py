import numpy as np
import matplotlib.pyplot as plt

# m scaling
m_data=np.loadtxt("result_m.txt", comments="#")
m=m_data[:,0]
time_m=m_data[:,2]

plt.figure()
plt.plot(m, time_m, 'o-')
plt.xlabel("m (number of rows)")
plt.ylabel("Time (seconds)")
plt.title("TSQR Scaling with respect to m (n=50, p=4)")
plt.grid()
plt.savefig("scaling_m.png")

# n scaling
n_data=np.loadtxt("result_n.txt", comments="#")
n=n_data[:,1]
time_n=n_data[:,2]

plt.figure()
plt.plot(n, time_n, 'o-')
plt.xlabel("n (number of columns)")
plt.ylabel("Time (seconds)")
plt.title("TSQR Scaling with respect to n (m=160000, p=4)")
plt.grid()
plt.savefig("scaling_n.png")

# log-log plot
plt.figure()
plt.loglog(n, time_n, 'o-')
plt.xlabel("n (log scale)")
plt.ylabel("Time (log scale)")
plt.title("TSQR n-scaling (log-log)")
plt.grid()
plt.savefig("scaling_n_loglog.png")

plt.show()