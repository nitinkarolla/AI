
#importing packages
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


p = 0.2
n= 10
def generate_environment(n, p):
    out = np.array([list(np.random.binomial(1, 1-p,n)) for i in range(n)])
    out[0,0] = 1
    out[n-1,n-1] = 1
    return(out)

#print(generate_environment(n,p))

test = generate_environment(n,p)

plt.imsave('test.png', test , cmap = 'Greys')

