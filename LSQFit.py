import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

xmin=1.0
xmax=20.0
npoints=12
sigma=0.2
lx=np.zeros(npoints)
ly=np.zeros(npoints)
ley=np.zeros(npoints)
pars=[0.5,1.3,0.5]

from math import log
def f(x,par):
    return par[0]+par[1]*log(x)+par[2]*log(x)*log(x)

from random import gauss
def getX(x):  # x = array-like
    step=(xmax-xmin)/npoints
    for i in range(npoints):
        x[i]=xmin+i*step
        
def getY(x,y,ey):  # x,y,ey = array-like
    for i in range(npoints):
        y[i]=f(x[i],pars)+gauss(0,sigma)
        ey[i]=sigma

# get a random sampling of the (x,y) data points, rerun to generate different data sets for the plot below
getX(lx)
getY(lx,ly,ley)

fig, ax = plt.subplots()
ax.errorbar(lx, ly, yerr=ley)
ax.set_title("Pseudoexperiment")
fig.show()


# *** modify and add your code here ***
nexperiments = 1000  # for example
nPar = 3
ndf = npoints-nPar
par_a = np.zeros(nexperiments)
par_b = np.zeros(nexperiments)
par_c = np.zeros(nexperiments)
chi2_raw = np.zeros(nexperiments)
chi2_reduced = np.zeros(nexperiments)


# perform many least squares fits on different pseudo experiments here
for i in range(nexperiments):
    getX(lx)
    getY(lx, ly, ley)

    weights = 1.0/ley

    A = np.zeros((npoints, nPar))
    log_lx = np.log(lx)
    A[:, 0] = 1.0
    A[:, 1] = log_lx
    A[:, 2] = log_lx**2

    A_w = A*weights[:, np.newaxis]
    y_w = (ly*weights).reshape(npoints, 1)

    AT_A_inv = inv(A_w.T @ A_w)
    AT_y = A_w.T @ y_w

    theta = AT_A_inv @ AT_y

    par_a[i] = theta[0,0]
    par_b[i] = theta[1,0]
    par_c[i] = theta[2,0]

    residuals_w = y_w - (A_w @ theta)
    chi2 = np.sum(residuals_w**2)
    chi2_raw[i] = chi2
    chi2_reduced[i] = chi2/ndf
# fill histograms w/ required data

# par_a = np.random.rand(1000)   # simple placeholders for making the plot example
# par_b = np.random.rand(1000)   # these need to be filled using results from your fits
# par_c = np.random.rand(1000)
# chi2_reduced = np.random.rand(1000)

summary_string = f"""
True values: a={pars[0]}, b={pars[1]}, c={pars[2]}

Fitted 'a': mean = {np.mean(par_a):.4f}, std = {np.std(par_a):.4f}
Fitted 'b': mean = {np.mean(par_b):.4f}, std = {np.std(par_b):.4f}
Fitted 'c': mean = {np.mean(par_c):.4f}, std = {np.std(par_c):.4f}
Degrees of Freedom (ndf): {ndf}

Chi^2:
- Mean: {np.mean(chi2_raw):.4f}
- Std Dev: {np.std(chi2_raw):.4f}
- Expected Mean (ndf): {ndf}
- Expected Std Dev (sqrt(2*ndf)): {np.sqrt(2*ndf):.4f}

Reduced Chi^2:
- Mean: {np.mean(chi2_reduced):.4f}
- Std Dev: {np.std(chi2_reduced):.4f}
- Expected Mean: 1.0

Note:
I observed that increasing the number of points decreases the uncertainty of the parameters,
while decreasing it increases the uncertainty.
"""
print(summary_string)

fig_text = plt.figure(figsize=(10, 8))
fig_text.text(0.05, 0.95, summary_string, ha='left', va='top', family='monospace', fontsize=10)
plt.axis('off')

fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))
bins = 60

axs1[0, 0].hist(par_a, bins=bins)
axs1[0, 0].set_title('Distribution of Parameter a')
axs1[0, 0].set_xlabel('Parameter a')
axs1[0, 0].set_ylabel('Frequency')

axs1[0, 1].hist(par_b, bins=bins)
axs1[0, 1].set_title('Distribution of Parameter b')
axs1[0, 1].set_xlabel('Parameter b')
axs1[0, 1].set_ylabel('Frequency')

axs1[1, 0].hist(par_c, bins=bins)
axs1[1, 0].set_title('Distribution of Parameter c')
axs1[1, 0].set_xlabel('Parameter c')
axs1[1, 0].set_ylabel('Frequency')

axs1[1, 1].hist(chi2_raw, bins=bins)
axs1[1, 1].set_title('Chi^2 distribution')
axs1[1, 1].set_xlabel('Chi^2')
axs1[1, 1].set_ylabel('Frequency')

fig1.tight_layout(pad=3.0)

fig, axs = plt.subplots(2, 2)
plt.tight_layout()

# careful, the automated binning may not be optimal for displaying your results!
axs[0, 0].hist2d(par_a, par_b)
axs[0, 0].set_title('Parameter b vs a')

axs[0, 1].hist2d(par_a, par_c)
axs[0, 1].set_title('Parameter c vs a')

axs[1, 0].hist2d(par_b, par_c)
axs[1, 0].set_title('Parameter c vs b')

axs[1, 1].hist(chi2_reduced)
axs[1, 1].set_title('Reduce chi^2 distribution')

with PdfPages('LSQFit.pdf') as pdf:
    pdf.savefig(fig_text)
    pdf.savefig(fig1)
    pdf.savefig(fig)
fig1.show()
fig.show()

# **************************************
  

input("hit Enter to exit")
