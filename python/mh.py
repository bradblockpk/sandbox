### Metropolis-Hastings sampler with self-tuning proposal distribution
### Brad Block
### 4/9/2012

from collections import deque
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import uniform

# target distribution from which to sample
target_alpha = 2
target_beta = 2
target_pdf = lambda x: beta.pdf(x, target_alpha, target_beta)

# sampler parameters
burnin_steps = 1000
steps_per_sample = 10
total_samples = 1000
target_acceptance_rate = 0.5
acceptance_adjustment_rate = 0.01
acceptance_history_sample_count = 50

# gaussian proposal distribution with initial std. deviation estimate
proposal_stddev = 1
proposal_sample = lambda z: norm.rvs(z, proposal_stddev)
proposal_cpdf = lambda z_star, z: norm.pdf(z_star, z, proposal_stddev)

# sampler storage
z = 1
samples = []
acceptance_history = deque(maxlen=acceptance_history_sample_count)
total_sampler_steps = burnin_steps + steps_per_sample * total_samples

def sample(z):
    # obtain new sample from conditional proposal distribution
    z_star = proposal_sample(z)
    a = target_pdf(z_star) / target_pdf(z)
    b = proposal_cpdf(z, z_star) / proposal_cpdf(z_star, z)
    # accept or reject new sample
    if uniform.rvs() < min(1, a * b):
        accepted = 1
        z = z_star
    else:
        accepted = 0
    return (accepted, z)

# main sampler loop
for i in xrange(1, total_sampler_steps):
    # obtain next candidate sample and evaluate acceptance
    (accepted, z) = sample(z)
    # adjust proposal std. deviation if still in burnin phase
    # otherwise adjust proposal std. deviation if still in burnin phase
    if i <= burnin_steps:
        acceptance_history.append(accepted)
        acceptance_rate = float(sum(acceptance_history)) / acceptance_history_sample_count
        proposal_stddev *= 1 + (acceptance_rate - target_acceptance_rate) * acceptance_adjustment_rate
    elif (i - burnin_steps + 1) % steps_per_sample == 0:
        samples.append(z)

