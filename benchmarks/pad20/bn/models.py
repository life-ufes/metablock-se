
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch
from pyro.infer import config_enumerate
from pyro.infer import TraceEnum_ELBO

"""
A Pyro‐based Bayesian network for skin lesion diagnosis with missing symptom data,
proposed by He et al. (DOI: 10.1109/BIBM62325.2024.10822856).

This code is based on He et al. provided implementation, with the exception of the get_nan_masked_sample function,
which skip nan values in paremeter estimation.
"""
class HeMaskedBayesianNetwork:
    @config_enumerate
    def model(self, itch_obs, grew_obs, hurt_obs, changed_obs, bleed_obs, elevation_obs, site_obs, diameter_obs, age_obs,
            ACK_obs, BCC_obs, MEL_obs, NEV_obs, SCC_obs, SEK_obs, 
            diagnosis_obs=None):
        diagnosis_probs = pyro.param('diagnosis_probs', (torch.ones(6, 6) / 6).cuda(), constraint=constraints.simplex)
        age_probs = pyro.param('age_probs', (torch.ones(36, 10) / 10).cuda(), constraint=constraints.simplex)
        itch_probs = pyro.param('itch_probs', (torch.ones(36, 2) / 2).cuda(), constraint=constraints.simplex)
        grew_probs = pyro.param('grew_probs', (torch.ones(36, 2) / 2).cuda(), constraint=constraints.simplex)
        hurt_probs = pyro.param('hurt_probs', (torch.ones(36, 2) / 2).cuda(), constraint=constraints.simplex)
        changed_probs = pyro.param('changed_probs', (torch.ones(36, 2) / 2).cuda(), constraint=constraints.simplex)
        bleed_probs = pyro.param('bleed_probs', (torch.ones(36, 2) / 2).cuda(), constraint=constraints.simplex)
        elevation_probs = pyro.param('elevation_probs', (torch.ones(36, 2) / 2).cuda(), constraint=constraints.simplex)
        site_probs = pyro.param('site_probs', (torch.ones(36, 14) / 14).cuda(), constraint=constraints.simplex)
        diameter_probs = pyro.param('diameter_probs', (torch.ones(36, 8) / 8).cuda(), constraint=constraints.simplex)

        with pyro.plate('data', len(age_obs)):
            cnn = pyro.sample('cnn', dist.Categorical(probs=torch.stack((ACK_obs, BCC_obs, MEL_obs, NEV_obs, SCC_obs, SEK_obs), dim=1)))
            diagnosis = pyro.sample('diagnosis', dist.Categorical(probs=diagnosis_probs[(cnn).long()]), obs=diagnosis_obs)
            itch = get_nan_masked_sample(itch_obs, itch_probs, cnn, diagnosis, 'itch')
            grew = get_nan_masked_sample(grew_obs, grew_probs, cnn, diagnosis, 'grew')
            hurt = get_nan_masked_sample(hurt_obs, hurt_probs, cnn, diagnosis, 'hurt')
            changed = get_nan_masked_sample(changed_obs, changed_probs, cnn, diagnosis, 'changed')
            bleed = get_nan_masked_sample(bleed_obs, bleed_probs, cnn, diagnosis, 'bleed')
            elevation = get_nan_masked_sample(elevation_obs, elevation_probs, cnn, diagnosis, 'elevation')
            diameter = get_nan_masked_sample(diameter_obs, diameter_probs, cnn, diagnosis, 'diameter')
            site = get_nan_masked_sample(site_obs, site_probs, cnn, diagnosis, 'site')
            age = get_nan_masked_sample(age_obs, age_probs, cnn, diagnosis, 'age')
            return diagnosis

    def guide(self, itch_obs, grew_obs, hurt_obs, changed_obs, bleed_obs, elevation_obs, site_obs, diameter_obs, age_obs, 
          ACK_obs, BCC_obs, MEL_obs, NEV_obs, SCC_obs, SEK_obs, 
          diagnosis_obs=None):
        return

    def predict(self, itch_obs, grew_obs, hurt_obs, changed_obs, bleed_obs, elevation_obs, site_obs, diameter_obs, age_obs, 
            ACK_obs, BCC_obs, MEL_obs, NEV_obs, SCC_obs, SEK_obs):
        conditional_marginals = TraceEnum_ELBO().compute_marginals(self.model, self.guide, itch_obs=itch_obs, grew_obs=grew_obs, hurt_obs=hurt_obs, changed_obs=changed_obs, bleed_obs=bleed_obs, elevation_obs=elevation_obs, site_obs=site_obs, diameter_obs=diameter_obs, age_obs=age_obs, 
                                                               ACK_obs=ACK_obs, BCC_obs=BCC_obs, MEL_obs=MEL_obs, NEV_obs=NEV_obs, SCC_obs=SCC_obs, SEK_obs=SEK_obs)
        
        diagnosis_probs = []
        for i in range(6):
            diagnosis_probs.append(
                conditional_marginals['diagnosis'].log_prob(torch.tensor(i).cuda()).exp().reshape(1, len(itch_obs))
            )
        probs = torch.cat(diagnosis_probs, dim=0).T
        return torch.argmax(probs, dim=1), probs

# Helper function to handle NaN masking in observations
def get_nan_masked_sample(obs, probs, cnn, diagnosis, name):
    valid = ~torch.isnan(obs)
    obs_filled = obs.clone().to(torch.long)
    obs_filled[~valid] = 0
    distribution = dist.Categorical(
        probs=probs[(cnn*6+diagnosis).long()]
    ).mask(valid)
    return pyro.sample(
        name,
        distribution,
        obs=obs_filled
    )