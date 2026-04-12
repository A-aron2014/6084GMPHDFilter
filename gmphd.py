import numpy as np
from copy import deepcopy
from operator import attrgetter
from dataclasses import dataclass,field
#why?
simplesum = sum

myfloat = np.float64
#making this a data class because of the built in decorators
@dataclass
class GmphdComponent:
    weight  : myfloat
    mean     : np.array
    cov     : np.array

    #These values are computed in __post_init__
    k       : int        = field(default=0, init=False) #This is the coefficient that will define the shape of the gaussian mixture
    dmv_part1   : float  = field(default=0.0,init=False)
    dmv_part2   : float  = field(default=0.0,init=False)
    def __post_init__(self):

        self.mean    = np.array(self.mean, dtype=myfloat, ndmin=2)
        self.cov    = np.array(self.cov, dtype=myfloat, ndmin=2)
        self.mean    = self.mean.reshape(-1, 1)
        n           = self.mean.shape[0]
        self.cov    = self.cov.reshape(n, n)
        
        #precalculated values for evaluating a gaussian
        self.k          = n
        self.dmv_part1  = (2*np.pi) ** (-self.k*0.5)
        self.dmv_part2  = np.power(np.linalg.det(self.cov), -0.5)
        self.incov      = np.linalg.inv(self.cov)

    def dmvnorm(self, x):
        #Evaluate the multivariate normal component at location x
        x = np.array(x,dtype=myfloat)
        error = x - self.mean
        S = np.exp(-0.5*(error@self.incov)@error.T)

        return self.dmv_part1*self.dmv_part2*S
    
def dmvnorm(mean,cov,x):
    "Evaluate a multivariate normal, given an estimated mean (vector) and covariance (matrix) and a position x (vector) at which to evaluate"
    mean = np.array(mean, dtype=myfloat).flatten()
    cov = np.array(cov, dtype=myfloat)
    x = np.array(x, dtype=myfloat).flatten()
    k = len(mean)
    #Here build the gaussian mixture
    p1 = (2.0 * np.pi) ** (-k * 0.5)
    p2 = np.power(np.linalg.det(cov), -0.5)
    error = x - mean
    exponent = (-0.5 * (error@ np.linalg.inv(cov))@ error.T)
    return p1 * p2 * np.exp(exponent)

def sampleGaussianMixture(component_list):
    #Given a list of GmPHD components, randomly sample a value from the density they represent
    weights = np.array([c.weight for c in component_list])
    weights /= np.sum(weights) #Could choose simpleSum too but np.sum should be faster for arrays
    choice = np.random.random()
    cumulative = 0.0
    for i,w in enumerate(weights):
        cumulative += w
        if choice <= cumulative:
            #sample from the chosen component and return a value
            comp = component_list[i]
            return np.random.multivariate_normal(comp.loc.flatten(),comp.cov) #Do I really need to flatten this?
    raise RuntimeError("SampleGaussianMixture terminated without selecting a component")

class Gmphd:
    def __init__(self, birthgmm,p_birth, p_survival, p_detect, F, Q, H, R, clutter):
        self.gmm = []
        self.birthgmm = birthgmm
        self.p_survival = myfloat(p_survival)
        self.p_detect = myfloat(p_detect)
        self.F = np.array(F,dtype=myfloat)
        self.Q = np.array(Q,dtype=myfloat)
        self.H = np.array(H,dtype=myfloat)
        self.R = np.array(R,dtype=myfloat)
        self.clutter = myfloat(clutter)

    def update(self,measurements):
        #Unpack the measurements: Accept dicts {"z","conf"} or raw arrays
        parsed = []
        for m in measurements:
            if isinstance(m,dict):
                parsed.append((np.array(m["z"],dtype=myfloat), float(m.get("conf",1.0))))
            else:
                parsed.append((np.array(m,dtype=myfloat),1.0))

        #Step 1 Birthing and Spawning
        born = [deepcopy(component) for component in self.birthgmm]

        #look at the paper for records but they have a spawning mechanism as part of step 1
        spawned = [] # check other PHD papers about adaptive spawning

        #Step 2 - Prediction of existing targets
        #This appears to be like the kalman filter propagation step for every component within the RFS
        #TODO maybe break this out into the propagated x_apriori and P_apriori?
        updated = [GmphdComponent(self.p_survival*c.weight, (self.F@c.mean), (self.F@c.cov)@self.F.T + self.Q) for c in self.gmm]

        predicted = born + spawned + updated

        #step 3 Construct the PHD Kalman updated components
        mu = [(self.H @ c.mean) for c in predicted]
        S  = [(self.H @ c.cov @ self.H.T + self.R) for c in predicted]
        K = [(c.cov @ self.H.T) @ np.linalg.inv(S[index]) for index, c in enumerate(predicted)]
        P_posteriori = [((np.eye(len(K[index])) - (K[index]) @ self.H) @ c.cov) for index,c in enumerate(predicted)]

        #Step 4 - Missed Detection Components
        new_gmms = [
            GmphdComponent(c.weight*(1.0-self.p_detect),c.mean, c.cov) 
            for c in predicted
            ]

        #Step 5 Update components per measurement
        for meas_vec,conf in parsed:
            new_gmm_partial = []

            for j,component in enumerate(predicted):
                likelihood = dmvnorm(mu[j],S[j],meas_vec)

                new_gmm_partial.append(
                    GmphdComponent(
                        self.p_detect * c.weight * conf * likelihood,
                        component.mean + K[j] @(meas_vec - mu[j]),
                         P_posteriori[j]
                    )
                )

            #Normalise against clutter only after all components are built
            weightedsum = np.sum(c.weight for c in new_gmm_partial)
            rectified_weight = 1.0/(self.clutter+weightedsum) #lambda exhogenous?

            for c in new_gmm_partial:
                c.weight *=rectified_weight
            new_gmms.extend(new_gmm_partial)

        self.gmm = new_gmms

    def prune_targets(self, prune_threshold=1e-6, merge_threshold=4.0, max_components=100):
        #Prune the Gaussian Mixtures -> Look at table 2 of Vo and Ma

        w0 = [simplesum(c.weight for c in self.gmm)] #diagnostic

        sourcegmm       = [c for c in self.gmm if c.weight > prune_threshold]
        original_length = len(self.gmm)
        pruned_length   = len(sourcegmm)
        
        newgmm = []
        while len(sourcegmm) > 0:
            windex   = int(np.argmax([c.weight for c in sourcegmm]))
            heaviest = sourcegmm[windex]
            sourcegmm = sourcegmm[:windex] + sourcegmm[windex + 1:]

            distances = [
                float((c.mean - heaviest.mean).T @ c.incov @ (c.mean - heaviest.mean))
                for c in sourcegmm
            ]
            to_merge = np.array([d <= merge_threshold for d in distances])

            merged = [heaviest]
            if any(to_merge):
                merged.extend(list(np.array(sourcegmm)[to_merge]))
                sourcegmm = list(np.array(sourcegmm)[~to_merge])

            agg_w    = simplesum(c.weight for c in merged)
            agg_mean = sum(c.weight * c.mean for c in merged) / agg_w
            agg_cov  = sum(
                c.weight * (c.cov + (heaviest.mean - c.mean) @ (heaviest.mean - c.mean).T)
                for c in merged
            ) / agg_w

            newgmm.append(GmphdComponent(agg_w, agg_mean, agg_cov))

        newgmm.sort(key=attrgetter('weight'), reverse=True)
        self.gmm = newgmm[:max_components]

        w1 = simplesum(c.weight for c in newgmm)
        w2 = simplesum(c.weight for c in self.gmm)

        print(f"prune(): {original_length} -> {pruned_length} -> {len(newgmm)} -> {len(self.gmm)}")
        print(f"prune(): weightsums {w0:.4g} -> {w1:.4g} -> {w2:.4g}")

        # Renormalise to conserve total PHD mass after truncation
        if w2 > 0:
            scale = w0 / w2
            for c in self.gmm:
                c.weight *= scale

    def extractstate(self, bias=1.0):
        """Extract target states from components with rounded weight >= 1."""
        items = []
        print("weights:", [round(c.weight, 7) for c in self.gmm])
        for c in self.gmm:
            val = c.weight * float(bias)
            if val > 0.5:
                for _ in range(int(round(val))):
                    items.append(deepcopy(c.loc))
        for x in items:
            print(x.T)
        return items

