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
        #Taking this out bc I think it's fucking me
        #self.mean    = np.array(self.mean, dtype=myfloat, ndmin=2)
        #self.cov    = np.array(self.cov, dtype=myfloat, ndmin=2)
        #self.mean    = self.mean.reshape(-1, 1)
        self.mean = np.asarray(self.mean, dtype=myfloat).reshape(-1)
        self.cov  = np.asarray(self.cov, dtype=myfloat)
        n           = self.mean.shape[0]
        #self.cov    = self.cov.reshape(n, n)
        assert self.cov.shape == (n, n), f"Bad cov shape: {self.cov.shape}"
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
            return np.random.multivariate_normal(comp.mean,comp.cov) #Do I really need to flatten this?
    raise RuntimeError("SampleGaussianMixture terminated without selecting a component")

class Gmphd:
    def __init__(self, birthgmm,p_birth, p_survival, p_detect, F, Q, H, R, clutter,birth_trigger_threshold= 0.15):
        self.gmm = []
        self.birthgmm = birthgmm
        self.p_survival = myfloat(p_survival)
        self.p_detect = myfloat(p_detect)
        self.p_birth    = myfloat(p_birth)
        self.F = np.array(F,dtype=myfloat)
        self.Q = np.array(Q,dtype=myfloat)
        self.H = np.array(H,dtype=myfloat)
        self.R = np.array(R,dtype=myfloat)
        self.clutter = myfloat(clutter)

        self.birth_trigger_threshold = myfloat(birth_trigger_threshold)

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
        #Removed based on Clark 2006
        spawned = [] # check other PHD papers about adaptive spawning
        #Step 2 - Prediction of existing targets
        F = self.F
        updated = []
        #This is encodes our dynamics from k-1 to k 
        for c in self.gmm:
            m = self.F@c.mean
            P = self.F@c.cov @self.F.T+ self.Q
            updated.append(GmphdComponent(self.p_survival*c.weight, m, P))

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
        new_birth = []
        for meas_vec,conf in parsed:
            new_gmm_partial = []

            for j,component in enumerate(predicted):
                likelihood = dmvnorm(mu[j],S[j],meas_vec)

                new_gmm_partial.append(
                    GmphdComponent(
                        self.p_detect * component.weight * conf * likelihood,
                        (component.mean + K[j] @(meas_vec - mu[j])),
                         P_posteriori[j]
                    )
                )

            #Normalise against clutter only after all components are built
            weightedsum = simplesum(c.weight for c in new_gmm_partial)
            rectified_weight = 1.0/(self.clutter+weightedsum) #lambda exhogenous?

            for c in new_gmm_partial:
                c.weight *=rectified_weight
            new_gmms.extend(new_gmm_partial)
            total_likelihood = simplesum(
                dmvnorm(mu[j], S[j], meas_vec) * predicted[j].weight
                for j in range(len(predicted))
            )

            # After the update step, identify "unassociated" measurements
            # (those that contributed low total likelihood across all predicted components)
            if total_likelihood < self.birth_trigger_threshold:
                new_birth.append(GmphdComponent(
                    weight = self.p_birth*conf,
                    mean = np.array([meas_vec[0],meas_vec[1],0.0,0.0]),
                    cov = np.diag([50.0,50.0,25.0,25.0])
                ))

        self.gmm = new_gmms

    def prune_targets(self, prune_threshold=1e-6, merge_threshold=4.0, max_components=100):
        #Prune the Gaussian Mixtures -> Look at table 2 of Vo and Ma

        w0 = simplesum(c.weight for c in self.gmm) #diagnostic

        sourcegmm       = [c for c in self.gmm if c.weight > prune_threshold]
        original_length = len(self.gmm)
        pruned_length   = len(sourcegmm)
        
        newgmm = []
        while len(sourcegmm) > 0:
            windex   = int(np.argmax([c.weight for c in sourcegmm]))
            heaviest = sourcegmm[windex]
            sourcegmm = sourcegmm[:windex] + sourcegmm[windex + 1:]
            inv_cov = heaviest.incov
            distances = [
                ((c.mean - heaviest.mean).T @ inv_cov @ (c.mean - heaviest.mean)).item() #replaced c.incov with inv_cov
                for c in sourcegmm
            ]
            to_merge = np.array([d <= merge_threshold for d in distances])

            merged = [heaviest]
            if np.any(to_merge):
                merged.extend(list(np.array(sourcegmm)[to_merge]))
                sourcegmm = list(np.array(sourcegmm)[~to_merge])

            agg_w    = simplesum(c.weight for c in merged)
            agg_mean = np.zeros_like(heaviest.mean)

            #agg_mean = sum(c.weight * c.mean for c in merged) / agg_w
            for c in merged:
                agg_mean += c.weight*c.mean
            agg_mean /= agg_w #normalize the mean
            agg_cov = np.zeros_like(heaviest.cov)
            # agg_cov  = sum(
            #     c.weight * (c.cov + (heaviest.mean - c.mean) @ (heaviest.mean - c.mean).T)
            #     for c in merged
            # ) / agg_w
      
            for c in merged:
                diff = (c.mean - agg_mean).reshape(-1,1) #adding reshape to test #Orignially was heaviest mean - c.mean which must have been skewing things hard
                agg_cov += c.weight*(c.cov + diff @ diff.T)
            agg_cov = agg_cov/agg_w
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
        #More GMPHD approach
        #if val > 0.2:

        for c in self.gmm:
            n = int(round(c.weight* bias))
            for _ in range(n):
                items.append(deepcopy(c.mean))

        # for c in self.gmm:
        #     val = c.weight * float(bias)

        #     if val> 0.05:
        #         items.append(deepcopy(c.mean))

        for x in items:
            print(x.T)
        return items