import numpy as np
import lmfit
import scipy.stats as st



class MeltCurveModel(lmfit.Model):
    def __init__(self, *args, **kwargs):
        self.T = kwargs['T']
        if 'f_margin' in kwargs.keys():
            self.f_margin = kwargs['f_margin']

        self.kB = 0.0019872

        self.nan_policy = 'omit'
        
        def melt_curve(T, fmin, fmax, dH, Tm):
            frac_unfolded = (fmin + (fmax - fmin)/
                 (1 + np.exp((dH/self.kB) * (1/Tm - 1/T))))
            return frac_unfolded
        
        super(MeltCurveModel, self).__init__(melt_curve, *args, **kwargs)

    def guess(self, data, **kwargs):
        """
        For single cluster fit
        """
        def guess_Tm(data):
            idmin = np.argmin(np.abs(data-0.5))
            return self.T[idmin]
        
        params = self.make_params()
        def pset(param, value, minimum=-np.inf, maximum=np.inf):
            params["%s%s" % (self.prefix, param)].set(value=value, min=minimum, max=maximum)

        pset("fmin", 0, minimum=-self.f_margin, maximum=self.f_margin)
        pset("fmax", 1, minimum=1-self.f_margin, maximum=1+self.f_margin)
        if 'dH' in kwargs.keys():
            pset("dH", kwargs["dH"], maximum=0)
        else:
            pset("dH", -40, maximum=0)
            
        if 'Tm' in kwargs.keys():
            pset("Tm", kwargs["Tm"])
        else:
            pset("Tm", guess_Tm(data))
        
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)



class MeltCurveRefineModel(MeltCurveModel):
    """
    One model instance for each variant. Repeatedly fit in each bootstrap iteration
    """
    def __init__(self, fmax_params_dict, variant_table_row, *args, **kwargs):

        self.variant_table_row = variant_table_row
        self.set_fmax_dist_params(fmax_params_dict, n=variant_table_row["numTests"])
        self.epsilon = 0.01
        super().__init__(*args, **kwargs)


    def set_fmax_dist_params(self, fmax_params_dict, n):
        def get_sigma(a,b,n):
            return max(a / np.sqrt(n) + b, 0)

        self.fmax_mu = fmax_params_dict["fmax"]["mu"]
        self.fmin_mu = fmax_params_dict["fmin"]["mu"]
        self.fmax_sigma = get_sigma( a=fmax_params_dict["fmax"]["sigma"]["a"],
                        b=fmax_params_dict["fmax"]["sigma"]["b"], n=n )
        self.fmin_sigma = get_sigma( a=fmax_params_dict["fmin"]["sigma"]["a"],
                        b=fmax_params_dict["fmin"]["sigma"]["b"], n=n )

        self.fmax_lb = self.fmax_mu - 10 * self.fmax_sigma
        self.fmin_ub = self.fmin_mu + 1 * self.fmin_sigma


    def guess(self, enforce_fmax, enforce_fmin, **kwargs):
        """
        For refine variant fit
        Args:
            variant_table_row - a row in the variant table with the bootstrapped single cluster fit results
            reach_fmax, reach_fmin - bool, whether the variant has passed the reach fmax or fmin criteria
                if not, draw from the estimated fmax fim distribution during fitting
        """        
        params = self.make_params()
        def pset(param, value, minimum=-np.inf, maximum=np.inf):
            params["%s%s" % (self.prefix, param)].set(value=value, min=minimum, max=maximum)

        pset("fmax", self.fmax_mu)
        pset("fmin", self.fmin_mu)
        pset("dH", self.variant_table_row["dH_init"], maximum=0)
        pset("Tm", self.variant_table_row["Tm_init"])
        
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

    def is_good_init_fit(self):
        return self.variant_table_row["pvalue"] < 0.01

    def get_params_from_results(self, results, postfix=''):
        params = self.make_params()

        for p in self.param_names:
            params[p].set(value=results[p + postfix])

        return params

    def draw_fmax_sample(self, params, var_name="fmax"):
        assert var_name in ("fmax","fmin")

        mu = getattr(self, "%s_mu"%var_name)
        sigma = getattr(self, "%s_sigma"%var_name)
        simulated_fmax = np.random.normal(loc=mu, scale=sigma)
        params[var_name].set(value=simulated_fmax, vary=False)

        return params

    def make_fmaxes(self, n_samples=100, var_name="fmax"):

        mu = getattr(self, "%s_mu"%var_name)
        sigma = getattr(self, "%s_sigma"%var_name)
        fmaxes = np.random.normal(loc=mu, scale=sigma, size=n_samples)

        return fmaxes

    def decide_enforce_fmax_distribution(self, median_signal):
        """
        If True, enforce to fitted fmax or fmin distribution;
        else, clamp to fmax init fit value during fitting
        """
        enforce_fmax, enforce_fmin = True, True

        if (median_signal[-1] > self.fmax_lb) and (self.is_good_init_fit()):
            enforce_fmax = False
        if (median_signal[0] < self.fmin_ub) and (self.is_good_init_fit()):
            enforce_fmin = False

        return enforce_fmax, enforce_fmin


class MeltCurveVanillaModel(MeltCurveRefineModel):
    """
    Clamp fmax and fmin to 1 and 0
    """
    def make_fmaxes(self, n_samples=100, var_name="fmax"):

        if var_name == "fmax":
            fmaxes = np.ones(n_samples)
        elif var_name == "fmin":
            fmaxes = np.zeros(n_samples)

        return fmaxes

    def decide_enforce_fmax_distribution(self, median_signal, model):
        """
        Always enforce
        """
        return True, True


class GammaModel(lmfit.Model):
    def __init__(self, var_name, *args, **kwargs):
        self.var_name = var_name
        self.nan_policy = 'omit'
        
        def gamma(x, mean, std, offset):
            k, theta = self.returnGammaParams(mean, std)
            cdf = st.gamma.cdf(x, k, scale=theta, loc=offset)
            return cdf
        
        super(GammaModel, self).__init__(gamma, *args, **kwargs)


    @staticmethod
    def returnGammaParams(mean, std):
        """Return shape and scale parameters from moments of distrubtion mean, std"""
        k = (mean/std)**2
        theta = (std)**2/mean
        return k, theta

    def getParams(self, **kwargs):
        params = self.make_params()
        def pset(param, value, minimum=-np.inf, maximum=np.inf, vary=True):
            params["%s%s" % (self.prefix, param)].set(value=value, min=minimum, max=maximum, vary=vary)

        pset("mean", 1)
        pset("std", 1, minimum=0)

        if ("set_offset" in kwargs.keys()):
            pset("offset", kwargs["set_offset"], vary=False)
        else:
            pset("offset", 0)

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class SigmaNModel(lmfit.Model):
    def __init__(self, *args, **kwargs):

        self.nan_policy = 'omit'
        
        def sigma_n(n, a, b):
            return a / np.sqrt(n) + b
        
        super(SigmaNModel, self).__init__(sigma_n, *args, **kwargs)


    def guess(self, **kwargs):
        params = self.make_params()
        def pset(param, value, minimum=-np.inf, maximum=np.inf, vary=True):
            params["%s%s" % (self.prefix, param)].set(value=value, min=minimum, max=maximum, vary=vary)

        pset("a", 0.1, minimum=0)
        pset("b", 0.01, minimum=0)

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)