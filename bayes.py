import numpy as np
from scipy.stats import norm
from dists import Beta, Norm
from preprocess import preprocess
import pandas as pd



BETA_PRIOR = 1, 1
NORM_PRIOR = 0, 1, 0


def binomial_ab_test(x_a, n_a, x_b, n_b, ccr=.05):
    alpha_a, beta_a = Beta.posterior(BETA_PRIOR, [x_a, n_a])
    alpha_b, beta_b = Beta.posterior(BETA_PRIOR, [x_b, n_b])

    mean_a, var_a = Beta.moments(alpha_a, beta_a, log=True)
    mean_b, var_b = Beta.moments(alpha_b, beta_b, log=True)

    mean_diff = mean_b - mean_a
    std_diff = np.sqrt(var_a + var_b)

    chance_to_win = norm.sf(0, mean_diff, std_diff)
    expected = np.exp(mean_diff) - 1
    ci = np.exp(norm.ppf([ccr / 2, 1 - ccr / 2], mean_diff, std_diff)) - 1
    risk_beta = Beta.risk(alpha_a, beta_a, alpha_b, beta_b)

    output = {'chance_to_win': chance_to_win,
              'expected': expected,
              'ci': ci.tolist(),
              'uplift': {'dist': 'lognormal',
                         'mean': mean_diff,
                         'stddev': std_diff},
              'risk': risk_beta.tolist()}

    return output


def gaussian_ab_test(m_a, s_a, n_a, m_b, s_b, n_b, ccr=.05):
    mu_a, sd_a = Norm.posterior(NORM_PRIOR, [m_a, s_a, n_a])
    mu_b, sd_b = Norm.posterior(NORM_PRIOR, [m_b, s_b, n_b])

    mean_a, var_a = Norm.moments(mu_a, sd_a, log=True)
    mean_b, var_b = Norm.moments(mu_b, sd_b, log=True)

    mean_diff = mean_b - mean_a
    std_diff = np.sqrt(var_a + var_b)

    chance_to_win = norm.sf(0, mean_diff, std_diff)
    expected = np.exp(mean_diff) - 1
    ci = np.exp(norm.ppf([ccr / 2, 1 - ccr / 2], mean_diff, std_diff)) - 1
    risk_norm = Norm.risk(mu_a, sd_a, mu_b, sd_b)

    output = {'chance_to_win': chance_to_win,
              'expected': expected,
              'ci': ci.tolist(),
              'uplift': {'dist': 'lognormal',
                         'mean': mean_diff,
                         'stddev': std_diff},
              'risk': risk_norm.tolist(),
              'mean': {'m_a':m_a,
                       'm_b':m_b}
            }
    return output



def run_test(data, metrics, **kwargs):
    df = preprocess(data, **kwargs)
    descriptive_metric = []
    number_user = df.groupby('experiment_group').agg({'resettable_device_id_or_app_instance_id':pd.Series.nunique})
    for i in metrics:
        descriptive_metric.append(df.groupby('experiment_group').agg({i: ['mean', 'std']}))
    descriptive_metric = pd.concat(descriptive_metric, axis=1)

    for metric in metrics:
        data_metrics = descriptive_metric[metric]
        na, nb = number_user.resettable_device_id_or_app_instance_id
        ma, mb = data_metrics['mean']
        sa, sb = data_metrics['std']
        globals()[metric] = gaussian_ab_test(m_a=ma, s_a=sa, n_a=na, m_b=mb, s_b=sb, n_b=nb)


if __name__ == '__main__':
    import json
    # import sys

    # metric = sys.argv[1]
    # data = json.loads(sys.argv[2])
    metric = 'normal'
    data = {"users":[1847,1887],"count":[950,1025],"mean":[819.7544,783.9616],"stddev":[1396.129,1311.846]}

    xa, xb = data['count']
    na, nb = data['users']
    ma, mb = data['mean']
    sa, sb = data['stddev']

    if metric == 'binomial':
        print(json.dumps(binomial_ab_test(x_a=xa, n_a=na, x_b=xb, n_b=nb)))

    else: 
        print(json.dumps(gaussian_ab_test(m_a=ma, s_a=sa, n_a=na, m_b=mb, s_b=sb, n_b=nb)))
