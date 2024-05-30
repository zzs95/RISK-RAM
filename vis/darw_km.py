import os
import sys
sys.path.append(os.getcwd())
import numpy as npz
from matplotlib import pyplot as plt
import pandas as pd
from lifelines.utils import survival_table_from_events
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import multivariate_logrank_test

exp_path = '/home/brownradai/Public/hhd_3t_shared/covid_cxr_RSPNet/gen_region_risk/run_1'

# setname = 'penn'
setname = 'brown'

KM_dict = pd.read_excel(os.path.join(exp_path, setname+'.xlsx'))
risk = KM_dict['risk_score']
risk_median = np.median(risk)
risk_idx = risk > risk_median
time = np.array(KM_dict['time_label'] * 77.025).astype(int)
event = np.array(KM_dict['event_label'] ).astype(int)
# fig, ax =  plt.subplots(figsize=(3,2.4))
fig, ax =  plt.subplots(figsize=(5,3), dpi=300)

kmf = KaplanMeierFitter()
for risk_label in ['Low Risk', 'High Risk']:
    if risk_label == 'High Risk':
            case_idx =  risk_idx
            color = '#fc4f30'
    else:
            case_idx =  (1-risk_idx).astype(bool)
            color = '#008fd5'
            
    # kmf.fit(time[case_idx], event[case_idx], label=risk_label)
    kmf.fit(time[case_idx], event[case_idx])
    kmf.plot_survival_function(color=color, ax=ax, )

results = multivariate_logrank_test(time, risk_idx, event)
# results.print_summary()
p_value = results.p_value
p_text = 'p<{:.5f}'.format(p_value)
y_lim = ax.get_ylim()
y = (y_lim[1] - y_lim[0]) * 0.1 + y_lim[0]
x_lim = ax.get_xlim()
x = (x_lim[1] - x_lim[0]) * 0.07 + x_lim[0]
ax.text(x=x, y=y, s=p_text)
# ax.set(
#     # title='Kaplan-Meier survival curves of Multimodal CoxPH model',
#     xlabel='Time in Days',
#     ylabel='Estimated Survival Probability',
# )
ax.set(
    # title='Kaplan-Meier survival curves of Multimodal CoxPH model',
    xlabel='',
    ylabel='',
)
ax.grid(linestyle = '--', linewidth = 0.5)
ax.get_legend().remove()
plt.savefig(setname+'_km.jpg')