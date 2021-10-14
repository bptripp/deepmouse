import pickle
import numpy as np
import matplotlib.pyplot as plt

corrs_by_step = []
for i in range(6):
    with open('../step_{}_probe_corr.pkl'.format(i), 'rb') as file:
        probe_areas, corrs = pickle.load(file)
        corrs_by_step.append(corrs)

plt.figure(figsize=(8,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.title(probe_areas[i])

    selected_corrs = np.zeros((6,4))
    for j in range(6):
        foo = corrs_by_step[j]
        selected_corrs[j,:] = corrs_by_step[j][i][0,1:]

    selected_corrs[0,1:] = np.nan
    print(selected_corrs)

    plt.plot(range(6), selected_corrs)
    plt.ylim([-1, 1])
    plt.xlim([0, 5])

plt.subplot(2,5,1), plt.ylabel('V-az Correlations')
plt.subplot(2,5,6), plt.ylabel('V-az Correlation')
plt.subplot(2,5,8), plt.xlabel('Steps')
plt.subplot(2,5,5), plt.legend(('V-el', 'B-az', 'B-el', 'A'), loc=(1.3,.15))
plt.tight_layout()
plt.savefig('corr-per-step.png')
plt.show()


