import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

coef_pos = [9,    1,-2,1,-0.1]
coef_neg = [6,    -1,-2,1,-0.1]
paperFontSize = 14


def func(x, coef: list()):
    y = 0
    for i, c in enumerate(coef):
        y = y + c * pow(x, i)

    return y

inputs = np.linspace(0, 2*np.pi, num=500).tolist()
output_pos = []
output_neg = []
output_mean = []
for input in inputs:
    a = func(input, coef_pos)
    b = func(input, coef_neg)
    output_pos.append(a)
    output_neg.append(b)
    output_mean.append((a+b)/2)


fig, ax = plt.subplots(figsize=(5, 4.2))
ax.plot(inputs, output_pos, label=r'$\epsilon^{+}$', color = 'steelblue')
ax.plot(inputs, output_neg, label=r'$\epsilon^{-}$', color = 'peru')
ax.plot(inputs, output_mean, label='mean', color = 'k', linestyle='--')


idx_c = 300
ax.plot([inputs[idx_c], inputs[idx_c]], [0, output_pos[idx_c]], color = 'gray', linestyle='--', linewidth=1)
ax.plot([0, inputs[idx_c]], [output_mean[idx_c], output_mean[idx_c]], color = 'gray', linestyle='--', linewidth=1)

d1 = 1.0
d2 = 1.4
ax.plot([inputs[idx_c], inputs[idx_c]+d2], [output_mean[idx_c], output_mean[idx_c]], color = 'gray', linestyle='--', linewidth=1)
ax.plot([inputs[idx_c], inputs[idx_c]+d1], [output_neg[idx_c], output_neg[idx_c]], color = 'gray', linestyle='--', linewidth=1)
ax.plot([inputs[idx_c], inputs[idx_c]+d1], [output_pos[idx_c], output_pos[idx_c]], color = 'gray', linestyle='--', linewidth=1)
# ax.plot([inputs[idx_c], inputs[idx_c]+0.6], [output_neg[idx_c], output_neg[idx_c]], color = 'gray', linestyle='-', linewidth=1)

plt.arrow(inputs[idx_c]+d2, output_mean[idx_c], 0, -4, head_width=0.06, head_length=0.5, facecolor='black')
plt.arrow(inputs[idx_c]+d2, 0, 0, 4, head_width=0.06, head_length=0.5, facecolor='black')
ax.text(inputs[idx_c]+d2-0.2, output_mean[idx_c]/2-0.5, r'$|\tau_c|$', fontsize=12)


plt.arrow(inputs[idx_c]+d1, output_mean[idx_c], 0, 1, head_width=0.06, head_length=0.5, facecolor='black')
plt.arrow(inputs[idx_c]+d1, output_pos[idx_c], 0, -1, head_width=0.06, head_length=0.5, facecolor='black')
ax.text(inputs[idx_c]+d1-0.2, (output_mean[idx_c] + output_pos[idx_c])/2-0.8, r'$|\tau_d|$', fontsize=12)

plt.arrow(inputs[idx_c]+d1, output_neg[idx_c], 0, 1, head_width=0.06, head_length=0.5, facecolor='black')
plt.arrow(inputs[idx_c]+d1, output_mean[idx_c], 0, -1, head_width=0.06, head_length=0.5, facecolor='black')
ax.text(inputs[idx_c]+d1-0.2, (output_neg[idx_c] + output_mean[idx_c])/2-0.8, r'$|\tau_d|$', fontsize=12)



x = [0, np.pi,2 * np.pi]
labels = ['0', r'$\pi$', '$2\pi$']
plt.xticks(x, labels)
plt.xlim([0, 2 * np.pi])
plt.ylim([0, 30])



# Save the figure and show
csfont = {'fontname': 'Times New Roman', 'fontsize': paperFontSize}
ax.set_xticklabels(labels, **csfont)
ax.set_ylabel(r'$\tau$ (N.m)', **csfont)
ax.set_xlabel(r'q', **csfont)
a = plt.gca()
a.set_yticklabels(a.get_yticks(), **csfont)

font = matplotlib.font_manager.FontProperties(family='Times New Roman', size=paperFontSize)
ax.legend(loc='upper center', prop=font, bbox_to_anchor=(0.5, 1.18),
          fancybox=True, shadow=True, ncol=3)
plt.xticks(fontsize=paperFontSize)
plt.yticks(fontsize=paperFontSize)
plt.tight_layout()
plt.show()
matplotlib.rcParams['pdf.fonttype'] = 42
fig.savefig(join('.', 'disturbance.pdf'), bbox_inches='tight')

# # pos arrow
# idx = 250
# x = inputs[idx]
# y = output_pos[idx]
# plt.arrow(x, y, x+0.5, y+0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')



