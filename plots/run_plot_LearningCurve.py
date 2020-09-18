from regularizeTool import EarlyStopping
from trainTool import train
from loadDataTool import load_train_N_validate_data
from os.path import join
from evaluateTool import *
import scipy.io as sio
from os import mkdir
from loadModel import get_model, load_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from AnalyticalModel import *
import numpy as np



################################################################################################################

# define train and test path
data_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual",'result')
file_name = 'ReLU_Dual_UDirection_PKD_learnCurve.mat'
train_losses = sio.loadmat(join(data_path,file_name))['train_losses']
valid_losses = sio.loadmat(join(data_path,file_name))['valid_losses']
train_losses = train_losses[0]
valid_losses = valid_losses[0]

paperFontSize = 14
paperLineWidth =2.5

legend_list = ['Training Loss', 'Validating Loss']


fig, ax = plt.subplots(figsize=(6.7, 3.2))
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = paperFontSize
fill_color_list = ['tab:green','tab:orange']

# print(np.argmin(valid_losses))
smallest_idx = np.argmin(valid_losses) +1
x = list(range(1,train_losses.shape[0]+1))
ax.plot(x, train_losses.tolist(),color=fill_color_list[0], alpha=0.8, label=legend_list[0],linewidth=paperLineWidth)
ax.plot(x, valid_losses.tolist(),color=fill_color_list[1], alpha=0.8, label=legend_list[1],linewidth=paperLineWidth)
ax.plot([smallest_idx, smallest_idx], [0, 0.3], '--')

arrow_properties = dict(
    facecolor="black", width=0.5,
    headwidth=6, shrink=0.1)

plt.annotate(
    "Early Stopping", xy=(smallest_idx, 0.1),
    xytext=(smallest_idx-60, 0.03),
    arrowprops=arrow_properties)
#
ax.margins(y=0.1, x=.03)

ax.yaxis.grid(True)
plt.rcParams["font.serif"] = "Times New Roman"

# maxValue = max([max(list) for list in abs_rms_list])
# plt.show()

csfont = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'fontsize'   : paperFontSize}
# Save the figure and show
ax.set_xlabel('Epoches',**csfont)
ax.set_ylabel(r'Loss',**csfont)
a = plt.gca()
#
# ax.set_ylim([-0.1, max(train_losses.tolist())+0.02])

xtick = a.get_xticks().astype(int)
ytick = np.around(a.get_yticks().astype(float), decimals=4)

a.set_xticklabels(xtick, **csfont)
a.set_yticklabels(ytick, **csfont)


#


font = matplotlib.font_manager.FontProperties(family='Times New Roman',size=paperFontSize)
ax.legend(loc='upper center', prop=font, bbox_to_anchor=(0.5, 1.3),
          fancybox=True, shadow=True, ncol=3)
#
plt.yscale('log')
#
# ax[0].tick_params(axis='both', which='major', labelsize=paperFontSize)
# ax[0].tick_params(axis='both', which='minor', labelsize=paperFontSize)
# ax[1].tick_params(axis='both', which='major', labelsize=paperFontSize)
# ax[1].tick_params(axis='both', which='minor', labelsize=paperFontSize)
# ax[0].set_tick_params(labelsize=paperFontSize)
# # ax[1].set_tick_params(labelsize=paperFontSize)
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False
plt.tight_layout()
plt.show()
# #
# if not os.path.exists(save_pdf_path):
#     os.makedirs(save_pdf_path)


fig.savefig(join(data_path,'ReLU_Dual_UDirection_PKD_learnCurve.pdf'),bbox_inches='tight')