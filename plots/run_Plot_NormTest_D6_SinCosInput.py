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
train_data_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual")
test_data_path = join("data", "MTMR_28002", "real", "random", 'N319','D6_SinCosInput')

# load Trajectory Test experiment data
test_dataset = load_data_dir(join(test_data_path, "data"), device='cpu',input_scaler=None, output_scaler=None,
                             is_inputScale = False, is_outputScale = False)

test_input_mat = test_dataset.x_data.numpy()
test_ouput_mat = test_dataset.y_data.numpy()


test_output_hat_mat_List = []
legend_list = []



device = 'cpu'
D = 6

norm_type_lst = ['noInOutNorm', 'noOutNorm', 'noInNorm', None]
# get predict DNN with Knowledge Distillation output

for norm_type in norm_type_lst:
    use_net = 'ReLU_Dual_UDirection'
    train_type = 'PKD'
    load_model_path = join(train_data_path, "result", "model")
    model = get_model('MTM', use_net, D, device=device)
    if norm_type is not None:
        load_name =  use_net+'_'+train_type+'_'+norm_type
    else:
        load_name = use_net + '_' + train_type
    model, _, _ = load_model(load_model_path,load_name, model)
    test_output_hat_mat_List.append(model.predict_NP(test_input_mat))




legend_list = ['Without Norm', 'Input Norm', 'Output Norm', 'Input-Output Norm']


# plot predict error bar figures
abs_rms_list = []
rel_rms_list = []
mean_rel_rms_list = []
for i in range(len(test_output_hat_mat_List)):
    err_output_mat = np.abs(test_output_hat_mat_List[i] - test_ouput_mat)
    abs_rms_list.append(np.sqrt(np.mean(np.square(err_output_mat), axis=0)).tolist())
    rel_rms_list.append(np.sqrt(np.divide(np.mean(np.square(err_output_mat), axis=0),
                                    np.mean(np.square(test_ouput_mat), axis=0))).tolist())

for i in range(len(rel_rms_list)):
    abs_rms_list[i].append(np.mean(abs_rms_list[i], axis=0))
    rel_rms_list[i].append(np.mean(rel_rms_list[i],axis=0))

for i in range(len(rel_rms_list)):
    rel_rms_list[i] =[k*100 for k in rel_rms_list[i]]

#print(err_output_mat)



paperFontSize = 16
jnt_index = np.arange(1,8)
# fig, ax = plt.subplots()
# # matplotlib.rc('text', usetex=True)
# # matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#
# plt.rcParams["font.family"] = "Times New Roman"
# w = 0.2
# space = 0.2
# capsize = 2
# fontsize = 30
# fill_color_list = ['tab:blue','tab:orange', 'tab:green']
#
# for i in range(len(abs_rms_list)):
#     ax.bar(jnt_index+space*(i-1), abs_rms_list[i],  width=w,align='center', color=fill_color_list[i], alpha=0.8, ecolor='black', capsize=capsize, label=legend_list[i])
#
# ax.set_xticks(jnt_index)
# labels = ['Joint '+str(i+1) for i in range(6)]
# labels.append('Avg')
# # ax.set_title('Absolute RMSE for Trajectory Test')
# ax.yaxis.grid(True)
# ax.autoscale(tight=True)
# # maxValue = max([max(list) for list in abs_rms_list])
# # plt.ylim(0, maxValue*1.2)
# ax.margins(y=.1, x=.03)
#
# # Save the figure and show
# csfont = {'fontname':'Times New Roman', 'fontsize':paperFontSize}
# ax.set_xticklabels(labels, **csfont)
# ax.set_ylabel(r'$\epsilon_{rms}$ (N.m)', **csfont)
# a = plt.gca()
# a.set_yticklabels(a.get_yticks(), **csfont)
# ax.legend(fontsize=paperFontSize)
# plt.xticks(fontsize=paperFontSize)
# plt.yticks(fontsize=paperFontSize)
#
#
# plt.tight_layout()
# plt.show()
# fig.savefig(join(train_data_path, "result",'TrajTest_AbsRMS.pdf'),bbox_inches='tight')
#


jnt_index = np.arange(1,8)
fig, ax = plt.subplots(figsize=(6.5, 4))

plt.rcParams["font.family"] = "Times New Roman"
w = 0.2
space = 0.2
capsize = 2
fontsize = 30
fill_color_list = ['#99C2FF','#66B2FF','#3399FF','tab:blue']

for i in range(len(rel_rms_list)):
    ax.bar(jnt_index+space*(i-2)+w/2, rel_rms_list[i],  width=w,align='center', color=fill_color_list[i], alpha=0.8, ecolor='black', capsize=capsize, label=legend_list[i])

ax.set_xticks(jnt_index)
labels = ['Joint '+str(i+1) for i in range(6)]
labels.append('Avg')
# ax.set_title('Absolute RMSE for Trajectory Test')
ax.yaxis.grid(True)
ax.autoscale(tight=True)
# maxValue = max([max(list) for list in rel_rms_list])
# plt.ylim(0, maxValue*1.2)
ax.margins(y=.1, x=.03)

# Save the figure and show
csfont = {'fontname':'Times New Roman', 'fontsize':paperFontSize}
ax.set_xticklabels(labels, **csfont)
ax.set_ylabel(r'$\epsilon_{rms}$% (N.m)',  **csfont)
a = plt.gca()
a.set_yticklabels(a.get_yticks(), **csfont)
# ax.legend(fontsize=paperFontSize)
font = matplotlib.font_manager.FontProperties(family='Times New Roman',size=paperFontSize)
ax.legend(loc='upper center', prop=font, bbox_to_anchor=(0.5, 1.4),
          fancybox=True, shadow=True, ncol=2)

# plot.legend(loc=2, prop={'size': 6})
plt.xticks(fontsize=paperFontSize)
plt.yticks(fontsize=paperFontSize)
plt.tight_layout()
plt.show()
fig.savefig(join(train_data_path, "result",'NormTest_RelRMS.pdf'),bbox_inches='tight')


print('Avg Absolute RMSE: ',[lst[-1] for lst in abs_rms_list])
print('Avg Relative RMSE: ',[lst[-1] for lst in rel_rms_list])
