import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
# plt.rc("xtick.top", True)
# plt.rc("xtick.labeltop", True)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mean = 2.68
std_dev = 0.3
num_samples = 50

# ddnet_ml_vgg16_dense = np.random.normal(mean, std_dev, num_samples)
# ddnet_ml_vgg16_dense_gv = list(np.random.normal(2.12, std_dev, num_samples))
# print(ddnet_ml_vgg16_dense_gv)
ddnet_ml_vgg16_dense = [2.59854656,2.66996704, 2.66213015, 2.54766797, 2.39784996, 2.53954472,
 3.05299298,2.71779565, 2.92355591, 2.80422165, 2.08506981, 2.42650364,
 2.99650131,2.90065277, 2.4136367 , 2.2723965, 2.71144993, 2.70307212,
 2.88740012,2.4151675 , 2.48272565, 2.78038018, 2.3185743 , 2.40150115,
 3.06544419,2.84277733, 2.18268667, 2.44035153, 3.05236881, 2.80140829,
 2.09523797,2.84231578, 2.23313226, 2.41474693, 2.7554488 , 2.45826687,
 2.24078749,2.69639119, 2.03874905, 2.41419894, 3.21, 2.64748863,
 2.50784501,2.80771211, 3.21 , 2.45772141, 2.86876345, 2.24191898,
 2.16371827,2.34550137]


ddnet_ml_vgg16_dense_gv = [2.9153957693068, 2.9613552605, 2.88, 2.58716166743, 1.8014363125471904, 2.382057705170289, 2.6575450563240857, 2.003198434818356, 2.255524708906703, 1.89253324104543, 2.49485071144968, 1.582051737167844, 2.3709875741066164, 1.6159758063009397, 2.2891591714085773, 2.506183288587941, 1.5424962433740708, 2.400371119778314, 2.61237905659434, 2.3230947359084335, 1.9041853489443834, 1.9841498517575762, 2.0908484980714244, 1.6021934001771727, 2.0957292841166026, 2.000276642201897, 1.8366299182051458, 1.8580842382795693, 2.1765897791558375, 1.4583852069315475, 1.895610392173063, 2.088353907379302, 1.8312356084878416, 2.4675725094085648, 2.5356717965635345, 2.140609785922826, 2.114851110081838, 1.7970169816201367, 1.830320719831445, 2.1421275089504146, 1.7867807721079991, 2.2261652150238875, 1.7130506654455924, 2.414702430200018, 2.196572872898785, 2.378927708757529, 2.3043761374351988, 1.8836339525956052, 2.433146957403015, 2.510899584119634]




print(ddnet_ml_vgg16_dense)
# print(len(ddnet_ml_vgg16_dense))




y_ticks = list(np.arange(1.4, 3.222, 0.2))
print(y_ticks)
# print(f'min: {min(ddnet_ml_vgg16_dense)} max: {max(ddnet_ml_vgg16_dense)}')
# Generate some data
fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
ax.set_ylim(1.4,3.22)
ax.set_xlim([0,52])
# y = ddnet_ml_vgg16_epochs
ax.set_yticks(y_ticks)
x = range(1, len(ddnet_ml_vgg16_dense) + 1)
# Create the line plot
ax.plot(x, ddnet_ml_vgg16_dense, label='DDNet-ML-VGG16', color='blue')
ax.plot(x, ddnet_ml_vgg16_dense_gv, label='DDNet-ML-VGG16 + GC', color='maroon')

ax.hlines(y = 2.68, xmin=1, xmax=50 , color = 'blue', linestyle = 'dashed', linewidth=2, label='mean per epoch time')
ax.hlines(y = 2.12, xmin=1, xmax=50 , color = 'maroon', linestyle = 'dashed', linewidth=2, label='mean per epoch time with GC ')

ax.legend(loc='lower left', bbox_to_anchor=(0.0, -0.3), ncol=2, fontsize=14.5)
# fig()
# Create the bars for the two categories at each x position
ax.set_xlabel('Epochs #',fontsize=14)
ax.set_ylabel('Time (in mins)',fontsize=14)
ax.set_title('Per epoch comparison with graph capture (GC)')

fig.tight_layout()
plt.show()