import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ['Tensor addition', 'Tensor transpose', 'Tensor product', 'Gradient', 'Tensor inverse', 'Primary invariants',
          'Principal invariants', 'Eigenvalues and eigenvectors', 'Spectral decomposition', 'Tensor rank']
cf = [491*17.5, 203*17.5, 54*17.5, 54*17.5, 313*17.5, 732*17.5, 832*17.5, 181*17.5, 173*17.5, 64*17.5]
df = [1363, 278, 78, 63, 57, 222, 210, 76, 79, 56]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, cf, width, label='Control-flow')
rects2 = ax.bar(x + width/2, df, width, label='Dataflow')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('execution time in ms scaled with transistor count ratio')
ax.set_title('Performance evaluation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
plt.xticks(rotation=60)
fig.tight_layout()

plt.show()