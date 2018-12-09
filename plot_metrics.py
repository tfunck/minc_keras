import json
import os
from os.path import splitext, basename
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(metric, history_fn,model_fn, report_dir):
    with open(history_fn, 'r') as fp: history=json.load(fp)

    training_fn=report_dir +os.sep +splitext(basename(model_fn))[0] +'_metric_plot.png'
    epoch_num = range(len(history[metric]))
    plt.clf()
    plt.plot(epoch_num, np.array(history[metric]), label='Training Accuracy')
    plt.plot(epoch_num, np.array(history['val_'+metric]), label="Validation Accuracy")
    plt.legend( loc="upper right", ncol=1, prop={'size':8})
    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Metric")
    plt.savefig(training_fn, bbox_inches="tight", dpi=500, width=1000)
    plt.close()

    training_fn=report_dir +os.sep +splitext(basename(model_fn))[0] +'_loss_plot.png'
    plt.clf()
    plt.plot(epoch_num, np.array(history['loss']), label='Training Loss')
    plt.plot(epoch_num, np.array(history['val_loss']), label="Validation Loss")
    plt.legend( loc="upper right", ncol=1, prop={'size':8})
    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(training_fn, bbox_inches="tight", dpi=500, width=1000)
    plt.close()
    print('Model training plot written to ',training_fn )


