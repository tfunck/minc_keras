import json
import os
from os.path import splitext, basename
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(history_fn,model_fn, report_dir):
    with open(history_fn, 'r') as fp: history=json.load(fp)

    training_fn=report_dir +os.sep +splitext(basename(model_fn))[0] +'_training_plot.png'
    epoch_num = range(len(history['dice_loss']))
    # train_error = np.subtract(1, np.array(hist.history['acc']))
    # test_error  = np.subtract(1, np.array(hist.history['val_acc']))
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(epoch_num, np.array(history['dice_loss']), label='Training Accuracy')
    plt.plot(epoch_num, np.array(history['val_dice_loss']), label="Validation Accuracy")
    plt.legend( loc="upper right", ncol=1, prop={'size':18})
    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Accuracy")

    plt.subplot(2,1,2)
    plt.plot(epoch_num, np.array(history['dice_loss']), label='Training Loss')
    plt.plot(epoch_num, np.array(history['val_dice_loss']), label="Validation Loss")
    plt.legend( loc="upper right", ncol=1, prop={'size':18})
    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(training_fn, bbox_inches="tight", dpi=1000, width=2000)
    plt.close()
    print('Model training plot written to ',training_fn )


