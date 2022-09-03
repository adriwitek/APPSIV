"""
Plot a Given a training log file.
"""
import csv
import matplotlib.pyplot as plt
from model_utils import get_last_model_log
import os.path


def execute(nb_classes, training_log =None,folder_data='data', save_fig_name = False):
    '''Plots validation/loss curves of a model during its epcohs in training.
        , model data should be within logs folder. It will automatilly choose the best trained model
    
        Args:
            nb_images(int):number of diferent video classes used
            training_log(string): IF not None, it will contain a manually log file located in a custom route, instead of selecting it in a atomated mode
            folder_data(str): folder in wich the datased is located
            save_fig_name(bool) = If svae images to a external file
     '''
   
    if(training_log is None):#if not name given
        log_name = get_last_model_log(folder_data, nb_classes)
        training_log = os.path.join(folder_data, 'logs',log_name)

        
    print('Opening: ',training_log )
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        accuracies_t = []
        accuracies_v = []
        losses_t = []
        losses_v = []
        #top_5_accuracies = []
        cnn_benchmark = []  # random results
        for epoch, acc, loss, val_acc, val_loss, in reader:
            accuracies_t.append(float(acc))
            accuracies_v.append(float(val_acc))
            losses_t.append(float(loss))
            losses_v.append(float(val_loss))
            cnn_benchmark.append(1./nb_classes)  # random


        #Plot acc curves
        plt.plot(accuracies_t, label = 'Accuracy Training CNN')
        plt.plot(accuracies_v,  label = 'Accuracy Validation CNN')
        plt.plot(cnn_benchmark,  label = 'Accuracy Random')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        if(save_fig_name):
            plt.savefig(  'acc_' + save_fig_name)
        plt.show()

        #plot loss curves
        plt.plot(losses_t, label = 'Loss Training CNN')
        plt.plot(losses_v,  label = 'Loss Validation CNN')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if(save_fig_name):
            plt.savefig(  'loss_' + save_fig_name)
        plt.show()
        

