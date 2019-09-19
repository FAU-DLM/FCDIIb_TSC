import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#visualize Augmentation from directory!
def looking_at_augmentation (data_generator, batchsize, path):
    im, label = next(data_generator)
    im = (im - np.min(im))/np.ptp(im) # to normalize all images --> matplotlib only takes pos. values between 0..1 / 0..255 
    imgs = list(im)
    labels = list(label)
    
    fig, ax = plt.subplots(ncols=3, nrows=3)
    fig.subplots_adjust(hspace=0.5)
    plt.suptitle('Augmented Images', fontsize=16)
    plt.figure(num=None, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k')

    for ax in ax.flatten():
        ax.axis('off')

    for i, im  in enumerate(imgs[:batchsize]):
        ax = fig.add_subplot(3,3,i+1)
        ax.imshow(im)
        fig.set_figheight(8)
        fig.set_figwidth(8)

    #fig.tight_layout()
    fig.savefig(path + '/Augmented-Images.png', dpi=300)

#fast plot of training history
def plot_history(history, modelname, path):
    hist_df = pd.DataFrame(history.history)
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))
    axs[0].plot(hist_df.val_categorical_accuracy, lw=5, label='Validation Accuracy')
    axs[0].plot(hist_df.categorical_accuracy, lw=5, label='Training Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].grid()
    axs[0].legend(loc=0)
    axs[1].plot(hist_df.val_loss, lw=5, label='Validation MLogLoss')
    axs[1].plot(hist_df.loss, lw=5, label='Training MLogLoss')
    axs[1].set_ylabel('MLogLoss')
    axs[1].set_xlabel('Epoch')
    axs[1].grid()
    axs[1].legend(loc=0)
    fig.savefig(path + '/History_{}.png' .format(modelname), dpi=300)
    hist_df.to_csv()
    plt.show();

#plotting multiple .csv histories in a single plot
def plotting_x_history (path, plot_title):   
    fileNames = os.listdir(PATH)
    fileNames = [file for file in fileNames if '.csv' in file]
    plt.rcParams.update({'legend.labelspacing':0.25, 'legend.fontsize': 5})
    for file in fileNames:
        df = pd.read_csv(PATH + file)
        print(df.columns)
        dfr = df.drop(['auc_roc', 'epoch', 'loss', 'val_loss', 'lr', 'val_auc_roc', 'val_categorical_accuracy'], 1)
        plt.plot(dfr, label='categorical_accuracy: {}'.format(file))

    plt.legend(loc = 4)
    plt.suptitle('{}'.format(plot_title), fontsize=15)
    plt.show()
    plt.savefig(path + '{}.png'.format(plot_title), dpi=100)

from sklearn.metrics import roc_curve, roc_auc_score, auc
#plotting the receiver operating characteristics --> evaluate performance cutting point vice
def plot_roc(label, predictions, modelname, path):    
    roc_auc_score(label, predictions)
    print('The ROC-Score is: {}' .format(roc_auc_score))

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(label, predictions)
    auc_keras = auc(fpr_keras, tpr_keras)
    #print(auc_keras)

    fig = plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve: {}' .format(auc_keras))
    plt.legend(loc='best')
    fig.savefig(path + '/ROC-Curve_{}.png' .format(modelname), dpi=300) #saving PLOT 
    plt.show()
    
from sklearn.metrics import confusion_matrix
import itertools
# Source: Scikit Learn website
# http://scikit-learn.org/stable/auto_examples/
# model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-
# selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
#plotting correctly classified images: https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
def plot_correct(vals, y_pred, y_label, modelname, path):
    correct = np.where(y_pred==y_label)[0]
    print ("Found %d correct labels" % len(correct))


    fig, ax = plt.subplots(ncols=3, nrows=3)
    fig.subplots_adjust(hspace=0.5)
    plt.suptitle('Correct Images', fontsize=16)
    plt.figure(num=None, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k')

    for ax in ax.flatten():
        ax.axis('off')

    for i, correct in enumerate(correct[:9]):
        ax = fig.add_subplot(3,3,i+1)
        ax.imshow(vals[correct])
        ax.set_title("Predicted {}, Class {}".format(y_pred[correct], y_label[correct]), fontsize=10)
        fig.set_figheight(8)
        fig.set_figwidth(8)

    fig.savefig(path + '/Correct_Images_{}.png' .format(modelname), dpi=100) #saving PLOT 

#Plotting incorrectly classified
def plot_incorrect(vals, y_pred, y_label, modelname, path):
    incorrect = np.where(y_pred!=y_label)[0]
    print ("Found %d incorrect labels" % len(incorrect))


    fig, ax = plt.subplots(ncols=3, nrows=3)
    fig.subplots_adjust(hspace=1)
    plt.suptitle('Incorrect Images', fontsize=16)
    plt.figure(num=None, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k')

    for ax in ax.flatten():
        ax.axis('off')

    for i, incorrect in enumerate(incorrect[:9]):
        ax = fig.add_subplot(3,3,i+1)
        ax.imshow(vals[incorrect])
        ax.set_title("Predicted {}, Class {}".format(y_pred[incorrect], y_label[incorrect]), fontsize=10)
        fig.set_figheight(8)
        fig.set_figwidth(8)

    fig.savefig(path + '/Incorrect_Images_{}.png' .format(modelname), dpi=100) #saving PLOT 
    
def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper