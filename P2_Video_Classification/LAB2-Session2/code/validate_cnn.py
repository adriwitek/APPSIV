"""
Classify some images 
"""
import numpy as np
import operator
import random
import glob
import os.path
from data import DataSet
from processor import process_image
from tensorflow.keras.models import load_model
from model_utils import get_last_model_checkpoint
import matplotlib.pyplot as plt




def execute(folder_data='data',nb_images=5, class_limit = 5, seq_length = 5, show_images = False):
    '''Demo of a CNN model in order to make a visual validation.
        Data should be located within checkpoints folder. It will automatilly choose the best trained model
    
        Args:
            folder_data(str): folder in wich the datased is located
            nb_images(int):number of frames to show. The will be selected randomly
            class_limit(int): Number of max. classes tu train the model  can be 1-101 or None
            seq_length(int): sez len in sec for each video
            show_images(bool) = If show images or only ther names(wich contains real class)
     '''
    

    data = DataSet(folder_data,seq_length, class_limit)

    best_model_name = get_last_model_checkpoint(folder_data, nb_images)
    path = os.path.join(folder_data, 'checkpoints',best_model_name)
    print('Loading model: ', path)
    model = load_model(path)

    # Get all our test images.
    images = glob.glob(os.path.join(folder_data, 'test', '**', '*.jpg'))

    for _ in range(nb_images):
        print('-'*80)
        # Get a random row.
        sample = random.randint(0, len(images) - 1)
        image = images[sample]

        # Turn the image into an array.
        print(image)
        image_arr = process_image(image, (299, 299, 3))
        if show_images:
            plt.imshow(image_arr)
            plt.show()
        image_arr = np.expand_dims(image_arr, axis=0)

        # Predict.
        predictions = model.predict(image_arr)

        # Show how much we think it's each one.
        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[0][i]

        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
        
        for i, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            if i > 4:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1

