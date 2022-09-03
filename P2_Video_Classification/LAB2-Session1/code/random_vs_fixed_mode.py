"""
Try to "classify" samples based on random chance vs always guessing
the same category.
"""
import random
from data import DataSet


##ApplyEyeMakeup, ApplyLipstick, Archery, BabyCrawling, BalanceBeam
#fix_mode = 'ApplyEyeMakeup'
def apply_random_vs_fixed_mode(folder_data='data',fix_mode = 'ApplyEyeMakeup', verbosse = True,  class_limit = 5,   seq_length = 5  ):
    '''Classificates videos located in folder_data route. Random algortihm and fixed one are applied. It computes accuracy in order to 
            show classifiers performance. 
        within logs folder
    
        Args:
            folder_data(str): folder in wich the datased is located
            verbosse(bool): It will print acc for random and fixed model. Useful to be set to False it function will be executed N number of times.
            class_limit(int): Number of max. classes tu train the model  can be 1-101 or None
            seq_length(int): sez len in sec for each video
        Returns:
            accuracy form random and fixed algs.
    '''
     
    data = DataSet(folder_data,seq_length,class_limit)
    nb_classes = len(data.classes)
    
    # Try a random guess.
    nb_random_matched = 0
    nb_mode_matched = 0
    for item in data.data:
        choice = random.choice(data.classes)
        actual = item[1]
    
        if choice == actual:
            nb_random_matched += 1
    
        if actual == fix_mode:
            nb_mode_matched += 1
    
    random_accuracy = nb_random_matched / len(data.data)
    mode_accuracy = nb_mode_matched / len(data.data)
    if(verbosse):
        print("Randomly matched %.2f%%" % (random_accuracy * 100))
        print("Mode matched %.2f%%" % (mode_accuracy * 100))
    return random_accuracy, mode_accuracy