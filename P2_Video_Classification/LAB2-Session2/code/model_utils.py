import os, shutil
import numpy as np


'''This libs contains some file automated tasks like get besk checckpoint-log
    after cnn model train or removed files used in demo notebook file.
'''


def get_last_model_checkpoint(folder_data,nb_classes):
    '''Returns the name of best checkpoint avaiable in checkpoints folder'''
    filepath=os.path.join(folder_data, 'checkpoints')
    file_names = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
    file_names_filtered = [f for f in file_names if int(f.split('_')[1]) == nb_classes]
    val_loss = [ float(f.split('-')[1])  for f in file_names_filtered]
    #we take the min. val_loss index
    index = np.argmin(val_loss)
    f_name = file_names_filtered[index]
    return f_name

def get_last_model_log(folder_data, nb_classes):
    '''Returns the name of best checkpoint avaiable in logs folder'''
    filepath=os.path.join(folder_data, 'logs')
    file_names = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
    files = [ f  for f in file_names if int(f.split('_')[1]) == nb_classes]#locate the log by num of classes used
    f_name = files[0]
    return f_name


def _delete_folder_files(folder):
    '''Internal lib aux function to delete folder contents'''  
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e)) 


def delete_all_checkpoints_and_logs(folder_data):
    '''Removes all previus checkpoints and logs generetad during training
        in order to save disk space'''
    folder=os.path.join(folder_data, 'checkpoints')
    print('Deleting checkpoints...')
    _delete_folder_files(folder)

    print('Deleting logs...')
    folder=os.path.join(folder_data, 'logs')
    _delete_folder_files(folder)

    print('Done')


def delete_unzipped_videos(folder_data='data'):
    '''Removes unzipped Train Test Videos in order to save storage space'''
    folder=os.path.join(folder_data, 'train')
    print('Deleting train videos...')
    _delete_folder_files(folder)

    print('Deleting test videos...')
    folder=os.path.join(folder_data, 'test')
    _delete_folder_files(folder)

    print('Done!')