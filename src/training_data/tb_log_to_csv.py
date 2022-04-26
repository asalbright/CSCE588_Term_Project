###############################################################################
# convert_to_csv.py
#
# A script for converting tensorboard log files to .csv files
# 
# Notes: https://stackoverflow.com/questions/42355122/can-i-export-a-tensorflow-summary-to-csv
#
# Copied: 09/21/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
###############################################################################


import os
import numpy as np
import pandas as pd
from pathlib import Path
import glob

import tkinter
from tkinter import messagebox
from tkinter import filedialog

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import sys
sys.path.append(str(Path.cwd()))
print(sys.path)

# Create gui to say yes/no
def guiYesNo(header, question):
    '''
    Creates a gui to ask the user if yes or no

    Parameters
    ----------
    header : str
        The header of the gui
    question : str
        The question to ask the user

    Returns
    -------
    str
        "yes" if yes, "no" if no
    '''
    root = tkinter.Tk()
    root.withdraw()
    answer = messagebox.askyesno(header, question)
    return answer

def getDirectoryPath(current_path, header=None, verbose=False):
    '''
    This function queues a directory path for the user to select.

    Parameters
    ----------
    current_path : Path
        The current path to the directory
    header : str
        The header of the gui
    verbose : bool 
        If true, prints the path to the directory

    Returns
    -------
    Path
        The path to the directory
    '''
    if header is None:
        header = "Select the Directory Where Data is Stored"

    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    tempdir = filedialog.askdirectory(parent=root, initialdir=current_path, title=header)
    if not os.path.exists(current_path):
        raise ValueError("\nPath does not exits!")
    elif not os.path.isdir(tempdir):
        raise ValueError("\nNot a directory!")
    elif tempdir is None:
        raise ValueError("\nNo directory selected!")
    if len(tempdir) > 0 & verbose == True:
        print("You chose %s" % tempdir)

    return Path(tempdir)

def getDirectoryPaths(current_path, header=None, verbose=False):
    '''
    This function gets the directory paths until the user is finished.

    Parameters
    ----------
    current_path : Path
        The current path to the directory
    header : str
        The header of the gui
    verbose : bool
        If true, prints the path to the directory

    Returns
    -------
    list
        The list of directory paths to the directories
    '''

    paths = []
    paths.append(getDirectoryPath(current_path, header, verbose))
    
    while guiYesNo("Select Yes or No", "Add another directory?"):
        paths.append(getDirectoryPath(current_path, header, verbose))

    return paths
    
def getFiles(file_type, path=None):
    '''
    This function gets all the files of a certain type in a directory.

    Parameters
    ----------
    file_type : str
        The file type to look for
    path : Path
        The path to the directory

    Returns
    -------
    list
        The list of file paths to the files
    path: Path
        The path to the directory if only one file is found
    '''

    if path is None:
        raise ValueError(f'Path not specified for {file_type} file finder.')
    else:
        files = glob.glob(str(path / f'*.{file_type}'))

    if len(files) == 0:
        raise ValueError(f'No {file_type} files found in {path}.')
    else:
        return files 
        
def combineDataInFolder(file_type, path=None, axis=1) -> pd.DataFrame:
    '''
    This function combines all the data in a folder into one dataframe.

    Parameters
    ----------
    file_type : str
        The file type to look for
    path : Path
        The path to the directory, if None, will ask the user

    Returns
    -------
    pd.DataFrame
        The dataframe containing all the data
    '''

    if path is None:
        path = getDirectoryPath(Path.cwd(), "Select path where data exists.")

    files = getFiles(file_type, path)
    data = None
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.replace(' ', '')
        df.columns = df.columns.str.replace('#', '')
        # get the name of the file
        fname = os.path.basename(f)
        # get rid of the file extension
        fname = fname.split('.')[0]
        # add the file name to the df column names
        # df.columns = [str(col) + f'{fname}' for col in df.columns]
        if not data is None:
            data = pd.concat([data, df], axis=axis)
        else:
            data = df
    
    return data

def tabulate_events(dpath, spath):

    final_out = {}
    for dname in os.listdir(dpath):
        print(f"Converting run {dname}",end="")
        ea = EventAccumulator(os.path.join(dpath, dname)).Reload()
        tags = ea.Tags()['scalars']

        out = {}

        for tag in tags:
            tag_values=[]
            wall_time=[]
            steps=[]

            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                wall_time.append(event.wall_time)
                steps.append(event.step)

            # Do you want wall time or not?
            # out[tag]=pd.DataFrame(data=dict(zip(steps,np.array([tag_values,wall_time]).transpose())), columns=steps,index=['value','wall_time'])
            out[tag]=pd.DataFrame(data=dict(zip(steps,np.array([tag_values]).transpose())), columns=steps,index=['value'])

        if len(tags)>0:      
            df= pd.concat(out.values(),keys=out.keys())
            df = df.transpose()
            path = spath / f'{dname}.csv'
            df.to_csv(path)
            print("- Done")
        else:
            print('- Not scalers to write')

        final_out[dname] = df

    return final_out

if __name__ == '__main__':
    '''
    Enter the path to the logs folder below, and this will create a folder, 'plotting_data'
    where .csv files will be saved for all the logs data along with a final concatinated
    'all_results.csv' file.
    '''

    log_paths = getDirectoryPaths(current_path=Path.cwd(), header="Select a path to the logs folder.")
    for path in log_paths:
        print(f"Converting logs in {path}")
        save_path = Path(path).parents[0] / 'training_logs'
        if not os.path.exists(save_path): os.makedirs(save_path)

        steps = tabulate_events(path, save_path)
        path = save_path / "Combined_Data"
        if not os.path.exists(path): os.makedirs(path)

        # Combine all the data from the save_path
        data = combineDataInFolder('csv', save_path)
        # Save the data within the combined data directory
        data.to_csv(path / 'all_results.csv')