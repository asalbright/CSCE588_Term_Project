###############################################################################
# functions.py
#
# Contains useful functions for the pogo-stick  
#
# Created: 02/23/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
# Modified:
# * 10/02/2021 - Updated saving data so it is all saved in one file
###############################################################################
# # ********************************************
# Author: Andrew Albright
# Date: 03/31/2021

# File containing useful functions

# ********************************************

import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pandas as pd
import os
import sys
import tkinter
from tkinter import messagebox
from tkinter import filedialog

def getFilePaths(current_path, header=None, verbose=False):
    '''
    This function gets the file paths until the user is finished.

    Parameters
    ----------
    current_path : Path
        The current path to the file
    header : str
        The header of the gui
    verbose : bool
        If true, prints the path to the file


    Returns
    -------
    list
        The list of file paths to the files
    '''

    paths = []
    paths.append(getFilePath(current_path, header, verbose))
    
    while guiYesNo("Select Yes or No", "Add another file?"):
        paths.append(getFilePath(current_path, header, verbose))

    return paths

def getFilePath(current_path, header=None, verbose=False):
    '''
    This function queues a file path for the user to select.

    Parameters
    ----------
    current_path : Path
        The current path to the file
    header : str
        The header of the gui
    verbose : bool
        If true, prints the path to the file

    Returns
    -------
    Path
        The path to the file
    '''
    if header is None:
        header = "Select the File"

    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    tempfile = filedialog.askopenfilename(parent=root, initialdir=current_path, title=header)
    if not os.path.exists(current_path):
        raise ValueError("\nPath does not exits!")
    elif not os.path.isfile(tempfile):
        raise ValueError("\nNot a file!")
    elif tempfile is None:
        raise ValueError("\nNo file selected!")
    if len(tempfile) > 0 & verbose == True:
        print("You chose %s" % tempfile)

    return Path(tempfile)

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

def getListOfDirectoryPaths(path, verbose=False):
    '''
    This function gets all the subdirectories in a directory.

    Parameters
    ----------
    path : Path
        The path to the directory

    Returns
    -------
    list
        The list of subdirectories in the directory
    '''

    if not os.path.exists(path):
        raise ValueError("\nPath does not exits!")
    elif not os.path.isdir(path):
        raise ValueError("\nNot a directory!")
    elif path is None:
        raise ValueError("\nNo directory selected!")
    if os.path.exists(path) & verbose == True:
        print("You chose %s" % path)

    return [x[0] for x in os.walk(path)]

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

def getFileNames(file_type, path=None)->list:
    '''
    This function gets all the file names of a certain type in a directory.

    Parameters
    ----------  
    file_type : str
        The file type to look for
    path : Path
        The path to the directory

    Returns
    -------
    list
        The list of file names to the files
    str
        The name of the file if only one file is found
    '''

    if path is None:
        raise ValueError(f'Path not specified for {file_type} name finder.')
    else:
        files = glob.glob(str(path / f'*.{file_type}'))
        print(f'Number of {file_type} files found: {len(files)}')
        file_names = []
        for f in files:
            file_names.append(os.path.basename(f).split(sep='.')[0])
        
        if len(file_names) == 0:
            raise ValueError(f'No {file_type} files found in {path}.')
        else:
            return file_names
            
def readCSVFile(f)->list:
    '''
    This function reads in a list of csv files and returns a list of dataframes.

    Parameters
    ----------
    files : list
        The list of csv files to read

    Returns
    -------
    list
        The list of dataframes
    '''

    
    df = pd.read_csv(f)
    df.columns = df.columns.str.replace(' ', '')
    df.columns = df.columns.str.replace('#', '')
    
    return df

def readCSVFiles(files)->list:
    '''
    This function reads in a list of csv files and returns a list of dataframes.

    Parameters
    ----------
    files : list
        The list of csv files to read

    Returns
    -------
    list
        The list of dataframes
    '''

    data = []
    for f in files:
        data.append(readCSVFile(f))
    return data

def combineDataInFolder(file_type, path=None, axis=0) -> pd.DataFrame:
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
        if not data is None:
            data = pd.concat([data, df], axis=axis)
        else:
            data = df
    
    return data

def parseDataFrame(df, unique_headers):
    '''
    This function parses a dataframe into a dictionary of dataframes.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to parse
    unique_headers : list
        The list of unique headers to parse the dataframe by

    Returns
    -------
    dict
        The dictionary of dataframes
    '''

    # Make sure all columns are strings without following characters
    df.columns = df.columns.str.replace(' ', '')
    df.columns = df.columns.str.replace('#', '')

    data_frames = {}
    for header in unique_headers:
        temp_df = None
        for col in df.columns:
            if header in col:
                if not temp_df is None:
                    temp_df = pd.concat([temp_df, df[col]], axis=1)
                else:
                    temp_df = df[col]
        # append to dict of dataframes
        data_frames[header] = temp_df
    
    return data_frames

def dfAverageStd(df):
    '''
    This function calculates the average and standard deviation of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to calculate the average and standard deviation of

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        The dataframes containing the average and standard deviation
    '''
    
    return df.mean(axis=1), df.std(axis=1)

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

def scrape_files_by_name(fpath, ftype, fname, save_path=None, save_name=None):
    '''
    This function scrapes files from all child directories of a path by name.

    Parameters
    ----------
    fpath : Path
        The path to the directory
    ftype : str
        The file type to look for
    fname : str
        The name of the file to look for
    save_path : Path
        The path to save the file to

    Returns
    -------
    list
        The list of file paths to the files
    '''
    import shutil
    import random

    # Get all the files of a certian type within all child directories
    files = []
    file_names = []
    for root, dirs, files_in_dir in os.walk(fpath):
        for file in files_in_dir:
            if file.endswith(ftype) and fname in file:
                # Add random number to file name to avoid overwriting
                # TODO: This is a hack, fix this. This might make the same number twice
                file_name = file.split(sep='.')[0] + '_' + str(random.randint(0, 100)) + '.' + file.split(sep='.')[1]
                files.append(os.path.join(root, file))
                file_names.append(file_name)

    # Save files to a new directory
    if save_name is None:
        save_name = "Scraped Files"
    if save_path is None:
        save_path = fpath / save_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(files)):
        shutil.copy(files[i], save_path / file_names[i])

    return files

if __name__ == "__main__":
    dirs = getDirectoryPaths(Path.cwd(), "Select path where models are saved")
    for _dir in dirs:
        scrape_files_by_name(_dir, 'zip', 'final', save_name='eval_models')