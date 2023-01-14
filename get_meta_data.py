# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 10:41:33 2022

@author: ankit
"""

import os
import pandas as pd



def scandirectory(dir, ext):    # dir: str, ext: list
    folders, files = [], []
    
    

    for f in os.scandir(dir):
        
        if f.is_dir():
            folders.append(f.path)
            
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)
                


    for dir in list(folders):
        sf, fi = scandirectory(dir, ext)
        folders.extend(sf)
        files.extend(fi)
        
        
    return folders, files


folders, files = scandirectory(r".\\dataset_car_logo\Train", [".jpg", ".png", ".jpeg"])
label = [(lambda x: x.split('\\')[4])(x) for x in files]
df = pd.DataFrame(zip(files,label), columns=["File_Path", "Class"])
df.to_csv('Train_metadata.csv', index=False)

