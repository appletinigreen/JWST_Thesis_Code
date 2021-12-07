import numpy as np
import os
import glob
from astropy.io import ascii
import shutil


''''
Purpose of this script is to convert the Brown templates into .dat files and to change units on columns
 wavelength from angstrom to microns and f_lambda to f_nu,
 this is pretty much a one-use script since
they're saved after being converted

edit, 10/18/21
Purpose of this script is to convert Brown templates into needed units. Last year, 2020, we needed to convert from f_lambda 
to f_nu and from angstroms to microns. For 2021 research, we need to convert from angstrons to microns and from f_lambda to millijanskies.
We can repurpose the converted files to make this easier. 
'''


projpath = os.getcwd() + "/Brown2014/An_Atlas_of_Galaxy_SEDs/An_Atlas_of_Galaxy_SEDs/"
files = glob.glob(projpath + '*.dat')
new_projpath = projpath + "Converted1/"
etc_projpath = projpath + "ETC/"


def remove_files(file_extension, path=None):
    if path == None:
        r_proj_path = os.getcwd() + "/"
        r_files = glob.glob(r_proj_path + file_extension)
    else:
        r_proj_path = os.getcwd() + path + "/"
        r_files = glob.glob(r_proj_path + file_extension)
    for r in r_files:
        os.remove(r)

for i in range(0, len(files)):

    table = ascii.read(files[i])

    wave = np.array(table['col1'])
    micro_wave = wave * 1e-4 #Converts to Microns from Angstroms
    flam = np.array(table['col2'])
    flux = wave * flam
    fnu = (flam * wave ** 2) / 3e18  # * 1e46
    mJy= fnu*10e26 #convert to mJy
    #con_data = [wave, fnu, table[2]]
    broken_path = files[i].split("/")
    old_filename = broken_path[len(broken_path) - 1].split(".") #removes file type
    new_name = old_filename[0] + "_conv.dat"
    print(f"new_name: {new_name}")
    ascii.write([micro_wave,fnu], new_name, names=['Column 1: Rest Wavelength (microns)','Column 2: Flux/f_nu(ergs/s/cm^2/Hz)'])
    ascii.close()
    #ascii.write([micro_wave, mJy], new_name,names=['Column 1: Rest Wavelength (microns)', 'Column 2: Flux/millijanskys(10^-26* ergs/s/cm^2/Hz)'])
    if os.path.exists(new_projpath):
        file_path = os.getcwd() + "/" + new_name
        shutil.move(file_path, new_projpath)