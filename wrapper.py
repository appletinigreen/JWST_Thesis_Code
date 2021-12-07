import Templates
from Templates import templates
import glob
import os
from astropy.io import ascii
import shutil
import numpy as np


def plot_choice(choice):
    """
        Choose which of the data set to plot, special 8, JWST, all gals etc.
    """
    if choice == "JWST":
        jwst_names = ['J1107', 'J1219', 'J1506', 'J1613', 'J2118']
        jwst_redshifts = [0.467, 0.451, 0.608, 0.449, 0.459]
        return jwst_names, jwst_redshifts

    elif choice == "Special":
        # These are for the original 8 galaxies of interest
        special_names = ['J0826', 'J0905', 'J0944', 'J1107', 'J1219', 'J1341', 'J1613', 'J2118']
        special_redshifts = []
        # This is for the best/worst SED pdf
        special_temps = {'J0826': [templates[1], templates[7]], 'J0905': [templates[1], templates[7]],
                         'J0944': [templates[1], templates[7]],
                         'J1107': [templates[1], templates[10]], 'J1219': [templates[1], templates[3]],
                         'J1341': [templates[1], templates[7]], 'J1613': [templates[0], templates[3]],
                         'J2118': [templates[1], templates[8]]}

        return special_names, special_redshifts, special_temps


def plot_seds(names_subset: list, redshifts):
    '''
    :param names_subset: list of galaxy names
    :return: doesn't return any data but creates pdf of each gal with corresponding templates
    '''

    for i in range(0, len(names)):
        if names[i] in names_subset:
            print("name", names[i])
            print('redshifts[i]', redshifts[i])
            # IR_SFRs returns two variables, one avg SFR and the other Stdev SFR
            # What if tems doesnt equal files?
            names_subset_sfr, names_subset_stdv_sfr, sed_list = Templates.IR_SFRs(redshifts[i], names[i], '_Analyze', tems=brown_tems)  # Does this need to be an assignment statement?
            Templates.IR_SFRs(redshifts[i], names[i], '_Composite', calc_SFR=True, tems=templates)
            # redshift_list.append(redshifts[i])
            Templates.plot_all(names[i]+"All", sed_list)

            # print(sed)
            # print(brown_sed)


projpath = os.getcwd() + '/Brown2014/An_Atlas_of_Galaxy_SEDs/An_Atlas_of_Galaxy_SEDs/Converted/'
brown_tems = glob.glob(projpath + '*.dat')
print(f"type bronwn tems:{type(brown_tems)}")

# can update these from Table 2 of Petter et al. 2020
# added J1558 for kicks
names = ['J0106', 'J0826', 'J0827', 'J0905', 'J0908', 'J0944', 'J1039', 'J1107', 'J1125', 'J1219', 'J1229', 'J1232',
         'J1248', 'J1341', 'J1506', 'J1613', 'J2116', 'J2118', 'J2140', 'J2256', 'J1558']
redshifts = np.array(
    [0.454, 0.603, 0.681, 0.711, 0.502, 0.514, 0.634, 0.467, 0.519, 0.451, 0.614, 0.401, 0.632, 0.661, 0.437, 0.449,
     0.728, 0.459, 0.751, 0.727, .403])

jwst_names,jwst_redshifts = plot_choice('JWST')

boundaries = {'J1107': 1.9, 'J1219': 2.2, 'J1506': 3, 'J1613': 3, 'J2118': 2}
most_weird= {'J1107':["IRAS_08572+3915","Mrk_0475"],'J1219':["UGCA_219","IRAS_08572+3915"],'J1506': ["IC_4553","UM_461"],
            'J1613':["UGCA_219","IC_4553"],'J2118':["NGC_1068","III_Zw_035"]}
top_5= {'J1107':["IRAS_08572+3915","NGC_1068","NGC_7714","NGC_7674","Mrk_33"],'J1219':["NGC_1614","IC_0860","NGC_1275","UGC_08335_SE","NGC_6240"],
             'J1506': ["Mrk_0930","IC_4553","UGCA_166","UGC_06850","Mrk_1450"],
            'J1613':["NGC_1275","UGC_08696","NGC_1614","IC_0860","II_Zw_096"],'J2118':["Mrk_0475","NGC_2623","Mrk_1490","NGC_1068","Mrk_33"]}

SNR_max_bounds = [17.5,22.5,25,227.5,30]
files = glob.glob(projpath + '*.dat')
new_proj_path = projpath + "Noisy/"


np.random.seed(39032)

def make_chi_tables():
    """
    No parameters because using global scope :(
    :return: makes files of chi values arranged from smallest to largest
    """
    projpath = os.getcwd()
    new_projpath = projpath + "/ChiTables/"
    file_name = f"{names[i]}_Chis"
    tem_names = [b[130:len(b) - 14] for b in brown_tems]
    zipped = zip(tem_names, chis)
    zipped = list(zipped)
    res = sorted(zipped, key=lambda x: x[1])
    new_tems, new_chis = zip(*res)
    new_chis = [round(c, 4) for c in new_chis]
    ascii.write([list(new_tems), new_chis], file_name,
                names=['Column 1: Template Name', 'Column 2: Chi Values'], overwrite=True)
    # ascii.write([micro_wave, mJy], new_name,names=['Column 1: Rest Wavelength (microns)', 'Column 2: Flux/millijanskys(10^-26* ergs/s/cm^2/Hz)'])

    if os.path.exists(new_projpath):
        file_path = os.getcwd() + "/" + file_name
        shutil.move(file_path, new_projpath)


for i in range(0, len(names)):
    if names[i] in jwst_names:
        print("gal name:", names[i])
        print('redshifts:', redshifts[i])
        chis = Templates.IR_Fluxes(redshifts[i], names[i], '_BrownMoreNoisy', tems=brown_tems, noisy=True)
        Templates.plot_most_unique(redshifts[i], names[i], most_weird[names[i]], projpath,brown_tems)
        Templates.plot_all(redshifts[i], names[i], "_FindingWeirdOnes", chis, boundaries[names[i]], brown_tems)


Templates.all_gals_color_color(jwst_names)
