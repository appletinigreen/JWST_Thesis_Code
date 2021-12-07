#
# Author: Grayson Petter
# Templates.py
# Functions to redshift, interpolate, and integrate IR templates in order to simulate observed colors & luminosities

import numpy as np
from astropy import units as u
import math
import pickle
import WISE
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
import glob
import importlib
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chisquare
from astropy.io import ascii
import random
import shutil

importlib.reload(WISE)
# reload(WISE)

# speed of light
c = 299792458.  # m/s

# path to project

projpath = os.getcwd() + '/'  # '../'#'/Users/graysonpetter/Desktop/IRSFRs/'
print("projpath:", projpath)
# set cosmology for calculating luminosity distance
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Read in all templates
templates = glob.glob(projpath + 'Comprehensive_library/SFG*.txt')
templates.extend(
    glob.glob(projpath + 'Comprehensive_library/Comp*.txt'))
templates.extend(
    glob.glob(projpath + 'Comprehensive_library/AGN*.txt'))

# read in the W3 & W4 bandpasses
wise_bandpasses_3_4 = sorted(glob.glob(projpath + 'bandpass/*.txt'))[2:4]


def redshift_spectrum(z, template, trim):
    '''
    redshift a template SED (wavelengths only)
    :param z: redshift
    :param template: template
    :param trim:
    :return:
    '''
    # t = pd.read_csv(template, delim_whitespace=True, engine='python', header=None, skiprows=3)
    t = pd.read_csv(template, delim_whitespace=True, engine='python', header=None, skiprows=9)

    # read off wavelengths and luminosities from template
    wavelengths = np.array(t.iloc[:, 0])
    wavelengths = wavelengths.astype(float)
    Lums = np.array(t.iloc[:, 1])

    # cut template down to 8-1000 microns (TIR) definition
    if trim:
        spec_range = np.where((wavelengths >= 8.) & (wavelengths <= 1000.))[0]
        wavelengths = wavelengths[spec_range]
        Lums = Lums[spec_range]

    # redshift wavelengths
    shifted_len = np.array(wavelengths) * (1 + z)

    # get luminosity at 12 & 22 micron in observed frame
    twelve_mu = (np.abs(shifted_len - 12)).argmin()
    twenty_two_mu = (np.abs(shifted_len - 22)).argmin()

    return [wavelengths, Lums, Lums[twelve_mu], Lums[twenty_two_mu], shifted_len]


def interpolate_spec(shifted_spec, model):
    '''
    linearly interpolate the SED in frequency space to make integration simple
    convert wavelengths in microns to frequencies in Hz, 10**6 converts microns to meters
    :param shifted_spec:
    :param model:
    :return:nu,lum
    '''
    nus = (10 ** 6) * c / (shifted_spec[0])

    # reverse lists so frequencies go from low to high for simplicity
    reversed_nus = np.flipud(nus).flatten()
    # also reverse luminosities
    reversed_lums = np.flipud(shifted_spec[1])

    # calculate constant frequency interval to interpolate on
    if model:
        step = reversed_nus[1] - reversed_nus[0]
        dx = round(step, -(len(str(int(step))) - 1))
    else:
        dx = 10000000000

    # find smallest factor of dx Hz greater than the smallest frequency in the list to start the interpolation
    start = (reversed_nus[0] + int(dx)) - (reversed_nus[0] % int(dx))

    # range of frequency across entire template
    span = reversed_nus[len(reversed_nus) - 1] - reversed_nus[0]
    # number of frequency intervals to interpolate on
    chunks = int(math.floor(span / dx))

    # lists for interpolated values
    new_nus, new_lums = [], []
    current_nu = start

    # linearly interpolate to frequencies in dx Hz steps
    for x in range(chunks):
        new_nus.append(current_nu)
        new_lums.append(np.interp(current_nu, reversed_nus, reversed_lums))
        current_nu += dx

    return new_nus, new_lums


def integrate_spectrum(freqs, Ls) -> np.array:
    '''
    integrate spectrum using trapezoidal method
    :param freqs:
    :param Ls:
    :return:integrated spectrum
    '''
    return np.trapz(y=Ls, x=freqs)


def simulate_wise_fluxes_for_colors(z: float, tems: str, bands, csv):
    '''

    :param z: redshift
    :param tems: templates
    :param bands:
    :param csv:
    :return: simulated wise fluxes
    '''
    tot_mag_list, template_names = [], []
    # iterate through templates
    for tem in tems:
        # redshift template
        red_spec = redshift_spectrum(z, tem, False)
        red_waves = np.array(red_spec[4])
        lumi = np.array(red_spec[1])

        normalized = []

        # iterate through WISE bands
        for y in range(len(bands)):
            if csv:
                band = pd.read_csv(bands[y], header=None, engine='python')
            else:
                band = pd.read_csv(bands[y], header=None, delim_whitespace=True, engine='python')
            bandwaves = np.array(band.iloc[:, 0])
            band_response = np.array(band.iloc[:, 1])

            # trim template to same wavelength range as WISE band
            cut = np.where((red_waves >= np.min(bandwaves)) & (red_waves <= np.max(bandwaves)))[0]
            trimmed_y = red_waves[cut]
            trimmed_L = lumi[cut]

            # interpolate template to band wavelengths, multiply by the response at that wavelength
            inter_lum = []
            for j in range(len(bandwaves)):
                inter_lum.append(band_response[j] * (np.interp(bandwaves[j], trimmed_y, trimmed_L)))

            # crude method
            """sum_lum = np.sum(np.array(inter_lum))
            sum_waves = np.sum(np.array(band_response))
            normalized.append(sum_lum/sum_waves)"""

            # integrate template multiplied by response function
            spectrum = [bandwaves, inter_lum]
            interped_again = interpolate_spec(spectrum, True)
            wise_lums = integrate_spectrum(interped_again[0], interped_again[1])

            # integrate wise band
            band_spectrum = [bandwaves, band_response]
            interped_band = interpolate_spec(band_spectrum, True)
            integrated_band = integrate_spectrum(interped_band[0], interped_band[1])
            # divide two
            normalized.append(wise_lums / integrated_band)

        tot_mag_list.append(normalized)

        template_names.append(tem.split('.txt')[0].split('/')[8])

    return tot_mag_list, template_names


# simulate observed WISE fluxes by integrating templates over WISE bandpasses
def simulate_wise_fluxes(z, tem, bands, csv):
    '''
    :param z: redshift
    :param tem: specific template
    :param bands: ?
    :param csv: Boolean,True when file = csv, false when not
    :return: simulated WISE fluxes
    '''
    # redshift template
    red_spec = redshift_spectrum(z, tem, False)

    shifted_wavelengths = np.array(red_spec[4])
    lumi = np.array(red_spec[1])
    normalized = []

    # iterate through WISE bands
    for y in range(len(bands)):
        if csv:
            band = pd.read_csv(bands[y], header=None, engine='python')
        else:
            band = pd.read_csv(bands[y], header=None, delim_whitespace=True, engine='python')
        # bandpass wavelength list
        bandwaves = np.array(band.iloc[:, 0])
        # bandpass response function values
        band_response = np.array(band.iloc[:, 1])
        # convolve wavelength with response function per Greg Rudnick's suggestion to account for the fact
        # that the WISE detectors are photon counting devices, while the templates are energy templates
        band_convolved = np.multiply(bandwaves, band_response)

        # trim template to same wavelength range as WISE band
        cut = np.where((shifted_wavelengths >= np.min(bandwaves)) & (shifted_wavelengths <= np.max(bandwaves)))[0]

        trimmed_y = shifted_wavelengths[cut]
        trimmed_L = lumi[cut]

        # interpolate template to wavelengths in the bandpass list, multiply by the convolved response function
        # at that wavelength
        inter_lum = []

        for i in range(len(bandwaves)):
            inter_lum.append(band_convolved[i] * (np.interp(bandwaves[i], trimmed_y, trimmed_L)))

        # integrate template multiplied by response function
        spectrum = [bandwaves, inter_lum]
        interped_again = interpolate_spec(spectrum, True)
        wise_lums = integrate_spectrum(interped_again[0], interped_again[1])

        # integrate wise response function to divide out
        band_spectrum = [bandwaves, band_convolved]
        interped_band = interpolate_spec(band_spectrum, True)
        integrated_band = integrate_spectrum(interped_band[0], interped_band[1])
        # divide two
        normalized.append(wise_lums / integrated_band)
    return normalized


# integrate the IR templates and write out the total IR luminosity so it can be recalled quickly without doing an
# integration each time
def writetotals():
    '''
    :return: creates file with total IR luminosity for each template in templates
    '''
    totlist = []
    # for each template
    for x in range(len(templates)):
        # call redshift function, but don't actually redshift, just trim to 8-1000 microns
        shifted_spectrum = redshift_spectrum(0, templates[x], True)
        # interpolate the template
        interped_spectrum = interpolate_spec(shifted_spectrum, False)
        # integrate template from 8-1000 micron
        total_ir = integrate_spectrum(interped_spectrum[0], interped_spectrum[1])
        totlist.append(total_ir)
    # write out the integral totals in a file
    with open(projpath + 'integrations/kirk.txt', 'wb') as fb:
        pickle.dump(totlist, fb)


writetotals()


def murphyIRSFR(L_IR):
    '''
    calculate SFRs using calibration given in Murphy+11
    :param L_IR:
    :return:
    '''
    L_IR = L_IR.to('erg/s').value
    SFR = 3.88e-44 * L_IR
    return SFR


def calc_ratio(measured, measured_errs, simulated, params:int):
    '''
    :param measured: measured values, flux, lums etc.
    :param measured_errs: errs associated w/ measured values
    :param simulated: simulated model data
    :param params: tells ratio to calculate w/ 1 or 2 quantities, W3 & W4 or just W3
    :return: ratio to scale template with
    '''
    if params == 2:
        '''
        print(f"meausured[0]: {measured[0]}, type = {type(measured[0])}")
        print(f"meausured[1]: {measured[1]}, type = {type(measured[1])}")
        print(f"measured_errs[0]: {measured_errs[0]}, type = {type(measured_errs[0])}")
        print(f"measured_errs[1]: {measured_errs[1]}, type = {type(measured_errs[1])}")
        print(f"simulated[0]: {simulated[0]}, type = {type(simulated[0])}")
        print(f"simulated[1]: {simulated[1]}, type = {type(simulated[1])}")
        '''

        ratio = (measured[0] * simulated[0] / (measured_errs[0]) ** 2 + measured[1] * simulated[
            1] / (measured_errs[1]) ** 2) / ((simulated[0] / measured_errs[0]) ** 2 + (
                simulated[1] / measured_errs[1]) ** 2)
    else:
        ratio = float(measured[0] / simulated[0])
    return ratio


def apply_template_fit(w_four_good:bool, w3_lum:np.array, w3_lum_err:np.array, w4_lum:np.array, w4_lum_err:np.array, z:float, tem):
    '''
      :param w_four_good: Boolean which tells if W4 is good or not
      :param w3_lum: W3 luminosities
      :param w3_lum_err: W3 lum errors
      :param w4_lum:  W4 luminosities
      :param w4_lum_err: W4 lum errors
      :param z: redshift
      :param tem: template
      :return: Checks if W4 is good (!=NaN), if it is then  makes array of luminosities including
        W4, if not, then makes arrays using only W3'
    '''
    if w_four_good:
        # join W3 & W4 observed luminosities
        measured_lums = np.array([float(w3_lum.value), float(w4_lum.value)])
        measured_lum_errs = np.array([float(w3_lum_err.value), float(w4_lum_err.value)])

        # simulate a WISE flux by integrating the template over the response curves
        simulated = np.array(simulate_wise_fluxes(z, tem, wise_bandpasses_3_4, False))

        # perform least squares fit of observed W3, W4 luminosities to the simulated W3, W4 luminosities
        # this gives a normalization parameter which can be multiplied by the template TIR luminosity to give an
        # estimate of the intrinsic luminosity of the source
        # note: this is equation 3 from this paper: https://aip.scitation.org/doi/pdf/10.1063/1.168428
        # can use flux instead of lum , need to figure out how simulated piece works
        l_ratio = (measured_lums[0] * simulated[0] / (measured_lum_errs[0]) ** 2 + measured_lums[1] * simulated[
            1] / (measured_lum_errs[1]) ** 2) / ((simulated[0] / measured_lum_errs[0]) ** 2 + (
                simulated[1] / measured_lum_errs[1]) ** 2)
        return measured_lums, measured_lum_errs, simulated, l_ratio

        # if there is no W4 data, simply take ratio of template and observed luminosity at W3
    else:
        measured_lums = np.array([float(w3_lum.value), 0.])
        measured_lum_errs = np.array([float(w3_lum_err.value), 0.])

        simulated = np.array(simulate_wise_fluxes(z, tem, wise_bandpasses_3_4, False))

        # l_ratio = float(w3_lum.value/tem_lum[2])
        l_ratio = float(w3_lum.value / simulated[0])

        return measured_lums, measured_lum_errs, simulated, l_ratio


def apply_flux_fit(w_four_good:bool, w3_val, w3_val_err, w4_val, w4_val_err, z, tem):
    '''
    Checks if W4 is good (!=NaN), if it is then  makes array of luminosities including
        W4, if not, then makes arrays using only W3
    :param w_four_good:
    :param w3_val:
    :param w3_val_err:
    :param w4_val:
    :param w4_val_err:
    :param z:
    :return:
    '''
    if w_four_good:
        # join W3 & W4 observed luminosities
        measured = np.array([float(w3_val.value), float(w4_val.value)])
        measured_errs = np.array([float(w3_val_err.value), float(w4_val_err.value)])
        params = 2
        # simulate a WISE flux by integrating the template over the response curves
        simulated = np.array(simulate_wise_fluxes(z, tem, wise_bandpasses_3_4, False))
        # perform least squares fit of observed W3, W4 luminosities to the simulated W3, W4 luminosities
        # this gives a normalization parameter which can be multiplied by the template TIR luminosity to give an
        # estimate of the intrinsic luminosity of the source
        # note: this is equation 3 from this paper: https://aip.scitation.org/doi/pdf/10.1063/1.168428
        # can use flux instead of lum , need to figure out how simulated piece works

        f_ratio = calc_ratio(measured, measured_errs, simulated, params)
        return measured, measured_errs, simulated, f_ratio

        # if there is no W4 data, simply take ratio of template and observed luminosity at W3
    else:
        params = 1
        measured = np.array([float(w3_val.value), 0.])
        measured_errs = np.array([float(w3_val_err.value), 0.])

        simulated = np.array(simulate_wise_fluxes(z, tem, wise_bandpasses_3_4, False))
        f_ratio = calc_ratio(measured, measured_errs, simulated, params)
        # l_ratio = float(w3_lum.value/tem_lum[2])

        return measured, measured_errs, simulated, f_ratio


def calc_sfr(total_ir, l_ratio):
    L_ir_tot = total_ir * l_ratio * u.W
    SFR = murphyIRSFR(L_ir_tot)
    return SFR


def make_title(tem, name, calc_SFR, SFR):
    """
    :param tem: takes in a template
    :param name: Template name
    :param calc_SFR: Boolean, tells whether to add SFR or not
    :param SFR: Takes in SFR value, regardles of calc_SFR, where calc_SFR == False, SFR is a dummy val
    :return: Title for plot
    """
    tem_break = tem.split("/")
    tem_trunc = tem_break[len(tem_break) - 1].split('_spec')[0]
    if calc_SFR:
        title = name + ': ' + tem_trunc + ' - SFR: ' + str(round(SFR))
    else:
        title = name + ': ' + tem_trunc
    return title


def calc_chi(measured_flux: list, simulated: list, f_ratio: float, measured_flux_errs) -> float:
    """
    :param measured_flux: galaxy flux, W3,W4 passband
    :param simulated: simulated galaxy flux from template for same range
    :param f_ratio: a ratio...
    :return:
    """
    unc = np.sqrt(measured_flux_errs ** 2 + (measured_flux * .1) ** 2)
    chi = (((measured_flux - simulated * f_ratio) / unc) ** 2).sum()
    # print(f"Chi: {chi}")
    return chi


def chi_hist(chis, pdf):
    """
    Creates histogram of chi values and saves to given pdf.
    :param chis: list of chi values
    :param pdf: pdf
    :return: None
    """
    plt.hist(chis, 15)
    plt.title("Chi Values")
    plt.xlabel("Chi Value")
    plt.ylabel("Number of Chis")
    pdf.savefig()
    plt.close()


def plot_Fluxes(tem_flux, f_ratio, simulated, z, name, tem, measured_flux, measured_flux_errs, pdf, noisy=None, SFR=0,
                calc_SFR=False, chi=None):
    if chi is not None:
        title = make_title(tem, name, calc_SFR, SFR) + f" Chi: {chi:.5f}"
    else:
        title = make_title(tem, name, calc_SFR, SFR)

    if noisy is not None:
        plt.plot(noisy[0], noisy[1] * f_ratio, linewidth=0.5, label="Noisy Template")
    else:
        plt.plot(tem_flux[0], tem_flux[1] * f_ratio, linewidth=0.5, label="Template")
    # plt.plot(tem_flux[0], tem_flux[1] * f_ratio, linewidth=0.5, label="Template")
    wave = np.array([12, 22]) / (1 + z)
    plt.scatter(wave, simulated * f_ratio, marker='s', facecolors='none', edgecolors='green',
                label="Model Photometry")
    plt.errorbar(wave, measured_flux, yerr=measured_flux_errs, color='red', marker='*', ls='none',
                 label="Photometry")

    plt.xlim(4, 30)
    plt.ylim(10e-5, 10e-1)
    plt.title(title)
    plt.xlabel('Wavelength [microns]')
    plt.ylabel(r"$F_{\nu}[ergs/s/cm^2/Hz]$")
    plt.legend()
    # plt.xscale('log')
    plt.yscale('log')

    pdf.savefig()
    plt.close()


def calc_sigma_f(snr: float, f: float) -> float:
    """
    :param SNR: Signal to Noise ratio from JWST ETC file
    :param f: frequency bin
    :return: sigma_f: width of gaussian distribution
    """
    return np.sqrt(
        (f / snr) ** 2 + (.05 * f) ** 2)  # added uncertainty floor of 10% bcs. random vals are too small alone?
    # return (f / snr)


def calc_snr(snr_max, f_max, f_bin):
    """
    :param snr_max:
    :param f_max:
    :param f_bin:
    :return:snr
    """
    snr = snr_max * np.sqrt(np.abs(f_bin / f_max))
    return snr


def make_noisy_f_bin(snr: float, f: float) -> float:
    """
    :param snr: snr of bin
    :param f: flux value for one pixel
    :return: noisy flux value
    """
    sigma_f = calc_sigma_f(snr, f)
    random_flux = np.abs(np.random.normal(f, sigma_f))
    return random_flux


def find_snr_max_index(flux, snr_max_bounds):
    for i, s in enumerate(snr_max_bounds):
        if flux <= s:
            return i


# def make_noisy_spectrum(fluxes: list,snr: list, f_maxes:list,snr_max_bounds:list)->np.array:
def make_noisy_spectrum(noise_tem_flux: list) -> np.array:
    """
    :param fluxes: array of fluxes from a single template
    :param snr: list of all max snrs
    :param f_maxes: list of all max fluxes
    :param snr_max_bounds: list of top boundaries for max snr bins
    :return: one noisy spectra
    """
    fluxes = noise_tem_flux[1]
    wavelengths = noise_tem_flux[0]
    max1 = fluxes[(0 < wavelengths) & (wavelengths <= 17.5)].max()
    max2 = fluxes[(17.5 < wavelengths) & (wavelengths <= 22.5)].max()
    max3 = fluxes[(22.5 < wavelengths) & (wavelengths <= 27.5)].max()
    max4 = fluxes[27.5 < wavelengths].max()
    # f_maxes = [16 / 1e26, 4 / 1e26, 4 / 1e26, 12 / 1e26, 27 / 1e26]# fluxes in f_nu units
    f_maxes = [max1, max2, max3, max4]
    snr = [175, 45, 50, 18, 30]
    snr_max_bounds = [17.5, 22.5, 25, 27.5, 30]
    noisy_spectrum = np.zeros(len(fluxes))
    for i in range(len(fluxes)):
        max_index = find_snr_max_index(fluxes[i], snr_max_bounds)
        snr_max = snr[max_index]
        f_max = f_maxes[max_index]
        bin_snr = calc_snr(snr_max, f_max, fluxes[i])
        noisy_spectrum[i] = make_noisy_f_bin(bin_snr, fluxes[i])
    return noisy_spectrum


def get_bandpass_flux(name):
    # convert WISE mag to flux in Janskys
    fluxes = WISE.mag_to_flux(name)
    w3_flux = fluxes[0] * u.Jy
    w3_flux_err = fluxes[2] * u.Jy
    # assume no W4 data for now
    w4_good = False
    # if there's data for W4
    if not np.isnan(fluxes[1]):
        w4_flux = fluxes[1] * u.Jy
        w4_flux_err = fluxes[3] * u.Jy
        w4_good = True
    return w3_flux, w3_flux_err, w4_flux, w4_flux_err, w4_good


def IR_Fluxes(z, name, pdfEnder, calc_SFR=False, tems=templates, noisy=False):
    # convert WISE mag to flux in Janskys
    fluxes = WISE.mag_to_flux(name)
    w3_flux = fluxes[0] * u.Jy
    w3_flux_err = fluxes[2] * u.Jy
    # assume no W4 data for now
    w4_good = False
    # if there's data for W4
    SFRs = []
    chis = []
    if not np.isnan(fluxes[1]):
        w4_flux = fluxes[1] * u.Jy
        w4_flux_err = fluxes[3] * u.Jy
        w4_good = True
    with open(projpath + 'integrations/kirk.txt', 'rb') as fb:
        total_ir = np.array(pickle.load(fb))

        # for each template
    with PdfPages(name + pdfEnder + ".pdf") as pdf:
        for i, tem in enumerate(tems):
            # redshift wavelengths of template
            tem_flux = redshift_spectrum(z, tem, False)

            noisy_tem_flux = tem_flux
            if noisy:
                noisy_tem_flux[1] = make_noisy_spectrum(noisy_tem_flux)
            else:
                noisy_tem_flux = None
            measured_flux, measured_flux_errs, simulated, f_ratio = apply_flux_fit(w4_good, w3_flux, w3_flux_err,
                                                                                   w4_flux, w4_flux_err, z, tem)

            # the observed LIR is just the template TIR luminosity multiplied by the normalization factor determined
            chi = calc_chi(measured_flux, simulated, f_ratio, measured_flux_errs)
            chis.append(chi)
            if calc_SFR == True:
                SFR = calc_sfr(total_ir[i], f_ratio)
                SFRs.append(SFR)
                # (tem_lum,l_ratio,simulated,z,name,tem,measured_lums,measured_lum_errs,pdf, SFR  = 0,calc_SFR = False)
                plot_Fluxes(tem_flux, f_ratio, simulated, z, name, tem, measured_flux, measured_flux_errs, pdf, SFR,
                            noisy=noisy_tem_flux,
                            calc_SFR=True, chi=chi)
            else:
                plot_Fluxes(tem_flux, f_ratio, simulated, z, name, tem, measured_flux, measured_flux_errs, pdf,
                            noisy=noisy_tem_flux, chi=chi)
        chi_hist(chis, pdf)
    return chis


def plot_spectral_features():
    """
    :return:None, plots PAH and spectral emission features
    """
    plt.axvline(8.6, ls='--', linewidth=.5, color='lightslategray')
    plt.axvline(11.3, ls='--', linewidth=.5, color='lightslategray')
    plt.axvline(12.8, ls=':', linewidth=.5, color='rosybrown')
    plt.axvline(15.6, ls=':', linewidth=.5, color='rosybrown')


def plot_most_unique(z, name, tem_list, projpath, brown_tems):
    with PdfPages(f"{name}_Top5.pdf") as pdf:
        plt.figure()
        for t in tem_list:
            w3_flux, w3_flux_err, w4_flux, w4_flux_err, w4_good = get_bandpass_flux(name)
            tem = projpath + t + '_spec_conv.dat'
            print(f"Tem:{tem}")
            tem_flux = redshift_spectrum(z, tem, False)
            tem_flux[1] = make_noisy_spectrum(tem_flux)
            tem_break = tem.split("/")
            tem_trunc = tem_break[len(tem_break) - 1].split('_spec')[0]
            measured_flux, measured_flux_errs, simulated, f_ratio = apply_flux_fit(w4_good, w3_flux, w3_flux_err,
                                                                                   w4_flux, w4_flux_err, z, tem)
            color_index = brown_tems.index(tem)
            print(f"color index: {color_index}")
            plt.plot(tem_flux[0], tem_flux[1] * f_ratio, color='darkslategray', linewidth=0.5, label=tem_trunc)

        plot_spectral_features()
        plt.xlim(4, 30)
        plt.ylim(10e-6, 10e-1)
        plt.title(name + " Top 5")
        plt.xlabel('Wavelength [microns]')
        plt.ylabel(r"$F_{\nu}[ergs/s/cm^2/Hz]$")
        plt.yscale('log')
        plt.legend()

        pdf.savefig()
        plt.close()


def plot_SFRs(tem_lum, l_ratio, simulated, z, name, tem, measured_lums, measured_lum_errs, pdf, SFR=0,
              calc_SFR=False) -> object:
    '''

    :param tem_lum: array of template luminosity
    :param l_ratio:
    :param simulated:array of simulated w3 and w4 values from template
    :param z:redshift of gal
    :param name:galaxy name
    :param tem:template file
    :param measured_lums:
    :param measured_lum_errs:
    :param pdf:pdf object
    :param SFR:default SFR value, filler #
    :param calc_SFR: Boolean that is True when SFRs should be calculated
    :return:
    '''
    # make a plot

    plt.plot(tem_lum[0], tem_lum[1] * l_ratio, linewidth=0.5, label="Template")
    wave = np.array([12, 22]) / (1 + z)
    plt.scatter(wave, simulated * l_ratio, marker='s', facecolors='none', edgecolors='green',
                label="Model Photometry")
    # plt.scatter(wave, measured_lums, color='red', marker='*')
    plt.errorbar(wave, measured_lums, yerr=measured_lum_errs, color='red', marker='*', ls='none',
                 label="Photometry")
    plt.xlim(1, 100)
    plt.ylim(10e21, 10e26)
    title = make_title(tem, name, calc_SFR, SFR)
    plt.title(title)
    plt.xlabel('Wavelength [microns]')
    plt.ylabel('Luminosity [W/Hz]')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')

    pdf.savefig()
    plt.close()
    return SFR


# calculate IR SFRs
def IR_SFRs(z, name, pdfEnder, calc_SFR=False, tems=templates):
    # luminosity distance
    d = cosmo.luminosity_distance(z)
    # convert WISE mag to flux in Janskys
    fluxes = WISE.mag_to_flux(name)
    w3_flux = fluxes[0] * u.Jy
    w3_flux_err = fluxes[2] * u.Jy
    # assume no W4 data for now
    w4_good = False
    # Set these as empty for use later
    w4_lum, w4_lum_err = 0, 0

    # calculate luminosities with fluxes & distances
    w3_lum = (w3_flux * 4 * np.pi * d ** 2).to('W/Hz')
    w3_lum_err = ((4 * np.pi * d ** 2) * w3_flux_err).to('W/Hz')
    # print('W3:', w3_lum, w3_lum_err)

    # if there's data for W4
    if not np.isnan(fluxes[1]):
        w4_flux = fluxes[1] * u.Jy
        w4_flux_err = fluxes[3] * u.Jy
        w4_good = True
        w4_lum = (w4_flux * 4 * np.pi * d ** 2).to('W/Hz')
        w4_lum_err = ((4 * np.pi * d ** 2) * w4_flux_err).to('W/Hz')

    # lists for SFR results
    SFRs = []

    # read in template total IR luminosities previously calculated
    with open(projpath + 'integrations/kirk.txt', 'rb') as fb:
        total_ir = np.array(pickle.load(fb))

    # for each template
    with PdfPages(name + pdfEnder + ".pdf") as pdf:
        # fig = plt.figure()
        # tems = np.sort(tems)
        SED_list = []
        for i, tem in enumerate(tems):

            # redshift wavelengths of template
            tem_lum = redshift_spectrum(z, tem, False)

            # if there is W4 data, do least squares fit of W3 & W4 points to the template curve
            measured_lums, measured_lum_errs, simulated, l_ratio = apply_template_fit(w4_good, w3_lum, w3_lum_err,
                                                                                      w4_lum, w4_lum_err, z, tem)

            # the observed LIR is just the template TIR luminosity multiplied by the normalization factor determined
            # print("len total_ir",len(total_ir))

            if calc_SFR == True:
                SFR = calc_sfr(total_ir[i], l_ratio)
                SFRs.append(SFR)
                # (tem_lum,l_ratio,simulated,z,name,tem,measured_lums,measured_lum_errs,pdf, SFR  = 0,calc_SFR = False)
                plot_SFRs(tem_lum, l_ratio, simulated, z, name, tem, measured_lums, measured_lum_errs, pdf, SFR,
                          calc_SFR=True)
            else:
                plot_SFRs(tem_lum, l_ratio, simulated, z, name, tem, measured_lums, measured_lum_errs, pdf)

        return np.average(SFRs), np.std(SFRs), SED_list


def plot_all(z, name:str, pdfEnder:str, chis, boundary, tems=templates):
    """
    Plots all given spectra that fall within the boundary value
    :param z: galaxy redshift
    :param name: name of galaxy
    :param pdfEnder: string to name file
    :param chis: array of chi values
    :param boundary: boundary of max allowed chi
    :param tems: templates
    :return: None, creates plot
    """
    fluxes = WISE.mag_to_flux(name)
    w3_flux = fluxes[0] * u.Jy
    w3_flux_err = fluxes[2] * u.Jy
    # assume no W4 data for now
    w4_good = False
    # if there's data for W4
    if not np.isnan(fluxes[1]):
        w4_flux = fluxes[1] * u.Jy
        w4_flux_err = fluxes[3] * u.Jy
        w4_good = True

    consistent_fit_test = np.array(chis) <= boundary
    with PdfPages(name + pdfEnder + ".pdf") as pdf:
        plt.figure()
        for i, tem in enumerate(tems):
            if consistent_fit_test[i]:
                # redshift wavelengths of template
                tem_flux = redshift_spectrum(z, tem, False)

                tem_break = tem.split("/")
                tem_trunc = tem_break[len(tem_break) - 1].split('_spec')[0]
                measured_flux, measured_flux_errs, simulated, f_ratio = apply_flux_fit(w4_good, w3_flux, w3_flux_err,
                                                                                       w4_flux, w4_flux_err, z, tem)
                plt.plot(tem_flux[0], tem_flux[1] * f_ratio, color="darkslategrey", linewidth=0.5, label=tem_trunc)
                wave = np.array([12, 22]) / (1 + z)
        plt.errorbar(wave, measured_flux, yerr=measured_flux_errs, color='red', marker='*', ls='none',
                     label="Photometry")
        plot_spectral_features()
        plt.xlim(4, 30)
        plt.ylim(10e-6, 10e-1)
        plt.title(name + " All Possible Fits")
        plt.xlabel('Wavelength [microns]')
        plt.ylabel(r"$F_{\nu}[ergs/s/cm^2/Hz]$")
        plt.yscale('log')
        plt.legend(prop={'size': 4})

        pdf.savefig()
        plt.close()


def wise_color_color_plot(name, x, y):
    '''
    :param name: galaxy name
    :param x: string of x axis name
    :param y: string of y axis name
    :return: None, plots color color plot of given x and y w/ error bars
    '''
    colors = WISE.colors(name)
    labels = ["one_two", "three_four", "two_three", "one_two_err", "three_four_err", "two_three_err"]
    color_dict = dict(zip(labels, colors))
    if x not in labels or y not in labels:
        return TypeError(
            "x or y is not one of the correct label. Must be either: 'one_two','three_four', or 'two_three'")
    # fluxes = WISE.mag_to_flux(name, w2=True
    plt.errorbar(color_dict[x], color_dict[y], xerr=color_dict[x + "_err"], yerr=color_dict[y + "_err"], label=name)


def plot_agn_bounds():
    '''
    Plots Stern et al. 2012 and Blecha et al. 2018 AGN boundaries
    :return: None
    '''
    plt.axhline(.8, ls='--', color = 'lawngreen', label='Stern et al. 2012')
    x_vals = np.linspace(2, 6, 1000)
    blecha_h = np.full(1000, .5)
    mask1 = (x_vals > 2.2) & (x_vals < 4.7)
    plt.plot(x_vals[mask1], blecha_h[mask1], color='darkblue')
    y_vals = np.linspace(0, 2, 1000)
    mask2 = y_vals > .5
    blecha_v = np.full(1000, 2.2)
    plt.plot(blecha_v[mask2], y_vals[mask2], color='darkblue')
    y_vals = 2.0 * x_vals - 8.9
    mask = (y_vals > .5) & (y_vals < 2)
    plt.plot(x_vals[mask], y_vals[mask], color='darkblue', label='Blecha et. al 2018')


def all_gals_color_color(gals_list):
    '''
    :param gals_list: list of galaxies to plot
    :return: creates pdf with two color-color plots containing the galaxies from gals_list
    '''
    with PdfPages("JWSTGalsColorColor.pdf") as pdf:
        plt.figure()
        for g in gals_list:
            wise_color_color_plot(g, "three_four", "two_three")
        plt.axvline(2.8, ls="--", color='slategray')
        plt.title("JWST Gals Color-Color Plot")
        plt.xlabel("W3-W4")
        plt.ylabel("W2-W3")
        plt.legend()
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        pdf.savefig()
        plt.close()

        plt.figure()
        for g in gals_list:
            wise_color_color_plot(g, "two_three", "one_two")
        plot_agn_bounds()
        plt.xlim(2, 6)
        plt.ylim(0, 2)
        plt.title("JWST Gals Color-Color Plot")
        plt.xlabel("W2-W3")
        plt.ylabel("W1-W2")
        plt.legend()
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        pdf.savefig()
        plt.close()


s = "aliceblue, aqua, aquamarine,blue,blueviolet, burlywood, cadetblue,chartreuse,coral, cornflowerblue,cornsilk,c, \
        crimson, cyan, darkblue, darkcyan,darkgoldenrod, darkgray, darkgrey, darkgreen,\
    darkkhaki, darkmagenta, darkolivegreen, darkorange,\
    darkorchid, darkred, darksalmon, darkseagreen,\
    darkslateblue, darkslategray, darkslategrey,\
    darkturquoise, darkviolet, deeppink, deepskyblue,\
    dimgray, dimgrey, dodgerblue, firebrick,\
    forestgreen, fuchsia, grey,\
    gold, goldenrod, gray, grey, green,\
    greenyellow, hotpink, indianred, indigo,k,\
    khaki, lavender, lawngreen,\
    lemonchiffon, lightblue, lightcoral, lightcyan,\
    lightgoldenrodyellow, gray, grey,\
    lightgreen, lightpink, lightsalmon, lightseagreen,\
    lightskyblue, lightslategray, lightslategrey,\
    lightsteelblue, lime, limegreen,m,\
    magenta, maroon, mediumaquamarine,\
    mediumblue, mediumorchid, mediumpurple,\
    mediumturquoise, mediumvioletred, midnightblue,\
    mistyrose, moccasin, navajowhite, navy,\
    olive, olivedrab, orange, orangered,\
    orchid, palegoldenrod, palegreen, paleturquoise,\
    palevioletred, papayawhip, peachpuff, peru, pink,\
    plum, powderblue, purple, red,r, rosybrown,\
    royalblue, saddlebrown, salmon, sandybrown,\
    seagreen, sienna, silver, skyblue,\
    slateblue, slategray, slategrey, springgreen,\
    steelblue, tan, teal, thistle, tomato, turquoise,\
    violet, wheat, lightslategrey, y, yellow,\
    yellowgreen"
li = s.split(',')
li = [l.replace('\n', '') for l in li]
li = [l.replace(' ', '') for l in li]  # list of colors?
