#%%
from utility.general import mkdir
from utility.processing import filter_by_channel_number, split_P_S_dataframe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
import statsmodels.api as sm

def write_regression_summary(regression_results_dir, file_name, reg):
    # make a directory to store the regression results in text
    regression_text = regression_results_dir + '/regression_results_txt'
    mkdir(regression_text)
    with open(regression_text + '/' + file_name + '.txt', "w") as text_file:
        text_file.write(reg.summary().as_text())

def fit_regression(peak_amplitude_df, weighted='ols', M_threshold=None, snr_threshold=None, min_channel=None):

    if M_threshold:
        peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]
    
    # use P and S separately to do the regression
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df, snr_threshold)
    
    if min_channel:
        peak_amplitude_df_P = filter_by_channel_number(peak_amplitude_df_P, min_channel)
        peak_amplitude_df_S = filter_by_channel_number(peak_amplitude_df_S, min_channel)

    if weighted == 'ols': # Ordinary Linear Square (OLS)
        # Linear regression without weight
        regP = smf.ols(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_P).fit()
        regS = smf.ols(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', data=peak_amplitude_df_S).fit()
    
    elif weighted == 'wls': # Weighted Linear Square (WLS)
        # Linear regression with weight, the weight is 10**(magnitude/2)
        regP = smf.wls(formula='np.log10(peak_P) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', 
                    data=peak_amplitude_df_P, weights = (10**peak_amplitude_df_P.magnitude)).fit()
        regS = smf.wls(formula='np.log10(peak_S) ~ magnitude + np.log10(distance_in_km) + C(region_site) - 1', 
                    data=peak_amplitude_df_S, weights = (10**peak_amplitude_df_S.magnitude)).fit() 
    else:
        raise TypeError(f'{weighted} is not defined, only "ols" or "wls"!')       

    print(regP.params[-2:])
    print(regS.params[-2:])
    print('\n\n')

    return regP,regS

def store_regression_results(regX, regression_results_dir, results_filename):
    """Write regressio summary and store results"""
    try:
        write_regression_summary(regression_results_dir, results_filename, regX)
        regX.save(regression_results_dir + '/' + results_filename + '.pickle', remove_data=True)
    except:
        print('Regression results file is empty, did nothing...')

def secondary_site_calibration(regX, peak_amplitude_df, wavetype):
    """Apply secondary calibration on the site terms"""
    wavetype = wavetype.upper()
    y_predict = regX.predict(peak_amplitude_df)

    peak_amplitude_df_temp = pd.DataFrame(columns=['region_site', 'channel_id', 'region', 'magnitude', f'diff_peak_{wavetype}', f'site_calibate_{wavetype}'])
    peak_amplitude_df_temp.channel_id = peak_amplitude_df.channel_id
    peak_amplitude_df_temp.region = peak_amplitude_df.region
    peak_amplitude_df_temp.magnitude = peak_amplitude_df.magnitude

    #weighted_all = np.nansum(10**peak_amplitude_df.magnitude)
    peak_amplitude_df_temp[f'diff_peak_{wavetype}'] = (-y_predict + np.log10(peak_amplitude_df[f'peak_{wavetype}']))#*10**peak_amplitude_df.magnitude/weighted_all

    second_calibration = peak_amplitude_df_temp.groupby(['channel_id', 'region'], as_index=False).mean()
    temp_df = peak_amplitude_df_temp[['channel_id', 'region']].drop_duplicates(subset=['channel_id', 'region'])
    second_calibration = pd.merge(second_calibration, temp_df, how='inner', left_on=['channel_id', 'region'], right_on=['channel_id', 'region'])
    second_calibration = second_calibration.drop(columns=['magnitude'])
    return second_calibration


def extract_site_terms(regP, regS, peak_amplitude_df):
    """Extract the site terms as individual files from regression"""
    region_site = np.sort(peak_amplitude_df.region_site.unique())

    temp_df_P = pd.DataFrame(columns=['region_site', 'site_term_P'])
    temp_df_S = pd.DataFrame(columns=['region_site', 'site_term_S'])

    try:
        site_term_P = regP.params[:-2]
        # temp_df_P = pd.DataFrame(columns=['combined_channel_id', 'site_term_P'])
        temp_df_P['region_site'] = [temp.replace('C(region_site)[', '').replace(']', '') for temp in site_term_P.index]
        temp_df_P['site_term_P'] = np.array(site_term_P)
        #temp_df_P = get_site_term_dataframe(temp_df_P)
    except:
        print('P regression not found, assign Nan to the site term')
        site_term_P = np.nan
    try:
        site_term_S = regS.params[:-2]
        # temp_df_S = pd.DataFrame(columns=['combined_channel_id', 'site_term_S'])
        temp_df_S['region_site'] = [temp.replace('C(region_site)[', '').replace(']', '') for temp in site_term_S.index]
        temp_df_S['site_term_S'] = np.array(site_term_S)
        #temp_df_S = get_site_term_dataframe(temp_df_S)
    except:
        print('S regression not found, assign Nan to the site term')
        site_term_S = np.nan

    temp_df1 = pd.DataFrame(columns=['region_site'])
    temp_df1['region_site'] = region_site
    temp_df1 = pd.merge(temp_df1, temp_df_P, on='region_site', how='outer')
    temp_df1 = pd.merge(temp_df1, temp_df_S, on='region_site', how='outer')

    temp_df2 = peak_amplitude_df.loc[:, ['region_site', 'channel_id']]
    site_term_df = pd.merge(temp_df2, temp_df1, on='region_site', how='outer')
    site_term_df = site_term_df.drop_duplicates(subset=['region_site', 'channel_id'])

    return site_term_df

def site_term_avg(site_term, site_term_key, peak_key, weights=False):
    if weights:
        # calculate weight normal
        weight_normal = site_term[['channel_id', 'weight', 'event_id']]
        weight_normal = weight_normal.groupby('channel_id').sum().reset_index()
        # calculate site term
        site_term[site_term_key] = site_term[peak_key]*site_term['weight']
        site_term = site_term.groupby('channel_id').sum().reset_index() 
        site_term[site_term_key] = site_term[site_term_key] / weight_normal['weight']
        return site_term[['channel_id', site_term_key]]
    else:
        site_term = site_term.groupby('channel_id').mean().reset_index()
        site_term = site_term.rename(columns={peak_key: site_term_key})
        return site_term[['channel_id', site_term_key]]
        

def initialize_site_term(regX, peak_amplitude_df, wavetype):
    """Funciton used to initialize the site terms dataframe with constant site term"""
    site_term_df = peak_amplitude_df.groupby(['region','channel_id']).size().reset_index().drop(columns=0)
    site_term_df[f'site_term_{wavetype}'] = 0

    for region_i in site_term_df.region.unique():
        site_term_df[f'site_term_{wavetype}'][site_term_df[site_term_df.region == region_i].index] = regX.params[f'C(region)[{region_i}]']
    return site_term_df

def fit_regression_iteration(peak_amplitude_df_fit, wavetype, weighted='wls', n_iter=50, rms_epsilon=0.2):
    """Funciton to iteratively fit for the scaling coefficients"""
    wavetype = wavetype.upper()
    if (wavetype != 'P') and (wavetype != 'S'):
        raise TypeError("wavetype must be 'P' or 'S' !")

    # Initial regression for the coefficients and site terms:
    if weighted == 'wls':
        # Weighted Linear Square (WLS)
        # Linear regression with weight, the weight is 10**(magnitude/2)
        regX = smf.wls(formula=f'np.log10(peak_{wavetype}) ~ magnitude + np.log10(distance_in_km) + C(region) - 1', 
                    data=peak_amplitude_df_fit, weights = (10**peak_amplitude_df_fit.magnitude)).fit()
    elif weighted == 'ols':
        # Weighted Linear Square (WLS)
        # Linear regression with weight, the weight is 10**(magnitude/2)
        # Initial regression:
        regX = smf.ols(formula=f'np.log10(peak_{wavetype}) ~ magnitude + np.log10(distance_in_km) + C(region) - 1', 
                    data=peak_amplitude_df_fit).fit()
    else:
        raise TypeError(f'"{weighted}" is not defined, only "ols" or "wls"!')

    site_term_df = initialize_site_term(regX, peak_amplitude_df_fit, wavetype)
    peak_amplitude_df_fit0 = peak_amplitude_df_fit

    # interation begin
    second_calibration = secondary_site_calibration(regX, peak_amplitude_df_fit, wavetype)
    second_calibration = second_calibration.rename(columns={f'diff_peak_{wavetype}': f'site_term_{wavetype}'})

    # initialize iteration parameters
    fitting_rms = []
    i_iter = 0
    fitting_rms_diff = 1e20 # to record the improvement of rms

    # to store the results in the previous iteration
    # for i_iter in range(n_iter):
    while (i_iter <= n_iter) and (fitting_rms_diff >= rms_epsilon):
        if i_iter >= 1:
            regX0, site_term_df0 = regX, site_term_df
        i_iter+=1
        # update site term
        site_term_df = pd.concat([site_term_df, second_calibration]).groupby(['channel_id', 'region']).sum(min_count=1).reset_index()

        # Update magnitude and distance coefficients
        peak_amplitude_df_fit = pd.merge(peak_amplitude_df_fit0, site_term_df, left_on=['channel_id', 'region'], right_on=['channel_id', 'region'])
        peak_amplitude_df_fit['combined_channel_id'] = 0

        # remove site term effects to fit magnitude and distance coefficients
        peak_amplitude_df_fit[f'peak_{wavetype}'] = peak_amplitude_df_fit[f'peak_{wavetype}']/10**peak_amplitude_df_fit[f'site_term_{wavetype}']

        if weighted == 'wls':
            regX = smf.wls(formula=f'np.log10(peak_{wavetype}) ~ magnitude + np.log10(distance_in_km) - 1', 
                        data=peak_amplitude_df_fit, weights = (10**peak_amplitude_df_fit.magnitude)).fit()
        elif weighted == 'ols':
            regX = smf.ols(formula=f'np.log10(peak_{wavetype}) ~ magnitude + np.log10(distance_in_km) - 1', 
                        data=peak_amplitude_df_fit).fit()
        else:
            raise TypeError(f'"{weighted}" is not defined, only "ols" or "wls"!')

        # record the misfit
        fitting_rms_iter = np.nanmean((np.log10(peak_amplitude_df_fit[f'peak_{wavetype}']) - regX.predict(peak_amplitude_df_fit))**2)**0.5

        if i_iter > 1:
            fitting_rms_diff = -((fitting_rms_iter - fitting_rms[-1])/fitting_rms[-1] * 100)
        fitting_rms.append(fitting_rms_iter)

        # print information about the regression
        print(f'================={i_iter} iter ---- fitting rms: {fitting_rms_iter}=================')
        print(f'======= {wavetype} parameters ======')
        print([regX.params])

        # Update site terms
        second_calibration = secondary_site_calibration(regX, peak_amplitude_df_fit, wavetype)
        second_calibration[f'site_term_{wavetype}'] = second_calibration[f'diff_peak_{wavetype}']
    
    return regX0, site_term_df0[['channel_id', 'region', f'site_term_{wavetype}']], fitting_rms


def fit_regression_transfer(peak_amplitude_df, regX, wavetype, weighted, M_threshold=[0, 10], snr_threshold=10, min_channel=100):
    wavetype = wavetype.upper()
    if M_threshold:
        peak_amplitude_df = peak_amplitude_df[(peak_amplitude_df.magnitude >= M_threshold[0]) & (peak_amplitude_df.magnitude <= M_threshold[1])]

    # use P and S separately to do the regression
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df, snr_threshold)

    if min_channel: # only keep events with available channels > min_channel
        peak_amplitude_df_P0 = filter_by_channel_number(peak_amplitude_df_P, min_channel)
        peak_amplitude_df_S0 = filter_by_channel_number(peak_amplitude_df_S, min_channel)

    if wavetype == 'P':
        site_term = peak_amplitude_df_P0.copy()
    elif wavetype == 'S':
        site_term = peak_amplitude_df_S0.copy()
    else:
        raise TypeError("wavetype must be 'P' or 'S' !")

    # remove the contribution from fixed coefficients contribution
    mag_coef, dist_coef = regX.params['magnitude'], regX.params['np.log10(distance_in_km)']
    
    if (wavetype == 'P') or (wavetype == 'S'):
        peak_key = f'peak_{wavetype}'
        site_term_key = f'site_term_{wavetype}'
    else:
        raise TypeError("wavetype must be 'P' or 'S' !")

    site_term[peak_key] = np.log10(site_term[peak_key] / 10**(site_term.magnitude * mag_coef) / site_term.distance_in_km**dist_coef)
    site_term['weight'] = 10**(site_term.magnitude/2) 

    if weighted == 'wls': # TODO: apply weighted average
        site_term = site_term_avg(site_term, site_term_key, peak_key, weights=True)
    elif weighted == 'ols':
        site_term = site_term_avg(site_term, site_term_key, peak_key, weights=False)
    else:
        raise TypeError('weighted must be "wls" or "ols"!')

    return site_term


def predict_strain(peak_amplitude_df, regX, site_term_df, wavetype):
    """Given data, regression results and site term DataFrame, predict the strain rate"""
    peak_amplitude_df_temp = pd.merge(peak_amplitude_df, site_term_df, how='outer', left_on=['channel_id', 'region'], right_on=['channel_id', 'region'])
    if wavetype.lower() == 'p':
        peak_predicted = 10**(regX.predict(peak_amplitude_df_temp) + peak_amplitude_df_temp.site_term_P)
    elif wavetype.lower() == 's':
        peak_predicted = 10**(regX.predict(peak_amplitude_df_temp) + peak_amplitude_df_temp.site_term_S)
    else:
        raise TypeError("wavetype must be 'P' or 'S' !")
    return peak_predicted, peak_amplitude_df_temp

def predict_magnitude(peak_amplitude_df, regX, site_term_df, wavetype):
    wavetype = wavetype.upper()
    """Given data, regression results and site term DataFrame, predict the magnitude"""
    peak_amplitude_df_temp = pd.merge(peak_amplitude_df, site_term_df, how='outer', left_on=['channel_id', 'region'], right_on=['channel_id', 'region'])
    if wavetype == 'P':
        predicted_magnitude = (np.log10(peak_amplitude_df_temp.peak_P) - peak_amplitude_df_temp.site_term_P  \
                    - regX.params['np.log10(distance_in_km)']*np.log10(peak_amplitude_df_temp.distance_in_km)) \
                    / regX.params['magnitude']
    elif wavetype == 'S':
        predicted_magnitude = (np.log10(peak_amplitude_df_temp.peak_S) - peak_amplitude_df_temp.site_term_S  \
                    - regX.params['np.log10(distance_in_km)']*np.log10(peak_amplitude_df_temp.distance_in_km)) \
                    / regX.params['magnitude']
    else:
        raise TypeError("wavetype must be 'P' or 'S' !")

    
    return predicted_magnitude, peak_amplitude_df_temp

def get_mean_magnitude(peak_amplitude_df, M_predict, method='median'):
    """Average the predicted magnitude for all available channels to obtain a final estimation"""
    temp_df = peak_amplitude_df[['event_id', 'magnitude']].copy()
    temp_df['predicted_M'] = M_predict

    if len(M_predict[~M_predict.isna()]) == 0:
        temp_df['predicted_M_std'] = M_predict

    else:
        if method == 'median':
            temp_df = temp_df.groupby(temp_df['event_id']).aggregate(np.nanmedian)
        elif method == 'mean':
            temp_df = temp_df.groupby(temp_df['event_id']).aggregate(np.nanmean)
        else:
            raise NameError('method needs to be either mean or median!')

        temp_df2 = peak_amplitude_df[['event_id', 'magnitude']].copy()
        temp_df2['predicted_M_std'] = M_predict
        temp_df2 = temp_df2.groupby(temp_df2['event_id']).aggregate(np.nanstd)
        temp_df = pd.concat([temp_df, temp_df2['predicted_M_std']], axis=1)

    return temp_df.reset_index()
# %%
