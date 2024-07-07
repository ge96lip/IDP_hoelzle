import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import joypy

from pcntoolkit.normative import estimate, predict, evaluate
from pcntoolkit.util.utils import compute_MSLL, create_design_matrix, create_bspline_basis
from nm_utils import calibration_descriptives, load_2d

from sklearn.metrics import r2_score
from scipy.stats import spearmanr, kendalltau

import argparse
import ast
import os 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--site_names_file', type=str, default='site_ids_ct_dk_59sites.txt')
    parser.add_argument('--map_roi', type=str, default="{'lh_entorhinal_thickness': 'L_entorhinal', 'lh_inferiortemporal_thickness': 'L_inferiortemporal', 'lh_middletemporal_thickness': 'L_middletemporal', 'lh_inferiorparietal_thickness': 'L_inferiorparietal', 'lh_fusiform_thickness': 'L_fusiform'}")
    parser.add_argument('--idp_ids', type=str, default="['L_entorhinal','L_inferiortemporal','L_middletemporal','L_inferiorparietal','L_fusiform']")
    parser.add_argument('--drop_columns', type=bool, default=True)
    parser.add_argument('--cols_cov', type=str, default="['AGE','PTGENDER']")
    
    args = parser.parse_args()
    # Convert the map_roi string to a dictionary
    args.map_roi = ast.literal_eval(args.map_roi)
    args.idp_ids = ast.literal_eval(args.idp_ids)
    args.cols_cov = ast.literal_eval(args.cols_cov)
    
    return args

def load_z_scoes(idp_ids, out_dir, data): 
    z_scores = {}
    # Load the Z-scores for each idp_id
    for col in idp_ids:
        print(f"Loading Z-scores for {col}")
        z_score_path = os.path.join(out_dir, f'{col}/Z_{suffix}.txt')
        z_scores[col] = np.loadtxt(z_score_path)

    # Convert the Z-scores dictionary to a DataFrame
    z_scores_df = pd.DataFrame(z_scores)
    # Ensure the Z-scores DataFrame has the same index as data
    z_scores_df.index = data.index

    # Combine the set and DX columns with the Z-scores
    combined_df = pd.concat([data[['set', 'DX']], z_scores_df], axis=1)

    output_file = os.path.join(out_dir, 'analysis', f'Z_score_alldata.txt')

    # Save the results to a CSV file
    combined_df.to_csv(output_file, index=False)

    return combined_df 

args = get_args()
# The rest of your code where you use map_roi
data_dir = args.data_dir
out_dir = args.out_dir 
# make analysis folder 
os.makedirs(os.path.join(out_dir, "analysis"), exist_ok=True)

data = pd.read_csv(os.path.join(data_dir, "data_merged.csv"))

# specific to chair's dataset: 
if args.drop_columns: 
    columns_to_drop = data.columns[2:6]
    df_ad = data.drop(columns_to_drop, axis=1)

valid_dx = ['AD', 'MCI', 'CN']
data = data[data['DX'].isin(valid_dx)]

# make gender numeric 
data.loc[:, 'PTGENDER'] = data['PTGENDER'].map({'Male': 0, 'Female': 1})

df_ad = data[data['DX'].isin(['CN'])]
df_patients = data[data['DX'].isin(['AD'])]

# test with full data 
df_test = data

df_test.reset_index(drop=True, inplace=True)
df_patients.reset_index(drop=True, inplace=True)
df_ad.reset_index(drop=True, inplace=True)

# load the sites used for training
site_names = args.site_names_file

with open(os.path.join(data_dir, site_names)) as f:
    site_ids_tr = f.read().splitlines()

# get the sites from the current dataset

df_test['sitenum'] = 0
df_patients['sitenum'] = 0
df_ad['sitenum'] = 0

site_ids_te = df_test["set"].unique()
site_ids_patients = df_patients["set"].unique()
site_ids_ad = df_ad["set"].unique()

print("Test Set: ")
site_to_sitenum = {}
for i, s in enumerate(site_ids_te):
    idx = df_test['set'] == s
    df_test.loc[idx, 'sitenum'] = i
    site_to_sitenum[s] = i
    print('site', s, sum(idx), "sitenum: ", i)


# Assign sitenum in patients set based on train set mapping
df_patients['sitenum'] = df_patients['set'].map(site_to_sitenum).fillna(-1).astype(int)
# Assign sitenum in patients set based on train set mapping
df_ad['sitenum'] = df_ad['set'].map(site_to_sitenum).fillna(-1).astype(int)

print("All Data Set: ")
for s in site_ids_ad:
    idx = df_ad['set'] == s
    print('site', s, sum(idx), "sitenum: ", df_ad.loc[idx, 'sitenum'].iloc[0])

print("Patients Set: ")
for s in site_ids_patients:
    idx = df_patients['set'] == s
    print('site', s, sum(idx), "sitenum: ", df_patients.loc[idx, 'sitenum'].iloc[0])

map_roi = args.map_roi

df_patients.rename(columns=map_roi, inplace=True)
df_ad.rename(columns=map_roi, inplace=True)
df_test.rename(columns=map_roi, inplace=True)

sns.set(font_scale=1.5, style='darkgrid')

# Create the plot
plot = sns.displot(df_test, x="AGE", hue="set", multiple="stack", height=6)

# Save the plot to a file
plot.savefig(os.path.join(out_dir, 'analysis', 'data_distribution.png'))  # Change this to your desired path and file name

# Close the plot
plt.close()

# Estimate the model on the data 
cols_cov = args.cols_cov

# limits for cubic B-spline basis 
xmin = -5
xmax = 110
force_refit = True 

# Absolute Z treshold above which a sample is considered to be an outlier (without fitting any model)
outlier_thresh = 3

if not all(elem in site_ids_tr for elem in site_ids_te):
    print('Warning: some of the testing sites are not in the training data')
  
site_ids_ad =  sorted(set(df_ad['set'].to_list()))  
if not all(elem in site_ids_ad for elem in site_ids_te):
    print('Warning: some of the testing sites are not in the adaptation data')
    
idp_ids = args.idp_ids

for idp_num, idp in enumerate(idp_ids): 
    print('Running IDP', idp_num, idp, ':')
    idp_dir = os.path.join(out_dir, idp)
    os.chdir(idp_dir)
    
    # extract and save the response variables for the test set
    y_te = df_test[idp].to_numpy() # originaly: df_test
    
    # save the variables
    resp_file_te = os.path.join(idp_dir, 'resp_te.txt') 
    np.savetxt(resp_file_te, y_te)
        
    # configure and save the design matrix
    cov_file_te = os.path.join(idp_dir, 'cov_bspline_te.txt')
    
    X_te = create_design_matrix(df_test[cols_cov], 
                                site_ids = df_test['set'],
                                all_sites = site_ids_tr,
                                basis = 'bspline', 
                                xmin = xmin, 
                                xmax = xmax)
    np.savetxt(cov_file_te, X_te)
    
    # check whether all sites in the test set are represented in the training set
    if all(elem in site_ids_tr for elem in site_ids_te):
        print('All sites are present in the training data')
        
        # just make predictions
        yhat_te, s2_te, Z = predict(cov_file_te, 
                                    alg='blr', 
                                    respfile=resp_file_te, 
                                    model_path=os.path.join(idp_dir,'Models'),
                                    outputsuffix='alldata')
    else:
        print('Some sites missing from the training data. Adapting model')

        # save the covariates for the adaptation data
        X_ad = create_design_matrix(df_ad[cols_cov],
                                    site_ids = df_ad['set'],
                                    all_sites = site_ids_tr,
                                    basis = 'bspline',
                                    xmin = xmin,
                                    xmax = xmax)
        cov_file_ad = os.path.join(idp_dir, 'cov_bspline_ad.txt')
        np.savetxt(cov_file_ad, X_ad)

        # save the responses for the adaptation data
        resp_file_ad = os.path.join(idp_dir, 'resp_ad.txt')
        y_ad = df_ad[idp].to_numpy()
        np.savetxt(resp_file_ad, y_ad)

        # save the site ids for the adaptation data
        sitenum_file_ad = os.path.join(idp_dir, 'sitenum_ad.txt')
        site_num_ad = df_ad['sitenum'].to_numpy(dtype=int)
        np.savetxt(sitenum_file_ad, site_num_ad)

        # save the site ids for the test data
        sitenum_file_te = os.path.join(idp_dir, 'sitenum_te.txt')
        site_num_te = df_test['sitenum'].to_numpy(dtype=int)
        np.savetxt(sitenum_file_te, site_num_te)
        yhat_te, s2_te, Z = predict(cov_file_te,
                                    alg = 'blr',
                                    respfile = resp_file_te,
                                    model_path = os.path.join(idp_dir,'Models'),
                                    adaptrespfile = resp_file_ad,
                                    adaptcovfile = cov_file_ad,
                                    adaptvargroupfile = sitenum_file_ad,
                                    testvargroupfile = sitenum_file_te, 
                                    outputsuffix = 'alldata')
        
# Extract unique set values
site_names = df_test['set'].unique()

# Create a dictionary to hold indices for each site
sites = {site: df_test.index[df_test['set'] == site].to_list() for site in site_names}

print("\n\n\n\n\n")
print("Evaluation on the full dataset")
suffix = 'alldata'
# initialise dataframe we will use to store quantitative metrics 
blr_metrics = pd.DataFrame(columns = ['eid', 'NLL', 'EV', 'MSLL', 'BIC', 'Skew', 'Kurtosis', 'mean_rec_error'])
blr_site_metrics = pd.DataFrame(columns = ['ROI', 'set', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])

for idp_num, idp in enumerate(idp_ids): 
    idp_dir = os.path.join(out_dir, idp)
    
    # load the predictions and true data. We use a custom function that ensures 2d arrays
    # equivalent to: y = np.loadtxt(filename); y = y[:, np.newaxis]
    yhat_te = load_2d(os.path.join(idp_dir, 'yhat_' + suffix + '.txt')) # load_2d(os.path.join(idp_dir, 'yhat_' + suffix + '.txt'))
    s2_te = load_2d(os.path.join(idp_dir, 'ys2_' + suffix + '.txt'))
    y_te = load_2d(os.path.join(idp_dir, 'resp_te.txt'))
    
    with open(os.path.join(idp_dir,'Models', 'NM_0_0_estimate.pkl'), 'rb') as handle:
        nm = pickle.load(handle) 
    
    # compute error metrics
    if nm.warp is None:
        metrics = evaluate(y_te, yhat_te)  
        
        # compute MSLL manually as a sanity check
        y_tr = df_test[idp].to_numpy() 
        y_tr_mean = np.array( [[np.mean(y_tr)]] )
        y_tr_var = np.array( [[np.var(y_tr)]] )
        MSLL = compute_MSLL(y_te, yhat_te, s2_te, y_tr_mean, y_tr_var)         
    else:
        warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1] 
        W = nm.blr.warp
        
        # warp predictions
        med_te = W.warp_predictions(np.squeeze(yhat_te), np.squeeze(s2_te), warp_param)[0]
        med_te = med_te[:, np.newaxis]
       
        # evaluation metrics
        metrics = evaluate(y_te, med_te)
        y_tr = df_test[idp].to_numpy() 
        # compute MSLL manually
        y_te_w = W.f(y_te, warp_param)
        y_tr_w = W.f(y_tr, warp_param)
        y_tr_mean = np.array( [[np.mean(y_tr_w)]] )
        y_tr_var = np.array( [[np.var(y_tr_w)]] )
        MSLL = compute_MSLL(y_te_w, yhat_te, s2_te, y_tr_mean, y_tr_var) 
           
        reconstruction_errors = np.mean(np.square(y_te - med_te), axis=1)

        # Calculate mean reconstruction error
        mean_reconstruction_error = np.mean(reconstruction_errors)
        print(f"Mean Reconstruction Error for {idp} is: ", mean_reconstruction_error) 
    
    Z = np.loadtxt(os.path.join(idp_dir, 'Z_' + suffix + '.txt'))
    [skew, sdskew, kurtosis, sdkurtosis, semean, sesd] = calibration_descriptives(Z)
    
    BIC = len(nm.blr.hyp) * np.log(y_tr.shape[0]) + 2 * nm.neg_log_lik
    
    blr_metrics.loc[len(blr_metrics)] = [idp, nm.neg_log_lik, metrics['EXPV'][0], 
                                         MSLL[0], BIC, skew, kurtosis, mean_reconstruction_error]
    for num, site in enumerate(sites):

        y_mean_te_site = np.array([[np.mean(y_te[sites[site]])]])
        y_var_te_site = np.array([[np.var(y_te[sites[site]])]])
        yhat_mean_te_site = np.array([[np.mean(yhat_te[sites[site]])]])
        yhat_var_te_site = np.array([[np.var(yhat_te[sites[site]])]])
       
        metrics_te_site = evaluate(y_te[sites[site]], yhat_te[sites[site]], s2_te[sites[site]], y_mean_te_site, y_var_te_site)

        site_name = num #site_names[num]
        blr_site_metrics.loc[len(blr_site_metrics)] = [idp, site_names[num], metrics_te_site['MSLL'][0], metrics_te_site['EXPV'][0], metrics_te_site['SMSE'][0], metrics_te_site['RMSE'][0], metrics_te_site['Rho'][0]]

blr_metrics.to_csv(os.path.join(out_dir,'blr_metrics'+suffix+'.csv'))

# Plot the EV

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def color_gradient(x=0.0, start=(0, 0, 0), stop=(1, 1, 1)):
    r = np.interp(x, [0, 1], [start[0], stop[0]])
    g = np.interp(x, [0, 1], [start[1], stop[1]])
    b = np.interp(x, [0, 1], [start[2], stop[2]])
    return r, g, b

plt.figure(dpi=380)

# Reorder the DataFrame to ensure the desired order
# blr_site_metrics['set'] = pd.Categorical(blr_site_metrics['set'], categories=['set_adni', 'set_aibl', 'set_delcode', 'set_jadni', 'set_ukb'], ordered=True)
# blr_site_metrics = blr_site_metrics.sort_values('set')

fig, axes = joypy.joyplot(blr_site_metrics, column=['EV'], overlap=2.5, by="set", ylim='own', fill=True, figsize=(8, 8),
                          legend=False, xlabels=True, ylabels=True, colormap=lambda x: color_gradient(x, start=(.08, .45, .8), stop=(.8, .34, .44)),
                          alpha=0.6, linewidth=.5, linecolor='w', fade=True)

plt.title('Test Set Explained Variance', fontsize=18, color='black', alpha=1)
plt.xlabel('Explained Variance', fontsize=14, color='black', alpha=1)
plt.ylabel('Set', fontsize=14, color='black', alpha=1)

# Save the plot as an image file
plt.savefig(os.path.join(out_dir, 'analysis', 'EV_all_data.png'), dpi=380)

# Close the plot to avoid displaying it
plt.close()

errors = {}

# Load the predictions for each feature
for col in idp_ids:
    idp_dir = os.path.join(out_dir, col)
    os.chdir(idp_dir)
    yhat_te = load_2d(os.path.join(out_dir, col, 'yhat_' + suffix + '.txt')) # load_2d(os.path.join(idp_dir, 'yhat_' + suffix + '.txt'))
    s2_te = load_2d(os.path.join(idp_dir, 'ys2_' + suffix + '.txt'))
    warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1] 
    W = nm.blr.warp
    
    # warp predictions
    med_te = W.warp_predictions(np.squeeze(yhat_te), np.squeeze(s2_te), warp_param)[0]
    
    if med_te.ndim > 1:
        med_te = np.squeeze(med_te)

    errors[col] = med_te
    
average_predictions = pd.DataFrame(errors)

if len(df_test) != len(average_predictions):
    print("The length of data and average_predictions must be the same.")

average_predictions.index = df_test.index


# Compute residuals for the entire dataset
residual_all = df_test[idp_ids].values - average_predictions[idp_ids].values

# Convert residuals to DataFrames for further processing if needed

residual_all = pd.DataFrame(residual_all, index=df_test.index, columns=idp_ids)

# Add 'set' and 'DX' columns back to the residual_all_df DataFrame

residual_all = pd.concat([df_test[['set', 'DX']], residual_all], axis=1)
numerical_cols = residual_all.columns.difference(['set', 'DX'])

# merge the Z-scores to one dataframe 
z_scores = load_z_scoes(idp_ids, out_dir, df_test)

selected_rows = z_scores[z_scores['set'].isin(['adni', 'aibl', 'jadni', 'delcode'])]

# Select the matched column names that are present in selected_rows
matched_column_names_selected = [col for col in idp_ids if col in selected_rows.columns]

# Calculate the mean of the matched columns, ignoring NaN values
selected_rows['ROI'] = selected_rows[matched_column_names_selected].mean(axis=1, skipna=True)

# Convert the 'DX' column to an ordered integer factor
dx_ordered = selected_rows['DX'].astype(pd.CategoricalDtype(categories=["CN", "MCI", "AD"], ordered=True)).cat.codes

# Calculate Spearman and Kendall correlation
spearman_corr, _ = spearmanr(dx_ordered, selected_rows['ROI'])
kendall_corr, _ = kendalltau(dx_ordered, selected_rows['ROI'])

res_stats = {
        'spearman': spearman_corr,
        'kendall': kendall_corr
    }
res_stats_df = pd.DataFrame([res_stats])

print("Correlation statistics: ", res_stats)
output_file = os.path.join(out_dir, "analysis",f'correlation_alldata.csv')


# Calculate means and standard deviations for numerical columns grouped by 'set' and 'DX'
results = residual_all.groupby(['set', 'DX']).agg(
    {col: ['mean', 'std'] for col in numerical_cols}
).reset_index()


output_file = os.path.join(out_dir, 'analysis', f'Mean_std_testall.txt')
results.columns = ['_'.join(col).strip() if col[1] else col[0] for col in results.columns.values]

# Save the results to a CSV file
results.to_csv(output_file, index=False)

print(f'Mean and Standard deviations calculated and saved to {output_file}')

numerical_cols = idp_ids
results_mae = residual_all.groupby(['set', 'DX']).agg(
    {col: lambda x: np.mean(np.abs(x)) for col in numerical_cols}
).reset_index()

# Find matched column names
matched_column_names = [col for col in results_mae.columns if any(pattern in col for pattern in idp_ids)]

# Calculate ROI
results_mae['ROI'] = results_mae[matched_column_names].mean(axis=1)

# Relocate ROI column to be after DX
cols = results_mae.columns.tolist()
roi_index = cols.index('ROI')
dx_index = cols.index('DX')
cols.insert(dx_index + 1, cols.pop(roi_index))
results_mae = results_mae[cols]
results_mae.to_csv(os.path.join(out_dir, f'MAE_{suffix}.csv'), index=False)

# Save results to a CSV file
results_mae.to_csv(os.path.join(out_dir, f'MAE_{suffix}.csv'), index=False)

# Calculate reconstruction error and its statistics
residual_all['err'] = residual_all[numerical_cols].abs().mean(axis=1)
rec_error_grouped = residual_all.groupby(['set', 'DX']).agg(
    rec_error_mean=('err', lambda x: np.mean(x)),
    rec_error_std=('err', lambda x: np.std(x))
).reset_index()

# Save reconstruction error statistics to a CSV file
rec_error_grouped.to_csv(os.path.join(out_dir, f'Total_MAE_{suffix}.csv'), index=False)

#
results_count = z_scores.groupby(['set', 'DX']).agg(
    {col: lambda x: (x < -2).sum() / len(x) * 100 for col in numerical_cols}
).reset_index()

# Rename columns to indicate they are counts
results_count.columns = [f"{col}_count" if col not in ['set', 'DX'] else col for col in results_count.columns]

# Find matched column names
matched_column_names = [col for col in results_count.columns if any(pattern in col for pattern in idp_ids)]

# Remove the '_count' suffix from matched column names
matched_column_names_no_suffix = [col.replace('_count', '') for col in matched_column_names]

# Calculate ROI as the row means of matched columns
results_count['ROI'] = results_count[matched_column_names].mean(axis=1)

# Relocate ROI column to be after DX
cols = results_count.columns.tolist()
roi_index = cols.index('ROI')
dx_index = cols.index('DX')
cols.insert(dx_index + 1, cols.pop(roi_index))
results_count = results_count[cols]

print("Percentage of subjects with Z_score lower than -2: ", results_count)
# Save results to a CSV file
results_count.to_csv(os.path.join(out_dir, f'Perc_{suffix}_l-2.csv'), index=False)


# Define the directory structure
norm_curves_dir = os.path.join(out_dir, 'analysis', 'norm_curves')
male_dir = os.path.join(norm_curves_dir, 'male')
female_dir = os.path.join(norm_curves_dir, 'female')

# Create directories if they do not exist
os.makedirs(male_dir, exist_ok=True)
os.makedirs(female_dir, exist_ok=True)

def save_plots(sex, output_dir):
    xmin = 30
    xmax = 100
    if sex == 1:
        clr = 'red'
    else:
        clr = 'blue'

    # Create dummy data for visualization
    xx = np.arange(xmin, xmax, 0.5)
    X0_dummy = np.zeros((len(xx), 2))
    X0_dummy[:, 0] = xx
    X0_dummy[:, 1] = sex
    X_dummy = create_design_matrix(X0_dummy, 
                                   xmin=xmin, 
                                   xmax=xmax, 
                                   site_ids=None, 
                                   all_sites=site_ids_tr)
    cov_file_dummy = os.path.join(out_dir, 'cov_bspline_dummy_mean.txt')
    np.savetxt(cov_file_dummy, X_dummy)
    sns.set(style='whitegrid')
    
    # Create the design matrix using the sampled DataFrame
    X_te = create_design_matrix(df_patients[cols_cov],
                                site_ids=df_patients['set'],
                                all_sites=site_ids_tr,
                                basis='bspline',
                                xmin=xmin,
                                xmax=xmax)
    plt.figure(figsize=(15, 6))

    for idp_num, idp in enumerate(idp_ids):
        plt.subplot(2, 3, idp_num + 1)
        idp_dir = os.path.join(out_dir, idp)
        os.chdir(idp_dir)
        yhat_te = load_2d(os.path.join(idp_dir, 'yhat_' + suffix + '.txt'))
        s2_te = load_2d(os.path.join(idp_dir, 'ys2_' + suffix + '.txt'))
        y_te = load_2d(os.path.join(idp_dir, 'resp_te.txt'))
        yhat, s2 = predict(cov_file_dummy, 
                           alg='blr', 
                           respfile=None, 
                           model_path=os.path.join(idp_dir, 'Models'), 
                           outputsuffix='_dummy')
        print("mean yhat is: ", np.mean(yhat))
        with open(os.path.join(idp_dir, 'Models', 'NM_0_0_estimate.pkl'), 'rb') as handle:
            nm = pickle.load(handle)
        
        W = nm.blr.warp
        warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params() + 1]
        med_te = W.warp_predictions(np.squeeze(yhat_te), np.squeeze(s2_te), warp_param)[0]
        med_te = med_te[:, np.newaxis]
        evaluate(y_te, med_te)
        med, pr_int = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param)
        beta, _, _ = nm.blr._parse_hyps(nm.blr.hyp, X_dummy)
        s2n = 1 / beta
        s2s = s2 - s2n
        y_te_rescaled_all = np.zeros_like(y_te)
        for sid, site in enumerate(site_ids_te):
            if all(elem in site_ids_tr for elem in site_ids_te):
                idx = np.where(np.bitwise_and(X_te[:, 2] == sex, X_te[:, sid + len(cols_cov) + 1] != 0))[0]
                if len(idx) == 0:
                    continue
                idx_dummy = np.bitwise_and(X_dummy[:, 1] > X_te[idx, 1].min(), X_dummy[:, 1] < X_te[idx, 1].max())
                y_te_rescaled = df_test[idp][idx]
            else:
                idx = np.where(np.bitwise_and(X_te[:, 2] == sex, (df_patients['set'] == site).to_numpy()))[0]
                y_ad = load_2d(os.path.join(idp_dir, 'resp_ad.txt'))
                X_ad = load_2d(os.path.join(idp_dir, 'cov_bspline_ad.txt'))
                idx_a = np.where(np.bitwise_and(X_ad[:, 2] == sex, (df_ad['set'] == site).to_numpy()))[0]
                if len(idx) < 2 or len(idx_a) < 2:
                    continue
                y_te_rescaled, s2_rescaled = nm.blr.predict_and_adjust(nm.blr.hyp, X_ad[idx_a, :], np.squeeze(y_ad[idx_a]), Xs=None, ys=np.squeeze(y_te[idx]))
                idx_dummy = np.bitwise_and(X_dummy[:, 1] > X_te[idx, 1].min(), X_dummy[:, 1] < X_te[idx, 1].max())
                y_te_rescaled = y_te_rescaled + np.median(y_te[idx]) - np.median(med[idx_dummy])
            
            # uncomment for plotting AD patients into the growth curves
            #plt.scatter(X_te[idx, 1], y_te_rescaled, s=4, color=clr, alpha=0.1)
        
        # rescale the y-axis to match the actual values 
        scale_factor = np.median(y_te) / np.median(med)
        med_scaled = med * scale_factor
        pr_int_scaled = [pi * scale_factor for pi in pr_int]
        plt.plot(xx, med_scaled, clr)
        junk, pr_int25 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.25, 0.75])
        junk, pr_int95 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.05, 0.95])
        pr_int25_scaled = np.array([pi * scale_factor for pi in pr_int25])
        pr_int95_scaled = np.array([pi * scale_factor for pi in pr_int95])
        plt.fill_between(xx, pr_int25_scaled[:, 0], pr_int25_scaled[:, 1], alpha=0.1, color=clr)
        plt.fill_between(xx, pr_int95_scaled[:, 0], pr_int95_scaled[:, 1], alpha=0.1, color=clr)
        junk, pr_int25l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2 - 0.5 * s2s), warp_param, percentiles=[0.25, 0.75])
        junk, pr_int95l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2 - 0.5 * s2s), warp_param, percentiles=[0.05, 0.95])
        pr_int25l_scaled = np.array([pi * scale_factor for pi in pr_int25l])
        pr_int95l_scaled = np.array([pi * scale_factor for pi in pr_int95l])
        junk, pr_int25u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2 + 0.5 * s2s), warp_param, percentiles=[0.25, 0.75])
        junk, pr_int95u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2 + 0.5 * s2s), warp_param, percentiles=[0.05, 0.95])
        pr_int25u_scaled = np.array([pi * scale_factor for pi in pr_int25u])
        pr_int95u_scaled = np.array([pi * scale_factor for pi in pr_int95u])
        plt.fill_between(xx, pr_int25l_scaled[:, 0], pr_int25u_scaled[:, 0], alpha=0.3, color=clr)
        plt.fill_between(xx, pr_int95l_scaled[:, 0], pr_int95u_scaled[:, 0], alpha=0.3, color=clr)
        plt.fill_between(xx, pr_int25l_scaled[:, 1], pr_int25u_scaled[:, 1], alpha=0.3, color=clr)
        plt.fill_between(xx, pr_int95l_scaled[:, 1], pr_int95u_scaled[:, 1], alpha=0.3, color=clr)
        plt.plot(xx, pr_int25_scaled[:, 0], color=clr, linewidth=0.5)
        plt.plot(xx, pr_int25_scaled[:, 1], color=clr, linewidth=0.5)
        plt.plot(xx, pr_int95_scaled[:, 0], color=clr, linewidth=0.5)
        plt.plot(xx, pr_int95_scaled[:, 1], color=clr, linewidth=0.5)

        """plt.xlabel('Age')
        plt.ylabel(idp)
        plt.title(idp)
        plt.xlim((50, 90))
        plt.savefig(os.path.join(output_dir, f'centiles_{idp}.png'), bbox_inches='tight')
        plt.close()"""
        plt.xlabel('Age')
        plt.ylabel('Thickness')
        plt.title(idp.replace('_', ' ').title())
        plt.xlim((50, 90))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'combined_plots_sex_{sex}_noPatients.png'), bbox_inches='tight')
    plt.close()

save_plots(0, male_dir)  # For males
save_plots(1, female_dir)  # For females

os.chdir(out_dir)
