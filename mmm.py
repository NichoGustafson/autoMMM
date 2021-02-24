# Implementation of Multiplicative Marketing Mix Model, Adstock and Diminishing Return
# Author: Sibyl He and Nicholas Gustafson

# Pystan Installation Tips (mac, anaconda3)
# 1. In bash:    
# (create a stan environment, install pystan, current version is 2.19)
# conda create -n stan_env python=3.7 -c conda-forge
# conda activate stan_env
# conda install pystan -c conda-forge
# (install gcc5, pystan 2.19 requires gcc4.9.3 and above)
# brew install gcc@5
# (look for 'gcc-10', 'g++-10')
# ls /usr/local/bin | grep gcc
# ls /usr/local/bin | grep g++
#     
# 2. Open Anaconda Navigator > Home > Applications on: select stan_env as environment, launch Notebook    
#     
# 3. In python:    
# import os
# os.environ['CC'] = 'gcc-10'
# os.environ['CXX'] = 'g++-10'``


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
import time
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pystan
import "helper_functions"

sns.color_palette("husl")
sns.set_style('darkgrid')

class autoMMM():
    """
    documentation goes here
    """
    def __init__(self, df):
        self.df = df
        self.columns = {}
        self.columns["mdip_cols"] = [col for col in df.columns if 'mdip_' in col]
        self.columns["mdsp_cols"] = [col for col in df.columns if 'mdsp_' in col]
        self.columns["me_cols"] = [col for col in df.columns if 'me_' in col]
        self.columns["st_cols"] = [col for col in df.columns if 'st_ct' == col]
        self.columns["mrkdn_cols"] = [col for col in df.columns if 'mrkdn_' in col]
        self.columns["hldy_cols"] = [col for col in df.columns if 'hldy_' in col]
        self.columns["seas_cols"] = [col for col in df.columns if 'seas_' in col]
        self.columns["sales_cols"] =[col for col in df.columns if 'sales' == col]
        update_base_cols()
        for col in self.columns:
            if col == "base_sales":
                print("WARNING: Your column named \"base_sales\" will be overwritten. Please rename it.")
            print("Found " + str(len(self.columns[col])) + " columns for " + col)


        self.ctrl_code = '''
            data {
            int N; // number of observations
            int K1; // number of positive predictors
            int K2; // number of positive/negative predictors
            real max_intercept; // restrict the intercept to be less than the minimum y
            matrix[N, K1] X1;
            matrix[N, K2] X2;
            vector[N] y; 
            }
            parameters {
            vector<lower=0>[K1] beta1; // regression coefficients for X1 (positive)
            vector[K2] beta2; // regression coefficients for X2
            real<lower=0, upper=max_intercept> alpha; // intercept
            real<lower=0> noise_var; // residual variance
            }
            model {
            // Define the priors
            beta1 ~ normal(0, 1); 
            beta2 ~ normal(0, 1); 
            noise_var ~ inv_gamma(0.05, 0.05 * 0.01);
            // The likelihood
            y ~ normal(X1*beta1 + X2*beta2 + alpha, sqrt(noise_var));
            }
            '''

        self.mmm_code = '''
            functions {
            // the adstock transformation with a vector of weights
            real Adstock(vector t, row_vector weights) {
                return dot_product(t, weights) / sum(weights);
            }
            }
            data {
            // the total number of observations
            int<lower=1> N;
            // the vector of sales
            real y[N];
            // the maximum duration of lag effect, in weeks
            int<lower=1> max_lag;
            // the number of media channels
            int<lower=1> num_media;
            // matrix of media variables
            matrix[N+max_lag-1, num_media] X_media;
            // vector of media variables' mean
            real mu_mdip[num_media];
            // the number of other control variables
            int<lower=1> num_ctrl;
            // a matrix of control variables
            matrix[N, num_ctrl] X_ctrl;
            }
            parameters {
            // residual variance
            real<lower=0> noise_var;
            // the intercept
            real tau;
            // the coefficients for media variables and base sales
            vector<lower=0>[num_media+num_ctrl] beta;
            // the decay and peak parameter for the adstock transformation of
            // each media
            vector<lower=0,upper=1>[num_media] decay;
            vector<lower=0,upper=ceil(max_lag/2)>[num_media] peak;
            }
            transformed parameters {
            // the cumulative media effect after adstock
            real cum_effect;
            // matrix of media variables after adstock
            matrix[N, num_media] X_media_adstocked;
            // matrix of all predictors
            matrix[N, num_media+num_ctrl] X;
            
            // adstock, mean-center, log1p transformation
            row_vector[max_lag] lag_weights;
            for (nn in 1:N) {
                for (media in 1 : num_media) {
                for (lag in 1 : max_lag) {
                    lag_weights[max_lag-lag+1] <- pow(decay[media], (lag - 1 - peak[media]) ^ 2);
                }
                cum_effect <- Adstock(sub_col(X_media, nn, media, max_lag), lag_weights);
                X_media_adstocked[nn, media] <- log1p(cum_effect/mu_mdip[media]);
                }
            X <- append_col(X_media_adstocked, X_ctrl);
            } 
            }
            model {
            decay ~ beta(3,3);
            peak ~ uniform(0, ceil(max_lag/2));
            tau ~ normal(0, 5);
            for (i in 1 : num_media+num_ctrl) {
                beta[i] ~ normal(0, 1);
            }
            noise_var ~ inv_gamma(0.05, 0.05 * 0.01);
            y ~ normal(tau + X * beta, sqrt(noise_var));
            }
            '''

        self.hill_code = '''
            functions {
            // the Hill function
            real Hill(real t, real ec, real slope) {
            return 1 / (1 + (t / ec)^(-slope));
            }
            }
            data {
            // the total number of observations
            int<lower=1> N;
            // y: vector of media contribution
            vector[N] y;
            // X: vector of adstocked media spending
            vector[N] X;
            }
            parameters {
            // residual variance
            real<lower=0> noise_var;
            // regression coefficient
            real<lower=0> beta_hill;
            // ec50 and slope for Hill function of the media
            real<lower=0,upper=1> ec;
            real<lower=0> slope;
            }
            transformed parameters {
            // a vector of the mean response
            vector[N] mu;
            for (i in 1:N) {
                mu[i] <- beta_hill * Hill(X[i], ec, slope);
            }
            }
            model {
            slope ~ gamma(3, 1);
            ec ~ beta(2, 2);
            beta_hill ~ normal(0, 1);
            noise_var ~ inv_gamma(0.05, 0.05 * 0.01); 
            y ~ normal(mu, sqrt(noise_var));
            }
            '''

    def update_base_cols(self):
        self.columns["base_vars"] = self.columns["me_cols"] + self.columns["st_cols" + self.columns["mrkdn_cols"
                                     + self.columns["hldy_cols"] + self.columns["self.seas_cols"]

    def set_columns(self, column_type, columns):
        self.columns[column_type] = columns
        self.update_base_cols()

    def correlation_plot(self):
        # EDA - correlation, distribution plots
        plt.figure(figsize=(24,20))
        sns.heatmap(df[self.columns["mdip_cols"] + self.columns["sales_cols"]].corr(), square=True, annot=True, vmax=1, vmin=-1, cmap='RdBu')

    def pairplot(self):
        plt.figure(figsize=(50,50))
        sns.pairplot(self.df[elf.columns["mdip_cols"] + self.columns["sales_cols"]], vars=mdip_cols+['sales'])


    def fit(self):
        self.fit_control()
        self.fit_mmm()
        self.mc_df = self.mmm_decompose_contrib(original_sales=df[self.columns["sales_cols"]])
        self.adstock_params = self.mmm['adstock_params']
        self.mc_pct, self.mc_pct2 = self.calc_media_contrib_pct(period=52)
    
    def fit_diminishing_returns(self):
        self.fit_hill()
        self.fit_roas()
    
    def summary(self)
        pass

    def diminishing_returns_summary():
        pass

    def fit_control(self):
        """
        2. Model Implementation
        The model is built in a stacked way. Three models are trained:   
        - Control Model
        - Marketing Mix Model
        - Diminishing Return Model    
        """
        # mean-centralize: sales, numeric base_vars
        df_ctrl, sc_ctrl = mean_center_trandform(self.df, self.columns['sales_cols'] + self.columns["me_cols" 
                                                + self.columns["st_cols"] + self.columns["mrkdn_cols"])
        df_ctrl = pd.concat([df_ctrl, df[self.columns["hldy_cols" + self.columns["seas_cols"]]]], axis=1)

        # variables positively related to sales: macro economy, store count, markdown, holiday
        pos_vars = [col for col in self.columns["base_vars"] if col not in self.columns["seas_cols"]]
        X1 = df_ctrl[pos_vars].values

        # variables may have either positive or negtive impact on sales: seasonality
        pn_vars = seas_cols
        X2 = df_ctrl[pn_vars].values

        ctrl_data = {
            'N': len(df_ctrl),
            'K1': len(pos_vars), 
            'K2': len(pn_vars), 
            'X1': X1,
            'X2': X2, 
            'y': df_ctrl['sales'].values,
            'max_intercept': min(df_ctrl['sales'])
        }

        sm1 = pystan.StanModel(model_code = self.ctrl_code, verbose=True)
        fit1 = sm1.sampling(data=ctrl_data, iter=2000, chains=4)
        fit1_result = fit1.extract()

        # extract control model parameters and predict base sales -> df['base_sales']
        def extract_ctrl_model(fit_result, pos_vars=pos_vars, pn_vars=pn_vars, 
                            extract_param_list=False):
            ctrl_model = {}
            ctrl_model['pos_vars'] = pos_vars
            ctrl_model['pn_vars'] = pn_vars
            ctrl_model['beta1'] = fit_result['beta1'].mean(axis=0).tolist()
            ctrl_model['beta2'] = fit_result['beta2'].mean(axis=0).tolist()
            ctrl_model['alpha'] = fit_result['alpha'].mean()
            if extract_param_list:
                ctrl_model['beta1_list'] = fit_result['beta1'].tolist()
                ctrl_model['beta2_list'] = fit_result['beta2'].tolist()
                ctrl_model['alpha_list'] = fit_result['alpha'].tolist()
            return ctrl_model

        def ctrl_model_predict(ctrl_model, df):
            pos_vars, pn_vars = ctrl_model['pos_vars'], ctrl_model['pn_vars'] 
            X1, X2 = df[pos_vars], df[pn_vars]
            beta1, beta2 = np.array(ctrl_model['beta1']), np.array(ctrl_model['beta2'])
            alpha = ctrl_model['alpha']
            y_pred = np.dot(X1, beta1) + np.dot(X2, beta2) + alpha
            return y_pred

        self.base_sales_model = extract_ctrl_model(fit1_result, pos_vars=pos_vars, pn_vars=pn_vars)
        self.base_sales = ctrl_model_predict(base_sales_model, df_ctrl)
        self.df['base_sales'] = self.base_sales*sc_ctrl['sales']
        # evaluate control model
        print('mape: ', mean_absolute_percentage_error(df['sales'], df['base_sales']))


    def fit_mmm(self):
        """
        doc string here
        """
        # 2.2 Marketing Mix Model
        df_mmm, sc_mmm = mean_log1p_trandform(self.df, [self.columns["sales_cols"], 'base_sales'])
        mu_mdip = df[self.columns["mdip_cols"]].apply(np.mean, axis=0).values
        max_lag = 8
        num_media = len(self.columns["mdip_cols"])
        # padding zero * (max_lag-1) rows
        X_media = np.concatenate((np.zeros((max_lag-1, num_media)), df[self.columns["mdip_cols"]].values), axis=0)
        X_ctrl = df_mmm['base_sales'].values.reshape(len(df),1)

        model_data2 = {
            'N': len(df),
            'max_lag': max_lag, 
            'num_media': num_media,
            'X_media': X_media, 
            'mu_mdip': mu_mdip,
            'num_ctrl': X_ctrl.shape[1],
            'X_ctrl': X_ctrl, 
            'y': df_mmm[self.columns["sales_cols"]].values
        }

        sm2 = pystan.StanModel(model_code=mmm_code, verbose=True)
        fit2 = sm2.sampling(data=model_data2, iter=1000, chains=3)
        fit2_result = fit2.extract()

        # extract mmm parameters
        def extract_mmm(fit_result, max_lag=max_lag, 
                        media_vars=mdip_cols, ctrl_vars=['base_sales'], 
                        extract_param_list=True):
            mmm = {}
            
            mmm['max_lag'] = max_lag
            mmm['media_vars'], mmm['ctrl_vars'] = media_vars, ctrl_vars
            mmm['decay'] = decay = fit_result['decay'].mean(axis=0).tolist()
            mmm['peak'] = peak = fit_result['peak'].mean(axis=0).tolist()
            mmm['beta'] = fit_result['beta'].mean(axis=0).tolist()
            mmm['tau'] = fit_result['tau'].mean()
            if extract_param_list:
                mmm['decay_list'] = fit_result['decay'].tolist()
                mmm['peak_list'] = fit_result['peak'].tolist()
                mmm['beta_list'] = fit_result['beta'].tolist()
                mmm['tau_list'] = fit_result['tau'].tolist()
            
            adstock_params = {}
            media_names = [col.replace('mdip_', '') for col in media_vars]
            for i in range(len(media_names)):
                adstock_params[media_names[i]] = {
                    'L': max_lag,
                    'P': peak[i],
                    'D': decay[i]
                }
            mmm['adstock_params'] = adstock_params
            return mmm

        self.mmm = extract_mmm(fit2, max_lag=max_lag, 
                        media_vars=self.columns["mdip_cols"], ctrl_vars=['base_sales'])


    

    # Decompose sales to media channels' contribution
    # Each media channel's contribution = total sales - sales upon removal the channel    
    # decompose sales to media contribution
    def mmm_decompose_contrib(self, original_sales=df['sales']):
        # adstock params
        adstock_params = self.mmm['adstock_params']
        # coefficients, intercept
        beta, tau = self.mmm['beta'], self.mmm['tau']
        # variables
        media_vars, ctrl_vars = self.mmm['media_vars'], self.mmm['ctrl_vars']
        num_media, num_ctrl = len(media_vars), len(ctrl_vars)
        # X_media2: adstocked, mean-centered media variables + 1
        X_media2 = adstock_transform(self.df, media_vars, adstock_params)
        X_media2, sc_mmm2 = mean_center_trandform(X_media2, media_vars)
        X_media2 = X_media2 + 1
        # X_ctrl2, mean-centered control variables + 1
        X_ctrl2, sc_mmm2_1 = mean_center_trandform(self.df[ctrl_vars], ctrl_vars)
        X_ctrl2 = X_ctrl2 + 1
        # y_true2, mean-centered sales variable + 1
        y_true2, sc_mmm2_2 = mean_center_trandform(self.df, self.columns["sales_cols"])
        y_true2 = y_true2 + 1
        sc_mmm2.update(sc_mmm2_1)
        sc_mmm2.update(sc_mmm2_2)
        # X2 <- media variables + ctrl variable
        X2 = pd.concat([X_media2, X_ctrl2], axis=1)

        # 1. compute each media/control factor: 
        # log-log model: log(sales) = log(X[0])*beta[0] + ... + log(X[13])*beta[13] + tau
        # multiplicative model: sales = X[0]^beta[0] * ... * X[13]^beta[13] * e^tau
        # each factor = X[i]^beta[i]
        # intercept = e^tau
        factor_df = pd.DataFrame(columns=media_vars+ctrl_vars+['intercept'])
        for i in range(num_media):
            colname = media_vars[i]
            factor_df[colname] = X2[colname] ** beta[i]
        for i in range(num_ctrl):
            colname = ctrl_vars[i]
            factor_df[colname] = X2[colname] ** beta[num_media+i]
        factor_df['intercept'] = np.exp(tau)

        # 2. calculate the product of all factors -> y_pred
        # baseline = intercept * control factor = e^tau * X[13]^beta[13]
        y_pred = factor_df.apply(np.prod, axis=1)
        factor_df['y_pred'], factor_df['y_true2'] = y_pred, y_true2
        factor_df['baseline'] = factor_df[['intercept']+ctrl_vars].apply(np.prod, axis=1)

        # 3. calculate each media factor's contribution
        # media contribution = total volume – volume upon removal of the media factor
        mc_df = pd.DataFrame(columns=media_vars+['baseline'])
        for col in media_vars:
            mc_df[col] = factor_df['y_true2'] - factor_df['y_true2']/factor_df[col]
        mc_df['baseline'] = factor_df['baseline']
        mc_df['y_true2'] = factor_df['y_true2']

        # 4. scale contribution
        # predicted total media contribution: product of all media factors
        mc_df['mc_pred'] = mc_df[media_vars].apply(np.sum, axis=1)
        # true total media contribution: total volume - baseline
        mc_df['mc_true'] = mc_df['y_true2'] - mc_df['baseline']
        # predicted total media contribution is slightly different from true total media contribution
        # scale each media factor’s contribution by removing the delta volume proportionally
        mc_df['mc_delta'] =  mc_df['mc_pred'] - mc_df['mc_true']
        for col in media_vars:
            mc_df[col] = mc_df[col] - mc_df['mc_delta']*mc_df[col]/mc_df['mc_pred']

        # 5. scale mc_df based on original sales
        mc_df['sales'] = original_sales
        for col in media_vars+['baseline']:
            mc_df[col] = mc_df[col]*mc_df['sales']/mc_df['y_true2']
        
        print('rmse (log-log model): ', 
            mean_squared_error(np.log(y_true2), np.log(y_pred)) ** (1/2))
        print('mape (multiplicative model): ', 
            mean_absolute_percentage_error(y_true2, y_pred))
        return mc_df

    # calculate media contribution percentage
    def calc_media_contrib_pct(self, period=52):
        '''
        returns:
        mc_pct: percentage over total sales
        mc_pct2: percentage over incremental sales (sales contributed by media channels)
        '''
        media_vars = self.columns["mdip_cols"]
        sales_col = self.columns["sales_cols"]
        mc_pct = {}
        mc_pct2 = {}
        s = 0
        if period is None:
            for col in (media_vars+['baseline']):
                mc_pct[col] = (self.mc_df[col]/self.mc_df[sales_col]).mean()
        else:
            for col in (media_vars+['baseline']):
                mc_pct[col] = (self.mc_df[col]/self.mc_df[sales_col])[-period:].mean()
        for m in media_vars:
            s += mc_pct[m]
        for m in media_vars:
            mc_pct2[m] = mc_pct[m]/s
        return mc_pct, mc_pct2

    
    def plot_media_coeff(self):
        # plot media coefficients' distributions
        # red line: mean, green line: median
        beta_media = {}
        for i in range(len(mmm['media_vars'])):
            md = self.mmm['media_vars'][i]
            betas = []
            for j in range(len(self.mmm['beta_list'])):
                betas.append(self.mmm['beta_list'][j][i])
            beta_media[md] = np.array(betas)

        f = plt.figure(figsize=(18,15))
        for i in range(len(mmm['media_vars'])):
            ax = f.add_subplot(5,3,i+1)
            md = mmm['media_vars'][i]
            x = beta_media[md]
            mean_x = x.mean()
            median_x = np.median(x)
            ax = sns.distplot(x)
            ax.axvline(mean_x, color='r', linestyle='-')
            ax.axvline(median_x, color='g', linestyle='-')
            ax.set_title(md)

    # 2.3 Diminishing Return Model    

    def fit_hill(self):
        # train hill models for all media channels
        sm3 = pystan.StanModel(model_code=hill code, verbose=True)
        self.hill_models = {}
        to_train = self.columns["mdsp_cols"]
        for media in to_train:
            print('training for media: ', media)
            hill_model = self.train_hill_model(media, sm3)
            self.hill_models[media] = hill_model

        # extract params by mean
        self.hill_model_params_mean, self.hill_model_params_med = {}, {}
        for md in list(self.hill_models.keys()):
            hill_model = self.hill_models[md]
            params1 = self.extract_hill_model_params(hill_model, method='mean')
            params1['sc'] = hill_model['sc']
            params2 = self.extract_hill_model_params(hill_model, method='median')
            params2 = hill_model['sc']
            self.hill_model_params_mean[md] = params1
            self.hill_model_params_med[md] = params2

        # evaluate model params extracted by mean
        for md in list(self.hill_models.keys()):
            print('evaluating media: ', md)
            hill_model = self.hill_models[md]
            hill_model_params = hill_model_params_mean[md]
            _ = evaluate_hill_model(hill_model, hill_model_params)
        


    def create_hill_model_data(self, media):
        y = self.mc_df['mdip_'+media].values
        L, P, D = adstock_params[media]['L'], adstock_params[media]['P'], adstock_params[media]['D']
        x = self.df['mdsp_'+media].values
        x_adstocked = apply_adstock(x, L, P, D)
        # centralize
        mu_x, mu_y = x_adstocked.mean(), y.mean()
        sc = {'x': mu_x, 'y': mu_y}
        x = x_adstocked/mu_x
        y = y/mu_y
            
        model_data = {
            'N': len(y),
            'y': y,
            'X': x
        }
        return model_data, sc

    # pipeline for training one hill model for a media channel
    def train_hill_model(self, media, sm):
        '''
        params:
        df: original data
        mc_df: media contribution df derived from MMM
        adstock_params: adstock parameter dict output by MMM
        media: 'dm', 'inst', 'nsp', 'auddig', 'audtr', 'vidtr', 'viddig', 'so', 'on', 'sem'
        sm: stan model object    
        returns:
        a dict of model data, scaler, parameters
        '''
        data, sc = self.create_hill_model_data(media)
        fit = sm.sampling(data=data, iter=2000, chains=4)
        fit_result = fit.extract()
        hill_model = {
            'beta_hill_list': fit_result['beta_hill'].tolist(),
            'ec_list': fit_result['ec'].tolist(),
            'slope_list': fit_result['slope'].tolist(),
            'sc': sc,
            'data': {
                'X': data['X'].tolist(),
                'y': data['y'].tolist(),
            }
        }
        return hill_model

    # extract params by mean or median
    # almost no difference, choose either one
    def extract_hill_model_params(hill_model, method='mean'):
        if method=='mean':
            hill_model_params = {
                'beta_hill': np.mean(hill_model['beta_hill_list']), 
                'ec': np.mean(hill_model['ec_list']), 
                'slope': np.mean(hill_model['slope_list'])
            }
        elif method=='median':
            hill_model_params = {
                'beta_hill': np.median(hill_model['beta_hill_list']), 
                'ec': np.median(hill_model['ec_list']), 
                'slope': np.median(hill_model['slope_list'])
            }
        return hill_model_params

    def hill_model_predict(hill_model_params, x):
        beta_hill, ec, slope = hill_model_params['beta_hill'], hill_model_params['ec'], hill_model_params['slope']
        y_pred = beta_hill * hill_transform(x, ec, slope)
        return y_pred

    def evaluate_hill_model(hill_model, hill_model_params):
        x = np.array(hill_model['data']['X'])
        y_true = np.array(hill_model['data']['y']) * hill_model['sc']['y']
        y_pred = hill_model_predict(hill_model_params, x) * hill_model['sc']['y']
        print('mape on original data: ', 
            mean_absolute_percentage_error(y_true, y_pred))
        return y_true, y_pred

    def plot_ec_dist(self):
        # plot ec distribution
        f = plt.figure(figsize=(18,12))
        hm_keys = list(self.hill_models.keys())
        for i in range(len(hm_keys)):
            plt.tight_layout()
            ax = f.add_subplot(4,3,i+1)
            md = hm_keys[i]
            x = self.hill_models[md]['ec_list']
            mean_x = np.mean(x)
            median_x = np.median(x)
            ax = sns.distplot(x)
            ax.axvline(mean_x, color='r', linestyle='-', alpha=0.5)
            ax.axvline(median_x, color='g', linestyle='-', alpha=0.5)
            ax.set_title(md)


    def plot_slope_dist(self):
        # plot slope distribution
        f = plt.figure(figsize=(18,12))
        hm_keys = list(self.hill_models.keys())
        for i in range(len(hm_keys)):
            plt.tight_layout()
            ax = f.add_subplot(4,3,i+1)
            md = hm_keys[i]
            x = self.hill_models[md]['slope_list']
            mean_x = np.mean(x)
            median_x = np.median(x)
            ax = sns.distplot(x)
            ax.axvline(mean_x, color='r', linestyle='-', alpha=0.5)
            ax.axvline(median_x, color='g', linestyle='-', alpha=0.5)
            ax.set_title(md)


    def plot_hill_function(self):
        # plot fitted hill function
        f = plt.figure(figsize=(18,16))
        hm_keys = list(self.hill_models.keys())
        for i in range(len(hm_keys)):
            plt.tight_layout()
            ax = f.add_subplot(4,3,i+1)
            md = hm_keys[i]
            hm = self.hill_models[md]
            hmp = self.hill_model_params_mean[md]
            x, y = hm['data']['X'], hm['data']['y']
            #mu_x, mu_y = hm['sc']['x'], hm['sc']['y']
            ec, slope = hmp['ec'], hmp['slope']
            x_sorted = np.array(sorted(x))
            y_fit = self.hill_model_predict(hmp, x_sorted)
            ax = sns.scatterplot(x=x, y=y, alpha=0.2)
            ax = sns.lineplot(x=x_sorted, y=y_fit, color='r', 
                        label='ec=%.2f, slope=%.2f'%(ec, slope))
            ax.set_title(md)


    def fit_roas(self)
        # Calculate overall ROAS and weekly ROAS
        # - Overall ROAS = total contribution / total spending
        # - Weekly ROAS = weekly contribution / weekly spending

        # adstocked media spending
        ms_df = pd.DataFrame()
        for md in list(self.hill_models.keys()):
            hill_model = hill_models[md]
            x = np.array(hill_model['data']['X']) * hill_model['sc']['x']
            self.ms_df['mdsp_'+md] = x
        # ms_df.to_csv('ms_df1.csv', index=False)

        # calc overall ROAS of a given period
        def calc_roas(mc_df, ms_df, period=None):
            roas = {}
            md_names = [col.split('_')[-1] for col in ms_df.columns]
            for i in range(len(md_names)):
                md = md_names[i]
                sp, mc = ms_df['mdsp_'+md], mc_df['mdip_'+md]
                if period is None:
                    md_roas = mc.sum()/sp.sum()
                else:
                    md_roas = mc[-period:].sum()/sp[-period:].sum()
                roas[md] = md_roas
            return roas

        # calc weekly ROAS
        def calc_weekly_roas(mc_df, ms_df):
            weekly_roas = pd.DataFrame()
            md_names = [col.split('_')[-1] for col in ms_df.columns]
            for md in md_names:
                weekly_roas[md] = mc_df['mdip_'+md]/ms_df['mdsp_'+md]
            weekly_roas.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
            return weekly_roas

        self.roas_1y = calc_roas(self.mc_df, self.ms_df, period=52)
        self.weekly_roas = calc_weekly_roas(self.mc_df, self.ms_df)
        self.roas1y_df = pd.DataFrame(index=self.weekly_roas.columns.tolist())
        self.roas1y_df['roas_mean'] = self.weekly_roas[-52:].apply(np.mean, axis=0)
        self.roas1y_df['roas_median'] = self.weekly_roas[-52:].apply(np.median, axis=0)


        def calc_mroas(hill_model, hill_model_params, period=52):
            '''
            calculate mROAS for a media
            params:
            hill_model: a dict containing model data and scaling factor
            hill_model_params: a dict containing beta_hill, ec, slope
            period: in weeks, the period used to calculate ROAS and mROAS. 52 is last one year.
            return:
            mROAS value
            '''
            mu_x, mu_y = hill_model['sc']['x'], hill_model['sc']['y']
            # get current media spending level over the period specified
            cur_sp = np.asarray(hill_model['data']['X'])
            if period is not None:
                cur_sp = cur_sp[-period:]
            cur_mc = sum(self.hill_model_predict(hill_model_params, cur_sp) * mu_y)
            # next spending level: increase by 1%
            next_sp = cur_sp * 1.01
            # media contribution under next spending level
            next_mc = sum(self.hill_model_predict(hill_model_params, next_sp) * mu_y)
            
            # mROAS
            delta_mc = next_mc - cur_mc
            delta_sp = sum(next_sp * mu_x) - sum(cur_sp * mu_x)
            mroas = delta_mc/delta_sp
            return mroas

        # calc mROAS of recent 1 year
        self.mroas_1y = {}
        for md in list(self.hill_models.keys()):
            hill_model = hill_models[md]
            hill_model_params = self.hill_model_params_mean[md]
            mroas_1y[md] = calc_mroas(hill_model, hill_model_params, period=52)


        self.roas1y_df = pd.concat([
            self.roas1y_df[['roas_mean', 'roas_median']],
            pd.DataFrame.from_dict(self.mroas_1y, orient='index', columns=['mroas']),
            pd.DataFrame.from_dict(self.roas_1y, orient='index', columns=['roas_avg'])
        ], axis=1)

    def plot_weekly_roas(self)
        # plot weekly ROAS distribution of past 1 year
        # median: green line, mean: red line
        f = plt.figure(figsize=(18,12))
        for i in range(len(weekly_roas.columns)):
            md = self.weekly_roas.columns[i]
            ax = f.add_subplot(4,3,i+1)
            x = self.weekly_roas[md][-52:]
            mean_x = np.mean(x)
            median_x = np.median(x)
            ax = sns.distplot(x)
            ax.axvline(mean_x, color='r', linestyle='-', alpha=0.5)
            ax.axvline(median_x, color='g', linestyle='-', alpha=0.5)
            ax.set(xlabel=None)
            ax.set_title(md)
