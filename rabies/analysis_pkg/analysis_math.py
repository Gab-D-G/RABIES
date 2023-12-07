import numpy as np

def vcorrcoef(X, y):  # return a correlation between each row of X with y
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym), axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2, axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r


def elementwise_corrcoef(X, Y):
    # X and Y are each of shape num_observations X num_element
    # computes the correlation between each element of X and Y
    Xm = X.mean(axis=0)
    Ym = Y.mean(axis=0)
    r_num = np.sum((X-Xm)*(Y-Ym), axis=0)

    r_den = np.sqrt(np.sum((X-Xm)**2, axis=0)*np.sum((Y-Ym)**2, axis=0))
    r = r_num/r_den
    return r


def elementwise_spearman(X,Y):    
    order = X.argsort(axis=0)
    X_ranks = order.argsort(axis=0)
    order = Y.argsort(axis=0)
    Y_ranks = order.argsort(axis=0)
    return elementwise_corrcoef(X_ranks, Y_ranks)
    

def dice_coefficient(mask1,mask2):
    dice = np.sum(mask1*mask2)*2.0 / (np.sum(mask1) + np.sum(mask2))
    return dice


'''
LINEAR REGRESSION --- CLOSED-FORM SOLUTION
'''


def closed_form(X, Y, intercept=False):  # functions that computes the Least Squares Estimates
    if intercept:
        X = np.concatenate((X, np.ones([X.shape[0], 1])), axis=1)
    return np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)


def mse(X, Y, w):  # function that computes the Mean Square Error (MSE)
    return np.mean((Y-np.matmul(X, w))**2)


def dual_regression(all_IC_vectors, timeseries):
    ### compute dual regression
    ### Here, we adopt an approach where the algorithm should explain the data
    ### as a linear combination of spatial maps. The data itself, is only temporally
    ### detrended, and not spatially centered, which could cause inconsistencies during
    ### linear regression according to https://mandymejia.com/2018/03/29/the-role-of-centering-in-dual-regression/#:~:text=Dual%20regression%20requires%20centering%20across%20time%20and%20space&text=time%20points.,each%20time%20course%20at%20zero
    ### The fMRI timeseries aren't assumed theoretically to be spatially centered, and
    ### this measure would be removing global signal variations which we are interested in.
    ### Thus we prefer to avoid this step here, despite modelling limitations.
    X = all_IC_vectors.T
    Y = timeseries.T
    # for one given volume, it's values can be expressed through a linear combination of the components
    W = closed_form(X, Y, intercept=False).T
    W /= np.sqrt((W ** 2).mean(axis=0)) # the temporal domain is variance-normalized so that the weights are contained in the spatial maps

    # for a given voxel timeseries, it's signal can be explained a linear combination of the component timecourses
    C = closed_form(W, Y.T, intercept=False).T

    S = np.sqrt((C ** 2).mean(axis=0)) # the component variance/scaling is taken from the spatial maps
    C /= S # the spatial maps are variance normalized; the variance is stored in S

    # we thus output a model of the timeseries of the form X = W.dot((S*C).T)
    DR = {'C':C, 'W':W, 'S':S}
    return DR


'''
Convergence through alternating minimization using OLS
'''

from sklearn.utils import check_random_state

def dual_OLS_fit(X, q=1, c_init=None, C_prior=None, W_prior=None, tol=1e-6, max_iter=200, verbose=1):
    # X: time by voxel matrix
    # q: number of new components to fit
    # c_init: can specify an voxel by component number matrix for initiating weights
    # C_prior: a voxel by component matrix of priors that are included in the fitting, but fixed as constant components
    
    # q defines the number of new sources to fit
    if c_init is None:
        random_state = check_random_state(None)
        c_init = random_state.normal(
            size=(X.shape[1], q))
        
    if C_prior is None:
        C_prior = np.zeros([X.shape[1], 0])
    C_prior /= np.sqrt((C_prior ** 2).sum(axis=0))
    
    C = c_init
    C /= np.sqrt((C ** 2).sum(axis=0))
    
    if W_prior is None:
        W_prior = np.zeros([X.shape[0], 0])
    W_prior /= np.sqrt((W_prior ** 2).sum(axis=0))
    num_prior_W = W_prior.shape[1]
    
    Cw = closed_form(W_prior, X).T # compute an initial C representation of W_prior
    
    for i in range(max_iter):
        C_prev = C
        C_ = np.concatenate((C, C_prior, Cw), axis=1) # add in the prior to contribute to the fitting
        
        ##### first OLS convergence step
        W = closed_form(C_, X.T).T
        
        if num_prior_W>0:
            W[:,-num_prior_W:] = W_prior # add back W_prior
               
        ##### second OLS convergence step
        C_ = closed_form(W, X).T
        C_ /= np.sqrt((C_ ** 2).sum(axis=0))
        
        if num_prior_W>0:
            Cw = C_[:,-num_prior_W:] # update Cw
            
        C = C_[:,:q] # take out the fitted components

        ##### evaluate convergence
        if q<1: # break if no new component is being fitted
            break
        lim = np.abs(np.abs((C * C_prev).sum(axis=0)) - 1).mean()
        if verbose > 2:
            print('lim:'+str(lim))
        if lim < tol:
            if verbose > 1:
                print(str(i)+' iterations to converge.')
            break
        if i == max_iter-1:
            if verbose > 0:
                print(
                    'Convergence failed. Consider increasing max_iter or decreasing tol.')
    return C, C_,W


def spatiotemporal_prior_fit(X, C_prior, num_W, num_C):
    num_priors = C_prior.shape[1]
    # first fit data-driven temporal components
    W_extra,W_, C = dual_OLS_fit(X.T, q=num_W, c_init=None, C_prior=None, W_prior = C_prior, tol=1e-6, max_iter=200, verbose=1)
    # second fit data-driven spatial components
    C_extra,C_, W = dual_OLS_fit(X, q=num_C, c_init=None, C_prior=C_prior, W_prior = W_extra, tol=1e-6, max_iter=200, verbose=1)
    Cw = C_[:,num_priors+num_C:] # take out the Cw before fitting the priors, so that it's estimation was not biased by any particular prior

    C_fitted_prior = np.zeros([X.shape[1], num_priors])
    for i in range(num_priors):
        prior = C_prior[:,i] # the prior that will be fitted
        C_extra_prior = np.concatenate((C_extra, C_prior[:,:i], C_prior[:,i+1:]), axis=1) # combine previously-fitted extra components with priors not getting fitted
        C_fit,C_, W_fit = dual_OLS_fit(X, q=1, c_init=None, C_prior=C_extra_prior, W_prior = W_extra, tol=1e-6, max_iter=200, verbose=1)

        C_fitted_prior[:,i] = C_fit[:,0]

        corr = np.corrcoef(C_fitted_prior[:,i].T, prior.T)[0,1]
        if corr<0: # if the correlation is negative, invert the weights on the fitted component
            C_fitted_prior[:,i]*=-1

    # estimate a final set for C and W, offering a final linear fit to X = WtC
    C = np.concatenate((C_fitted_prior, C_extra, Cw), axis=1)
    W = closed_form(C, X.T).T
    if num_W>0:
        W[:,-num_W:] = W_extra # add back W_prior
    W /= np.sqrt((W ** 2).mean(axis=0)) # the temporal domain is variance-normalized so that the weights are contained in the spatial maps
    C = closed_form(W, X).T

    S = np.sqrt((C ** 2).mean(axis=0)) # the component variance/scaling is taken from the spatial maps
    C /= S # the spatial maps are variance normalized; the variance is stored in S

    # Fitted priors are at the first indices
    C_fitted_prior = C[:,:num_priors]
    W_fitted_prior = W[:,:num_priors]
    S_fitted_prior = S[:num_priors]
        
    # temporal components are at the last indices
    C_temporal = C[:,num_priors+num_C:]
    W_temporal = W[:,num_priors+num_C:]
    S_temporal = S[num_priors+num_C:]

    # spatial components are in the middle
    C_spatial = C[:,num_priors:num_priors+num_C]
    W_spatial = W[:,num_priors:num_priors+num_C]
    S_spatial = S[num_priors:num_priors+num_C]

    corr_list=[]
    for i in range(num_priors):
        corr = np.corrcoef(C_fitted_prior[:,i].T, C_prior[:,i].T)[0,1]
        corr_list.append(corr)

    # we thus output a model of the timeseries of the form X = W.dot((S*C).T)
    return {'C_fitted_prior':C_fitted_prior, 'C_spatial':C_spatial, 'C_temporal':C_temporal, 
            'W_fitted_prior':W_fitted_prior, 'W_spatial':W_spatial, 'W_temporal':W_temporal, 
            'S_fitted_prior':S_fitted_prior, 'S_spatial':S_spatial, 'S_temporal':S_temporal, 
            'corr_list':corr_list}


def spatiotemporal_C_PCA_DR(X, C_prior, num_W, num_C):
    C_prior = C_prior.copy()
    num_priors = C_prior.shape[1]
    # first fit data-driven temporal components
    W_extra,W_, C = dual_OLS_fit(X.T, q=num_W, c_init=None, C_prior=None, W_prior = C_prior, tol=1e-6, max_iter=200, verbose=1)
    # second fit data-driven spatial components
    C_extra,C_, W = dual_OLS_fit(X, q=num_C, c_init=None, C_prior=C_prior, W_prior = W_extra, tol=1e-6, max_iter=200, verbose=1)
    Cw = C_[:,num_priors+num_C:] # the spatial part of W_extra
    #Wc = W[:,:num_C] # the temporal part of C_extra

    # estimate a final set for C and W, offering a final linear fit to X = WtC
    C = np.concatenate((C_prior, C_extra, Cw), axis=1)
    C /= np.sqrt((C ** 2).mean(axis=0)) # normalized prior to regression to that S1 is standardized

    W = closed_form(C, X.T).T
    if num_W>0:
        W[:,-num_W:] = W_extra # add back W_prior
        
    S_pre = np.sqrt((W ** 2).mean(axis=0)) # store variance when using the prior as regressor
    S_prior = S_pre[:num_priors]    
    W /= S_pre # the temporal domain is variance-normalized so that the weights are contained in the spatial maps
    C = closed_form(W, X).T

    S = np.sqrt((C ** 2).mean(axis=0)) # the component variance/scaling is taken from the spatial maps
    C /= S # the spatial maps are variance normalized; the variance is stored in S

    # Fitted priors are at the first indices
    C_fitted_prior = C[:,:num_priors]
    W_fitted_prior = W[:,:num_priors]
    S_fitted_prior = S[:num_priors]    
    
    # temporal components are at the last indices
    C_temporal = C[:,num_priors+num_C:]
    W_temporal = W[:,num_priors+num_C:]
    S_temporal = S[num_priors+num_C:]

    # spatial components are in the middle
    C_spatial = C[:,num_priors:num_priors+num_C]
    W_spatial = W[:,num_priors:num_priors+num_C]
    S_spatial = S[num_priors:num_priors+num_C]
    
    corr_list=[]
    for i in range(num_priors):
        corr = np.corrcoef(C_fitted_prior[:,i].T, C_prior[:,i].T)[0,1]
        corr_list.append(corr)

    # we thus output a model of the timeseries of the form X = W.dot((S*C).T)
    return {'C_fitted_prior':C_fitted_prior, 'C_spatial':C_spatial, 'C_temporal':C_temporal, 
            'W_fitted_prior':W_fitted_prior, 'W_spatial':W_spatial, 'W_temporal':W_temporal, 
            'S_fitted_prior':S_fitted_prior, 'S_spatial':S_spatial, 'S_temporal':S_temporal, 
            'S_prior':S_prior, 'corr_list':corr_list}
