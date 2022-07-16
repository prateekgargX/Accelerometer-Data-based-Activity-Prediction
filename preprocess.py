import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks

def create_windows(df_train_data,df_train_labels,window_size = 50,step_size = 10):
    x_list = []
    y_list = []
    z_list = []
    train_labels = []

    # creating overlaping windows of size window-size 100
    for i in range(0, df_train_data.shape[0] - window_size+1, step_size):
        xs = df_train_data['x'].values[i: i + window_size]
        ys = df_train_data['y'].values[i: i + window_size]
        zs = df_train_data['z'].values[i: i + window_size]
        label = df_train_labels.loc[df_train_data.iloc[[i + window_size-1]].index[0]]['label']

        x_list.append(xs)
        y_list.append(ys)
        z_list.append(zs)
        train_labels.append(label)

    return x_list,y_list,z_list,train_labels

def transform_windows(x_list,y_list,z_list):
    window_size=x_list[0].size[0]
    # Statistical Features on raw x, y and z in time domain
    X_train = pd.DataFrame()

    # mean
    X_train['x_mean'] = pd.Series(x_list).apply(lambda x: x.mean())
    X_train['y_mean'] = pd.Series(y_list).apply(lambda x: x.mean())
    X_train['z_mean'] = pd.Series(z_list).apply(lambda x: x.mean())

    # std dev
    X_train['x_std'] = pd.Series(x_list).apply(lambda x: x.std())
    X_train['y_std'] = pd.Series(y_list).apply(lambda x: x.std())
    X_train['z_std'] = pd.Series(z_list).apply(lambda x: x.std())

    # avg absolute diff
    X_train['x_aad'] = pd.Series(x_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['y_aad'] = pd.Series(y_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['z_aad'] = pd.Series(z_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    # min
    X_train['x_min'] = pd.Series(x_list).apply(lambda x: x.min())
    X_train['y_min'] = pd.Series(y_list).apply(lambda x: x.min())
    X_train['z_min'] = pd.Series(z_list).apply(lambda x: x.min())

    # max
    X_train['x_max'] = pd.Series(x_list).apply(lambda x: x.max())
    X_train['y_max'] = pd.Series(y_list).apply(lambda x: x.max())
    X_train['z_max'] = pd.Series(z_list).apply(lambda x: x.max())

    # max-min diff
    X_train['x_maxmin_diff'] = X_train['x_max'] - X_train['x_min']
    X_train['y_maxmin_diff'] = X_train['y_max'] - X_train['y_min']
    X_train['z_maxmin_diff'] = X_train['z_max'] - X_train['z_min']

    # median
    X_train['x_median'] = pd.Series(x_list).apply(lambda x: np.median(x))
    X_train['y_median'] = pd.Series(y_list).apply(lambda x: np.median(x))
    X_train['z_median'] = pd.Series(z_list).apply(lambda x: np.median(x))

    # median abs dev 
    X_train['x_mad'] = pd.Series(x_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['y_mad'] = pd.Series(y_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['z_mad'] = pd.Series(z_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))

    # interquartile range
    X_train['x_IQR'] = pd.Series(x_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['y_IQR'] = pd.Series(y_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['z_IQR'] = pd.Series(z_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    # negtive count
    X_train['x_neg_count'] = pd.Series(x_list).apply(lambda x: np.sum(x < 0))
    X_train['y_neg_count'] = pd.Series(y_list).apply(lambda x: np.sum(x < 0))
    X_train['z_neg_count'] = pd.Series(z_list).apply(lambda x: np.sum(x < 0))

    # positive count
    X_train['x_pos_count'] = pd.Series(x_list).apply(lambda x: np.sum(x > 0))
    X_train['y_pos_count'] = pd.Series(y_list).apply(lambda x: np.sum(x > 0))
    X_train['z_pos_count'] = pd.Series(z_list).apply(lambda x: np.sum(x > 0))

    # values above mean
    X_train['x_above_mean'] = pd.Series(x_list).apply(lambda x: np.sum(x > x.mean()))
    X_train['y_above_mean'] = pd.Series(y_list).apply(lambda x: np.sum(x > x.mean()))
    X_train['z_above_mean'] = pd.Series(z_list).apply(lambda x: np.sum(x > x.mean()))

    # number of peaks
    X_train['x_peak_count'] = pd.Series(x_list).apply(lambda x: len(find_peaks(x)[0]))
    X_train['y_peak_count'] = pd.Series(y_list).apply(lambda x: len(find_peaks(x)[0]))
    X_train['z_peak_count'] = pd.Series(z_list).apply(lambda x: len(find_peaks(x)[0]))

    # skewness
    X_train['x_skewness'] = pd.Series(x_list).apply(lambda x: stats.skew(x))
    X_train['y_skewness'] = pd.Series(y_list).apply(lambda x: stats.skew(x))
    X_train['z_skewness'] = pd.Series(z_list).apply(lambda x: stats.skew(x))

    # kurtosis
    X_train['x_kurtosis'] = pd.Series(x_list).apply(lambda x: stats.kurtosis(x))
    X_train['y_kurtosis'] = pd.Series(y_list).apply(lambda x: stats.kurtosis(x))
    X_train['z_kurtosis'] = pd.Series(z_list).apply(lambda x: stats.kurtosis(x))

    # energy
    X_train['x_energy'] = pd.Series(x_list).apply(lambda x: np.sum(x**2)/window_size)
    X_train['y_energy'] = pd.Series(y_list).apply(lambda x: np.sum(x**2)/window_size)
    X_train['z_energy'] = pd.Series(z_list).apply(lambda x: np.sum(x**2/window_size))

    # avg resultant
    X_train['avg_result_accl'] = [i.mean() for i in ((pd.Series(x_list)**2 + pd.Series(y_list)**2 + pd.Series(z_list)**2)**0.5)]

    # signal magnitude area
    X_train['sma'] =    pd.Series(x_list).apply(lambda x: np.sum(abs(x)/100)) + pd.Series(y_list).apply(lambda x: np.sum(abs(x)/100)) \
                    + pd.Series(z_list).apply(lambda x: np.sum(abs(x)/100))

    # converting the signals from time domain to frequency domain using FFT
    x_list_fft = pd.Series(x_list).apply(lambda x: np.abs(np.fft.fft(x))[1:1+window_size])
    y_list_fft = pd.Series(y_list).apply(lambda x: np.abs(np.fft.fft(x))[1:1+window_size])
    z_list_fft = pd.Series(z_list).apply(lambda x: np.abs(np.fft.fft(x))[1:1+window_size])

    # Statistical Features on raw x, y and z in frequency domain
    # FFT mean
    X_train['x_mean_fft'] = pd.Series(x_list_fft).apply(lambda x: x.mean())
    X_train['y_mean_fft'] = pd.Series(y_list_fft).apply(lambda x: x.mean())
    X_train['z_mean_fft'] = pd.Series(z_list_fft).apply(lambda x: x.mean())

    # FFT std dev
    X_train['x_std_fft'] = pd.Series(x_list_fft).apply(lambda x: x.std())
    X_train['y_std_fft'] = pd.Series(y_list_fft).apply(lambda x: x.std())
    X_train['z_std_fft'] = pd.Series(z_list_fft).apply(lambda x: x.std())

    # FFT avg absolute diff
    X_train['x_aad_fft'] = pd.Series(x_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['y_aad_fft'] = pd.Series(y_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['z_aad_fft'] = pd.Series(z_list_fft).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    # FFT min
    X_train['x_min_fft'] = pd.Series(x_list_fft).apply(lambda x: x.min())
    X_train['y_min_fft'] = pd.Series(y_list_fft).apply(lambda x: x.min())
    X_train['z_min_fft'] = pd.Series(z_list_fft).apply(lambda x: x.min())

    # FFT max
    X_train['x_max_fft'] = pd.Series(x_list_fft).apply(lambda x: x.max())
    X_train['y_max_fft'] = pd.Series(y_list_fft).apply(lambda x: x.max())
    X_train['z_max_fft'] = pd.Series(z_list_fft).apply(lambda x: x.max())

    # FFT max-min diff
    X_train['x_maxmin_diff_fft'] = X_train['x_max_fft'] - X_train['x_min_fft']
    X_train['y_maxmin_diff_fft'] = X_train['y_max_fft'] - X_train['y_min_fft']
    X_train['z_maxmin_diff_fft'] = X_train['z_max_fft'] - X_train['z_min_fft']

    # FFT median
    X_train['x_median_fft'] = pd.Series(x_list_fft).apply(lambda x: np.median(x))
    X_train['y_median_fft'] = pd.Series(y_list_fft).apply(lambda x: np.median(x))
    X_train['z_median_fft'] = pd.Series(z_list_fft).apply(lambda x: np.median(x))

    # FFT median abs dev 
    X_train['x_mad_fft'] = pd.Series(x_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['y_mad_fft'] = pd.Series(y_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['z_mad_fft'] = pd.Series(z_list_fft).apply(lambda x: np.median(np.absolute(x - np.median(x))))

    # FFT Interquartile range
    X_train['x_IQR_fft'] = pd.Series(x_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['y_IQR_fft'] = pd.Series(y_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['z_IQR_fft'] = pd.Series(z_list_fft).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))

    # FFT values above mean
    X_train['x_above_mean_fft'] = pd.Series(x_list_fft).apply(lambda x: np.sum(x > x.mean()))
    X_train['y_above_mean_fft'] = pd.Series(y_list_fft).apply(lambda x: np.sum(x > x.mean()))
    X_train['z_above_mean_fft'] = pd.Series(z_list_fft).apply(lambda x: np.sum(x > x.mean()))

    # FFT number of peaks
    X_train['x_peak_count_fft'] = pd.Series(x_list_fft).apply(lambda x: len(find_peaks(x)[0]))
    X_train['y_peak_count_fft'] = pd.Series(y_list_fft).apply(lambda x: len(find_peaks(x)[0]))
    X_train['z_peak_count_fft'] = pd.Series(z_list_fft).apply(lambda x: len(find_peaks(x)[0]))

    # FFT skewness
    X_train['x_skewness_fft'] = pd.Series(x_list_fft).apply(lambda x: stats.skew(x))
    X_train['y_skewness_fft'] = pd.Series(y_list_fft).apply(lambda x: stats.skew(x))
    X_train['z_skewness_fft'] = pd.Series(z_list_fft).apply(lambda x: stats.skew(x))

    # FFT kurtosis
    X_train['x_kurtosis_fft'] = pd.Series(x_list_fft).apply(lambda x: stats.kurtosis(x))
    X_train['y_kurtosis_fft'] = pd.Series(y_list_fft).apply(lambda x: stats.kurtosis(x))
    X_train['z_kurtosis_fft'] = pd.Series(z_list_fft).apply(lambda x: stats.kurtosis(x))

    # FFT energy
    X_train['x_energy_fft'] = pd.Series(x_list_fft).apply(lambda x: np.sum(x**2)/(window_size/2))
    X_train['y_energy_fft'] = pd.Series(y_list_fft).apply(lambda x: np.sum(x**2)/(window_size/2))
    X_train['z_energy_fft'] = pd.Series(z_list_fft).apply(lambda x: np.sum(x**2)/(window_size/2))

    # FFT avg resultant
    X_train['avg_result_accl_fft'] = [i.mean() for i in ((pd.Series(x_list_fft)**2 + pd.Series(y_list_fft)**2 + pd.Series(z_list_fft)**2)**0.5)]

    # FFT Signal magnitude area
    X_train['sma_fft'] = pd.Series(x_list_fft).apply(lambda x: np.sum(abs(x)/50)) + pd.Series(y_list_fft).apply(lambda x: np.sum(abs(x)/50)) \
                        + pd.Series(z_list_fft).apply(lambda x: np.sum(abs(x)/50))

    # Max Indices and Min indices 

    # index of max value in time domain
    X_train['x_argmax'] = pd.Series(x_list).apply(lambda x: np.argmax(x))
    X_train['y_argmax'] = pd.Series(y_list).apply(lambda x: np.argmax(x))
    X_train['z_argmax'] = pd.Series(z_list).apply(lambda x: np.argmax(x))

    # index of min value in time domain
    X_train['x_argmin'] = pd.Series(x_list).apply(lambda x: np.argmin(x))
    X_train['y_argmin'] = pd.Series(y_list).apply(lambda x: np.argmin(x))
    X_train['z_argmin'] = pd.Series(z_list).apply(lambda x: np.argmin(x))

    # absolute difference between above indices
    X_train['x_arg_diff'] = abs(X_train['x_argmax'] - X_train['x_argmin'])
    X_train['y_arg_diff'] = abs(X_train['y_argmax'] - X_train['y_argmin'])
    X_train['z_arg_diff'] = abs(X_train['z_argmax'] - X_train['z_argmin'])

    # index of max value in frequency domain
    X_train['x_argmax_fft'] = pd.Series(x_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:51]))
    X_train['y_argmax_fft'] = pd.Series(y_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:51]))
    X_train['z_argmax_fft'] = pd.Series(z_list_fft).apply(lambda x: np.argmax(np.abs(np.fft.fft(x))[1:51]))

    # index of min value in frequency domain
    X_train['x_argmin_fft'] = pd.Series(x_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:51]))
    X_train['y_argmin_fft'] = pd.Series(y_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:51]))
    X_train['z_argmin_fft'] = pd.Series(z_list_fft).apply(lambda x: np.argmin(np.abs(np.fft.fft(x))[1:51]))

    # absolute difference between above indices
    X_train['x_arg_diff_fft'] = abs(X_train['x_argmax_fft'] - X_train['x_argmin_fft'])
    X_train['y_arg_diff_fft'] = abs(X_train['y_argmax_fft'] - X_train['y_argmin_fft'])
    X_train['z_arg_diff_fft'] = abs(X_train['z_argmax_fft'] - X_train['z_argmin_fft'])

    return X_train