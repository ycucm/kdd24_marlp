import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

# def peak_time_error(pred, true):
#     # find the peak time of pred and true
#     peak_t_pred = np.argmin(pred)
#     peak_t_true = np.argmin(true)
#     return np.abs(peak_t_pred - peak_t_true)

# def peak_value_error(pred, true):
#     # find the peak value of pred and true
#     peak_v_pred = np.min(pred)
#     peak_v_true = np.min(true)
#     return np.abs(peak_v_pred - peak_v_true)

# Calculate the peak time error for each row in 2D arrays and return the average.
def peak_time_error(pred, true):
    # Initialize an array to store the time errors for each row
    time_errors = np.zeros(pred.shape[0])
    # Iterate over each row (assuming rows are the first dimension)
    for i in range(pred.shape[0]):
        # Find the index of the minimum value in the current row for both arrays
        peak_t_pred = np.argmin(pred[i])
        peak_t_true = np.argmin(true[i])
        print('peak_t_pred: ', peak_t_pred, 'peak_t_true: ', peak_t_true)
        # Calculate the absolute difference in indices and store it
        time_errors[i] = np.abs((peak_t_pred - peak_t_true) / 6)
    # Return the average time error
    return np.mean(time_errors)

# Calculate the peak value error for each row in 2D arrays and return the average.
def peak_value_error(pred, true):
    # Initialize an array to store the value errors for each row
    value_errors = np.zeros(pred.shape[0])
    # Iterate over each row
    for i in range(pred.shape[0]):
        # Find the minimum value in the current row for both arrays
        peak_v_pred = np.min(pred[i])
        peak_v_true = np.min(true[i])
        # print('peak_v_pred: ', peak_v_pred, 'peak_v_true: ', peak_v_true)
        # Calculate the absolute difference in values and store it
        value_errors[i] = np.abs(peak_v_pred - peak_v_true)
    # Return the average value error
    return np.mean(value_errors)

def metric(pred, true):
    print("Shape:", pred.shape, true.shape)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    peak_t_error = peak_time_error(pred, true)
    peak_v_error = peak_value_error(pred, true)

    return mae, mse, rmse, mape, mspe, peak_t_error, peak_v_error
