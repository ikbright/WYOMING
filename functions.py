import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import cv2
import lasio
from datetime import datetime
from skimage.filters import threshold_otsu

##### Functions #####

# explicit function to normalize array
def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

def create_gaborfilter(number, kernel_size, sigma, lambd, gamma, psi):
    # This function is designed to produce a set of GaborFilters 
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = number
    ksize = kernel_size  # The local area to evaluate
    sigma = sigma  # Larger Values produce more edges
    lambd = lambd
    gamma = gamma
    psi = psi  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

def apply_filter(img, filters):
# This general function is designed to apply filters to our image
     
    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)
     
    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1 # remain depth same as original image
     
    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv2.filter2D(img, depth, kern)  #Apply filter to image
         
        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage

def KMeans(mat2, k_value):
    # convert to np.float32
    Z = np.float32(mat2)
    # define criteria, number of clusters(K) and apply kmeans()
    k_value = k_value
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,k_value,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((mat2.shape))
    
    return res2

#TODO: Include option here
@st.cache_data
def load_depthlog(option):

    if option == 'Upper':
        string = 'upper'
    
    if option == 'Lower':
        string = 'lower'
    
    depthlog = pd.read_csv(f'{string}/depthlog.txt', header=None)
    
    return depthlog

def load_depths(bot, top, option):
    depthlog = load_depthlog(option)
    depth_values = depthlog[0].values
    depth_bot = (np.abs(depth_values - bot)).argmin()
    depth_top = (np.abs(depth_values - top)).argmin()
    
    return [depth_bot, depth_top]

    
    


#TODO: Include option here
@st.cache_data
def load_data(option):
    if option == 'Upper':
        string = 'upper'
    if option == 'Lower':
        string = 'lower'

    files_list = [f'{string}/data1.txt', f'{string}/data2.txt', f'{string}/data3.txt', f'{string}/data4.txt']
    df_list = []

    for file in files_list:
        df = pd.read_csv(
            file,
            header=None,
            delim_whitespace=True,  # Replaces sep='\s+'
            engine='python',
            on_bad_lines='skip'     # Skips malformed lines like line 28
        )
        df_list.append(df)

    big_df = pd.concat(df_list, ignore_index=True)
    return big_df


def data_gen(bot, top, option, interp_interval=0.01):
    
    donnee = load_data(option)
    depthlog = load_depthlog(option)
    depth_range = load_depths(bot, top, option)
    depth_bot = depth_range[0]
    depth_top = depth_range[1]
    donnee = donnee.iloc[depth_bot:depth_top+1,:]




    #if amplitudes are considered
    max_donnee = donnee.max().max()
    donnee = max_donnee*np.ones((donnee.shape)) - donnee

    ## Interpolation
    interp_interval = interp_interval

    # interpolates the depth values to a constant interval
    # depth = np.arange(depthlog.iloc[depth_bot,0], depthlog.iloc[depth_top, 0], interp_interval)
    depth = np.arange(depthlog.iloc[depth_bot,0], depthlog.iloc[depth_top, 0], interp_interval)
    depthlog_new = depthlog.iloc[depth_bot:depth_top+1,:]
    depthlog_flat = depthlog_new.to_numpy()
    depthlog_flat = depthlog_flat.flatten()

    # defines a new matrix with only the data we are interested in

    mat_list = []

    for col in np.arange(0, len(donnee.columns)):
        # values = donnee.iloc[:, col]
        # print(f"Length of depthlog_flat: {len(depthlog_flat)}")
        # print(f"Length of values: {len(values)}")
        # interpolated = interp1d(depthlog_flat, values)(depth)
        # mat_list.append(interpolated)
        
        values = donnee.iloc[:, col].values  # Convert column to NumPy array

        # Ensure both arrays are of the same length by trimming the longer array
        min_length = min(len(depthlog_flat), len(values))
        depthlog_flat_trimmed = depthlog_flat[:min_length]
        values_trimmed = values[:min_length]

        # Filter out NaN values in both arrays
        valid_indices = ~np.isnan(depthlog_flat_trimmed) & ~np.isnan(values_trimmed)
        depthlog_clean = depthlog_flat_trimmed[valid_indices]
        values_clean = values_trimmed[valid_indices]


        # Ensure both arrays are of the same length after filtering
        if len(depthlog_clean) != len(values_clean):
            continue  # Skip this column if lengths don't match

        # Interpolation (optional, if needed)
        try:
            interpolated = interp1d(depthlog_clean, values_clean, fill_value="extrapolate")(depth)
            mat_list.append(interpolated)
        except Exception as e:
            pass  # If there's an error, just skip the interpolation for this column
        
       
    
    
    mat = pd.DataFrame(mat_list)
    mat = mat.transpose()
    mat2 = mat.to_numpy()
    mat2 = mat2.astype(np.uint8)
    
    return mat2

@st.cache_data
def img_analysis(well_section, eq_check, filter_option, thresh_option):

    if well_section == 'Lower':

        bot = 9000.01580609481
        top = 9818.99080609481 #7198
        depths = load_depths(bot, top, well_section)
        mat2 = data_gen(bot, top, well_section)

    if well_section == 'Upper':

        bot = 8000.01580609481
        top = 9818.99080609481
        depths = load_depths(bot, top, well_section)
        mat2 = data_gen(bot, top, well_section)

    if eq_check == True:
        mat2 = cv2.equalizeHist(mat2)

    ##### Kernel selection #####

    if filter_option == 'Averaging Kernel':
        kernel_avg = np.ones((5,5),np.float32)/25
        output = cv2.filter2D(mat2,-1,kernel_avg)
    elif filter_option == 'Gaussian Blur':
        output = cv2.blur(mat2,(7,7))
    elif filter_option == 'Bilateral Filter':
        output = cv2.bilateralFilter(mat2,9,75,75)
    elif filter_option == 'Median Filter':
        output = cv2.medianBlur(mat2,5)

    ##### Gabor filters #####

    num_filters = 16
    ksize = 5
    sigma = 5.0
    lambd = 10.0
    gamma = 0.5
    psi = 0

    gfilters = create_gaborfilter(num_filters, ksize, sigma, lambd, gamma, psi)
    image_g = apply_filter(output, gfilters)

    ##### Thresholding #####

    if thresh_option == 'Otsu':
        globalthreshold = threshold_otsu(image_g)
        binary_img = image_g > globalthreshold
        binary_img = (binary_img * 255).astype(np.uint8)
    elif thresh_option == 'Adaptive Mean':
        binary_img = cv2.adaptiveThreshold(image_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 199, 5)
    elif thresh_option == 'Adaptive Gaussian':
        binary_img = cv2.adaptiveThreshold(image_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 199, 5)

    ##### Canny Edge Detection #####

    # Defining all the parameters
    threshold_values = [700,900]
    aperture_size = 5 # Aperture size
    L2Gradient = False # Boolean
    
    # Applying the Canny Edge filter with Aperture Size and L2Gradient

    if L2Gradient == 'True':
        edge = cv2.Canny(binary_img, threshold_values[0], threshold_values[1],
                        apertureSize = aperture_size, 
                        L2gradient = True )
    else:
        edge = cv2.Canny(binary_img, threshold_values[0], threshold_values[1],
                    apertureSize = aperture_size)

    edge_copy = edge.copy()

    ##### Threshold with Original #####

    contours, hierarchy = cv2.findContours(edge_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    data_points = []
    y_locations = []

    for i in range(len(contours)):
        if cv2.arcLength(contours[i],False) > 100:
            interm = contours[i]
            y = 0
            for j in range(len(interm)):
                y += interm[j][0][1]
            average_y = round(y/len(interm))
            y_locations.append(average_y)
            data_points.append(interm)

    df = pd.DataFrame(0, index=np.around(np.linspace(bot, top, len(mat2)), 2), columns=['Number of fractures'])
    for i in y_locations:
        df.iloc[i, 0] = 1

    df = df.sort_index(axis = 0)

    return df

def exportLAS(df, resolution):
    las = lasio.LASFile()
    las.well.DATE = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    las.well.COMPANY = 'WVU'
    las.well.WELL = 'CCS1'
    las.well.API = '1211523415'
    depths = df['DEPT'].round(2)
    curve = df['Number of fractures']
    las.append_curve('DEPT', depths, unit='ft')
    las.append_curve('NF', curve, descr='Number of baffles/fractures')
    las.update_start_stop_step(fmt='%.2f')
    return las.write(f'nf_density_{resolution}ft.las', version=2)