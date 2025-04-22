# pylint: disable=singleton-comparison
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import ndimage
import cv2
import plotly.express as px
from skimage.filters import threshold_otsu
import matplotlib.ticker as ticker
import lasio
from datetime import datetime
from functions import *
import os
import base64

##### Page Setup #####

st.set_page_config(page_title="WYOMING Dashboard")
st.title('Computer Vision Workflow')
st.write('''This dashboard is intended for easier observations when changing different values of the algorithms contained in this workflow. 
         The workflow now supports the full range of useable data from the Wyoming well. The upper section ranges from **8000 ft** to **9000 ft** while the lower section ranges from **9050 ft** to **9681 ft**.''')
st.header('Original Image')
st.sidebar.title('Parameters For Adjustment')
st.sidebar.header('Specify Well Section')

##### Load data and set range visualization #####

st.markdown("""
                <html>
                    <head>
                    <style>
                        ::-webkit-scrollbar {
                            width: 10px;
                            }

                            /* Track */
                            ::-webkit-scrollbar-track {
                            background: #f1f1f1;
                            }

                            /* Handle */
                            ::-webkit-scrollbar-thumb {
                            background: #888;
                            }

                            /* Handle on hover */
                            ::-webkit-scrollbar-thumb:hover {
                            background: #555;
                            }
                    </style>
                    </head>
                    <body>
                    </body>
                </html>
            """, unsafe_allow_html=True)

st.sidebar.header('Depth Interval')

well_section = st.sidebar.radio('Which well section do you want to analyze?', ('Upper', 'Lower'), index=0)

if well_section == 'Upper':
    bot = st.sidebar.number_input('State top depth in ft. The interval is 2ft for better resolution', value=8490, max_value = 9000, min_value = 8000)

if well_section == 'Lower':
    bot = st.sidebar.number_input('State top depth in ft. The interval is 2ft for better resolution', value=9120, max_value = 9681, min_value = 9050)

image_range = st.sidebar.radio('Choose the range of the image for analysis', ('1 ft', '2 ft'), index=1)
if image_range == '1 ft':
    top = bot + 1
elif image_range == '2 ft':
    top = bot + 2
depths = load_depths(bot, top, well_section)
mat2 = data_gen(bot, top, well_section)
ori_img = mat2.copy()

st.sidebar.header('Image manipulation')
color_range = st.sidebar.slider(
    "Pixel range:",
    value=(float(mat2.min().min()),float(mat2.max().max())))

fig_original = px.imshow(mat2, color_continuous_scale = 'YlOrBr_r', range_color=(color_range[0], color_range[1]))
fig_original.update_xaxes(visible=False)
st.plotly_chart(fig_original)

st.sidebar.header('Pixel Histogram Equalization')
eq_check = st.sidebar.checkbox('Check the box if you want to perform pixel normalization', value = False)

if eq_check == True:
    mat2 = cv2.equalizeHist(mat2)
    fig_eq = px.imshow(mat2, color_continuous_scale = 'YlOrBr_r', range_color=(color_range[0], color_range[1]))
    fig_eq.update_xaxes(visible=False)

    st.plotly_chart(fig_eq)
##### Kernel selection #####

st.header('Average / Gaussian / Bilateral')
st.write('Choose between the different filtering options on the sidebar or experiment to see which fits the depth interval better.')

st.sidebar.header('Kernel Selection')
filter_option = st.sidebar.radio('Filter Options', ('Averaging Kernel', 'Median Filter', 'Gaussian Blur', 'Bilateral Filter'), index=3) # Boolean

if filter_option == 'Averaging Kernel':
    kernel_avg = np.ones((5,5),np.float32)/25
    output = cv2.filter2D(mat2,-1,kernel_avg)
elif filter_option == 'Gaussian Blur':
    output = cv2.blur(mat2,(7,7))
elif filter_option == 'Bilateral Filter':
    output = cv2.bilateralFilter(mat2,9,75,75)
elif filter_option == 'Median Filter':
    output = cv2.medianBlur(mat2,5)

fig_filter = px.imshow(output,color_continuous_scale = 'YlOrBr_r', range_color=(color_range[0], color_range[1]))
st.plotly_chart(fig_filter)

##### Gabor filters #####
    
st.header('Gabor Filters')
st.write('This step filters the lines in the image based on a range of angles specified by the parameters in the sidebar.')
st.sidebar.header('Gabor Filter Args')

num_filters = st.sidebar.slider('Number of filters', 3, 50, 16)
ksize = st.sidebar.slider('Kernel size', 3, 9, 5)  # The local area to evaluate
sigma = st.sidebar.slider('Sigma value', 0.5, 10.0, 5.0)  # Larger Values produce more edges
lambd = st.sidebar.slider('Lambda value', 0.5, 50.0, 10.0)
gamma = st.sidebar.slider('Gamma value', 0.0, 5.0, 0.5)
psi = st.sidebar.slider('Psi value', 0, 5, 0)  # Offset value - lower generates cleaner results

gfilters = create_gaborfilter(num_filters, ksize, sigma, lambd, gamma, psi)
image_g = apply_filter(output, gfilters)

fig_gabor = px.imshow(image_g, color_continuous_scale = 'gray')
st.plotly_chart(fig_gabor)

##### KMeans #####
    
# st.header('Kmeans Computation')
# st.write('If K-Means is used in this step, the resultant image will be used in the rest of the workflow. Otherwise, this step will be skipped.')
# st.sidebar.header('Kmeans Args')
# kmeans_check = st.sidebar.checkbox('Would you like to run KMeans?', value = True)
# if kmeans_check == True:
#     k_value = st.sidebar.number_input('What K-value would you want to try?', value = 10, min_value = 2)
#     res2 = KMeans(image_g, k_value)

#     fig_kmeans = px.imshow(res2, color_continuous_scale = 'gray')
#     fig_kmeans.update_xaxes(visible=False)
#     st.plotly_chart(fig_kmeans)

##### Thresholding #####

st.header('Thresholding Options')
st.write('Thresholding is done to highlight the edges better.')
st.sidebar.header('Thresholding')
thresh_option = st.sidebar.radio('Thresholding Options', ('Otsu', 'Adaptive Mean', 'Adaptive Gaussian'), index=0) # Boolean

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

fig_thresh = px.imshow(binary_img, color_continuous_scale = 'gray')
st.plotly_chart(fig_thresh)

##### Canny Edge Detection #####

st.header('Canny Edge Detection')
st.write('This step detects edges contained in the resultant image and connects qualifying edges based on the threshold values in the sidebar. An optional L2 gradient can also be used.')
st.sidebar.header('Canny Edge Args')

# Defining all the parameters
threshold_values = st.sidebar.slider('Threshold values', 500, 1500, (700, 900))
aperture_size = 5 # Aperture size
L2Gradient = st.sidebar.radio('Enable L2 Gradient?', ('True', 'False'), index=1) # Boolean
  
# Applying the Canny Edge filter with Aperture Size and L2Gradient

if L2Gradient == 'True':
    edge = cv2.Canny(binary_img, threshold_values[0], threshold_values[1],
                    apertureSize = aperture_size, 
                    L2gradient = True )
else:
    edge = cv2.Canny(binary_img, threshold_values[0], threshold_values[1],
                apertureSize = aperture_size)

edge_copy = edge.copy()
    
fig_canny = px.imshow(edge, color_continuous_scale = 'gray')
st.plotly_chart(fig_canny)

##### Threshold with Original #####

st.header('Results')

# dummy = np.ones(shape=mat2.shape, dtype=np.uint8)
mask_thresh = np.ma.masked_where(edge == 0, edge)

y_range = np.around(np.linspace(bot, top, len(mat2)), 1)
x_range = np.linspace(0,96)

# fig_res, ax_res = plt.subplots()
# ax_res.imshow(ori_img, cmap='YlOrBr_r', vmin=color_range[0], vmax=color_range[1])
# ax_res.imshow(mask_thresh, cmap = 'gray', alpha=0.7)
# plt.yticks(range(len(mat2)), y_range)
# ax_res.xaxis.set_major_locator(ticker.NullLocator())
# every_nth = 49
# for n, label in enumerate(ax_res.yaxis.get_ticklabels()):
#     if n % every_nth != 0:
#         label.set_visible(False)

# plt.tick_params(axis='y', length = 0)
# st.pyplot(fig_res)

contours, hierarchy = cv2.findContours(edge_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
final_img = mat2.copy()
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

GREEN = (0, 255, 0)
img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(final_img, data_points, -1, GREEN, 1)

fig_cnt, ax_cnt = plt.subplots()
ax_cnt.imshow(final_img, cmap='YlOrBr_r')
plt.yticks(range(len(mat2)), y_range)
plt.yticks(fontsize=8)
ax_cnt.xaxis.set_major_locator(ticker.NullLocator())
every_nth = 49
for n, label in enumerate(ax_cnt.yaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)

plt.tick_params(axis='y', length = 0)
st.pyplot(fig_cnt)

st.header('Log per 0.01 ft of image')
st.write('The density log only accounts for the major edges detected.')
st.dataframe(df.reset_index().style.highlight_max(axis=0))



##### Hough Transform #####

# st.header('Hough Transform')
# st.write('This step will check for continuous lines and optionally remove any lines that exceed an angle threshold.')
# st.sidebar.header('Hough Transform Args')

# hough_rho = st.sidebar.slider('Rho', 1, 5, 1)
# hough_thresh = st.sidebar.slider('Threshold', 1, 100, 10)
# minLineLength = st.sidebar.slider('Minimum Line Length', 1, 50, 10)  # Larger Values produce more edges
# maxLineGap = st.sidebar.slider('Maximum Line Gap', 1, 50, 5)
# angle_check = st.sidebar.checkbox('Keep Only Horizontal Lines?', value = True)
# if angle_check == True:
#     angle_thresh = st.sidebar.slider('Angle Threshold', 10, 90, 10)

# lines = cv2.HoughLinesP(edge, hough_rho, np.pi/180, threshold=hough_thresh, minLineLength=minLineLength, maxLineGap=maxLineGap)
# dummy = np.ones(shape=mat2.shape, dtype=np.uint8)
# horizontals = 0

# if lines is not None:
#     for line in range(0, len(lines)):
#         l = lines[line][0]
#         pt1 = (l[0], l[1])
#         pt2 = (l[2], l[3])
#         # angle = np.arctan2(l[3] - l[1], l[2] - l[0]) * 180. / np.pi
#         #TODO: should this check be adjustable with thresh_angle?
#         if angle_check == True and (np.abs(l[3] - l[1]) < angle_thresh):
#             cv2.line(dummy, pt1, pt2, (0,0,255), 1)
#             horizontals += 1
#         elif angle_check == False:
#             cv2.line(dummy, pt1, pt2, (0,0,255), 1)
#     if horizontals > 0:
#         statement = f'{horizontals} horizontal lines are detected in this image'
#         st.success(statement)
#     elif horizontals == 0:
#         statement = f'No horizontal lines are detected in this image'
#         st.error(statement)
# elif lines == None:
#     statement = 'There are no lines in this image'
#     st.error(statement)
#     pass

# masked = np.ma.masked_where(dummy == 1, dummy)

# fig_hough, ax_hough = plt.subplots()
# ax_hough.imshow(ori_img, cmap='YlOrBr_r', vmin=color_range[0], vmax=color_range[1])
# ax_hough.imshow(masked, cmap = 'gray', alpha=0.7)
# plt.axis('off')
# st.pyplot(fig_hough)

##### Option for downloading all images in a subplot #####

st.header('Final Results')

axis_range = []

for i in np.arange(bot, top, 0.01):
    axis_range.append(i)

ticklabels = ["{:6.2f}".format(i) for i in axis_range]

fig_combo, ax_combo = plt.subplots(1,5, figsize=(15,15))

ax_combo[0].imshow(ori_img, cmap='YlOrBr_r')
ax_combo[0].set_title('Original')
ax_combo[0].axis('off')
ax_combo[1].imshow(output, cmap='gray')
ax_combo[1].set_title('Bilateral Filter')
ax_combo[1].axis('off')
ax_combo[2].imshow(image_g, cmap='gray')
ax_combo[2].set_title('Gabor Filter')
ax_combo[2].axis('off')
ax_combo[3].imshow(binary_img, cmap='gray')
ax_combo[3].set_title('Thresholding')
ax_combo[3].axis('off')
ax_combo[4].imshow(edge, cmap='gray')
ax_combo[4].set_title('Canny Edge Detection')
ax_combo[4].axis('off')
st.pyplot(fig_combo)

##### Option for downloading all images in a subplot #####

st.header('Experimental Features')

resolution = st.selectbox(
    'Resolution?',
    (0.5, 1, 2, 5, 10)
)


well_section2 = st.selectbox(
    'Which section of the well?',
    ('Upper', 'Lower'),
    index = 1
)

eq_check2 = st.selectbox(
    'Histogram Equalization?',
    (True, False),
    index = 1
)

filter_option2 = st.selectbox(
    'Filter Options?',
    ('Averaging Kernel', 'Gaussian Blur', 'Bilateral Filter', 'Median Filter'),
    index = 2
)

thresh_option2 = st.selectbox(
    'Thresholding Options?',
    ('Otsu', 'Adaptive Mean', 'Adaptive Gaussian'), 
    index = 0
)

final_data = img_analysis(well_section2, eq_check2, filter_option2, thresh_option2)
interp_interval = 0.01
calc = (1/interp_interval) * resolution
df2 = final_data.reset_index()
df2 = df2.rename(columns = {'index': 'DEPT'})
d = {'DEPT': 'last', 'Number of fractures': 'sum'}
# res = df2.groupby(df2.index // calc).agg(d).round(decimals=1).astype({'DEPT': 'float'})
res = df2.groupby(df2.index // calc).agg(d).round({'DEPT': 1, 'Number of fractures': 0})
st.dataframe(res.style.highlight_max(axis=0))

res = res[res['DEPT'] <= 9681]
fig_nf, ax_nf = plt.subplots(figsize=(5,10))
ax_nf.plot('Number of fractures', 'DEPT', data=res, color='red', lw=0.5)
ax_nf.set_xlim(0, 40)
ax_nf.set_ylim(ax_nf.get_ylim()[::-1])
ax_nf.set_title('Number of baffles/barriers')
st.pyplot(fig_nf)

las_file = exportLAS(res, resolution)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.markdown(get_binary_file_downloader_html(f'nf_density_{resolution}ft.las', 'Natural Fracture Density Log'), unsafe_allow_html=True)
