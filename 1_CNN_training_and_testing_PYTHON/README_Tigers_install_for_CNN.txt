Installation of UNet:

1) Open up Anaconda command line
2) pip install matplotlib scipy scikit-image pillow numpy natsort opencv-python keras pandas bcolz skan sklearn numba tifffile

If any of the above packages fail to install, try:

conda install package_name_here

Also, "bcolz", "sklearn", "skan", and "numba" are not that important if they don't install


To perform inference on new data:

1) Open the 1_UNet_inference_Bergles_cell_tracking.py file into Spyder (IDE within Anaconda)
2) Run the script and you will be prompted to select an input folder
3) Then, you can type "y" or "n" followed by "Enter" if you wish to choose more folders of data to analyze
4) The analysis should then start and the outputs will be placed in a new folder within the selected folders ending in "_analytic_results"


That's it! Then you can open up MATLAB and use the manual correction script to loop through the data and correct as needed.