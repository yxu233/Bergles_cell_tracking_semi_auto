# Bergles_cell_tracking_semi_auto

### Installation: (ensure > MATLAB 2019a, or else sliceViewer will not work)
* 1. Install github desktop: https://desktop.github.com/
* 2. Clone the git repo so it shows up in github desktop
   
   
### Analysis pipeline:
* 1. Run Ilastik on raw data to get cell body segmentations
* 2. Run Bergles_watershed_and_save.m on the Ilastik data to separate adjacent cell bodies
* 3. Place both the RAW data (fluorescence) and watershed identified cell bodies into a folder so the files are interleaved (time_1_raw, time_1_cell_bodies, time_2_raw, time_2_cell_bodies, ect...)
* 4. Run Bergles_cell_tracking_semi_auto.m to start analysis


### Manual analysis hot-keys:
* "1"
* "2"
* "a" - add
* "s" - scale
* "d" - delete
* "c" - CLAHE

### GUI inputs:
     * Size... ==> leave empty


### Demo run:
* 1. 
   
