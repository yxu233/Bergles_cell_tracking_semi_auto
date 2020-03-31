# Bergles_cell_tracking_semi_auto

### Installation: (ensure > MATLAB 2019a, or else sliceViewer will not work)
* 1. Install github desktop: https://desktop.github.com/
* 2. Clone the git repo so it shows up in github desktop
   
   
### Analysis pipeline:
* 1. Run Ilastik on raw data to get cell body segmentations
* 2. Run Bergles_watershed_and_save.m on the Ilastik data to separate adjacent cell bodies
* 3. Place both the RAW data (fluorescence) and watershed identified cell bodies into a folder so the files are interleaved (time_1_raw, time_1_cell_bodies, time_2_raw, time_2_cell_bodies, ect...)
* 4. Run Bergles_cell_tracking_semi_auto.m to start analysis


### Manual analysis hot-keys: - if want new hotkeys, modify "Bergles_manual_count.m"
* "1" - classify as same tracked cell
* "2" - classify as no longer tracked
* "a" - add
* "s" - scale
* "d" - delete cell (from current frame) permanently - i.e. if cell body is junk
* "c" - CLAHE

### GUI inputs:
     * Size... ==> leave empty


### Demo run:
* 1. Download data here (5 GB), includes output data from my corrections: https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/yxu130_jh_edu/EpeTEaEYmB5FvK4ESh-uv7oB3cjEyifYDWRSDdLitczvow?e=VHYxb2 
   
