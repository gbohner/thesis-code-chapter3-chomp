# Segmentation of neural cell bodies from calcium imaging

Code for Gergo Bohner's thesis work, Chapter 3. Full text: https://github.com/gbohner/thesis-pdf-git

--

### Technical notes

All code for this chapter was written in Matlab (R) and ran on version R2016b on Ubuntu 16.04 LTS.


### Data input

The data used in the chapter for exclusively from the Neurofinder challenge, and therefore in a well-defined format (see http://neurofinder.codeneuro.org/ and https://github.com/codeneuro/neurofinder-datasets for details).

Furthermore I use as input a preprocessed -- "corrected" -- version of the Neurofinder datasets, that are created via my method outlined in Chapter 2 of the thesis. For technical details, see the code repository for chapter 2 (https://github.com/gbohner/thesis-code-chapter2-preprocessing), and point #5 of its README.


### Running the pipeline

The inner workings of the CHOMP algorithm are described in the original repository (https://github.com/gbohner/CHOMP/tree/neurofinder), the whole segmentation pipeline can be run by creating a ```chomp_options``` object (see ```Classes/chomp_options.m``` for parameters) and calling the ```chomp()``` function on it.

To run the segmentation on the original datasets, see the ```run_dataset_*.m``` scripts.

To run segmentation on datasets preprocessed by the algorithm described in Chapter 2, see the ```run_all_neurofinder_preproc_*.m``` scripts.

To create the json file that can be submitted to the Neurofinder challenge interface, run ```final_submission_file.m``` with the appropriate timestamps.

The various example visualisation are created by the scripts in the ```Figures/``` folder. In particular, the detailed example of the presence of higher order cumulants in dataset 01.00 and their reconstruction by CHOMP is carried out by ```Figures/final_figure_1_cumulants_present.m```, and the generated images make up figures 3.5-3.9 in the thesis. The detailed evaluation of CHOMP segmentation on dataset 00.00 -- shown in figures 3.10-3.13 in thesis -- is implemented in ```Figures/figure_1_data00_results.m```, with the neural time series shown extracted by ```Figures/example_neurofinder_timeseries.m```. The supplemental videos referenced in figure 3.13 are created by ```Figures/createAllExampleVideos.m```.  The performance metric curves in figure 3.11 are plotted by ```Figures/plot_all_training_result_curves.m```. 


### Validation code

The validation (figure 3.4) is a reproduction from our earlier published work (https://ieeexplore.ieee.org/document/7738847, Fig. 4), with the simulation and plotting code available in the ```CHOMP_simulator/``` folder. The simulation can be carried out via ```figure_5_run.m```, and the 4 subfigures in the thesis are plotted by the ```figure_X_plot.m``` scripts, with X = {5,9,10,11}.

 

