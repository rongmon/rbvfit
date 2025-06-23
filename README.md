rbvfit
======


This suite of code will do a forward modeling analysis of absorption line spectrum. A Bayesian Voigt profile fitter.

Also includes a sophisticated Voigt profile generation toolbox, and an interactive Voigt profile fitting module.

Installation [using git]:

    From the command line: 
        git clone https://github.com/rongmon/rbvfit.git
        
        cd rbvfit
        
        python setup.py install

Description
===========

Main Modules:
    model.py:-
    		 Top level code allowing creation of a complex and flexible multi-component/ Multi-species Voigt Profile.
    rb_vfit.py:- 
    		 General code to create individual Voigt profiles.
    rb_setline.py:-
    		 Allows to read in line properties of an atomic transition using an approximate rest wavelength guess.

    rb_interactive_vpfit.py:- 
    		 A complex interactive Voigt profile fitter...
    		 	Two options for fitting: -  
    		 				(i)  Fast non-linear least squares to fit a model to data.
    		 				(ii) Detailed Markov Chain Monte Carlo fitting using emcee. 

    example_Voigt_profile_fitting.ipynb:- 

             Example jupyter notebook on how to use these codes to perform a fit [batch mode]

    example_Voigt_profile_fitting_interactive.ipynb:- 

             Example jupyter notebook on how to use these codes to perform a fit [interactive mode]. We guess the number of absorbing clouds and their b, v, n values interactively inside a jupyter notebook.


DOI
====
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10403231.svg)](https://doi.org/10.5281/zenodo.10403231)



Note
====
Written By: Rongmon Bordoloi.  July 2019.
Tested on : Python 3.7

Dependencies on linetools, emcee, corner


