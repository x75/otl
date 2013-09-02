%Matlab OTL setup script

%adds the path to the MATLAB system
disp 'Setting up Matlab OTL path'
curr_file = mfilename('fullpath');
mootell_dir = fileparts(curr_file);
addpath(genpath(mootell_dir));
%rmpath(strcat(mootell_dir, filesep, 'sandbox'));
disp 'Done! See examples folder for sample code'

