from astropy.io import fits
import numpy as np
import os
#import subprocess

dir_path = '/home/yymx2/HDD/data6/GFA/image2/drive-download-20240206T063357Z-001'
data_file_path = os.path.join(dir_path, 'filelist.dat')
data = np.genfromtxt(data_file_path, dtype=str)


for ii in range(len(data)):
    flname=data[ii]
    print(flname)

    data_in_path = os.path.join(dir_path, flname)
    # Read the FITS file    
    hdu_list = fits.open(data_in_path)
    ori = hdu_list[0].data  # Assuming the image data is in the primary HDU
    primary_header = hdu_list[0].header
    ra_in = primary_header['RA']
    dec_in = primary_header['DEC']

    # Get the sky values
    sky1 = ori[1000, 255]
    sky2 = ori[1000, 767]

    # Subtract sky values
    ori[:, 0:512] = ori[:, 0:512] - sky1  # Python indexing is inclusive at the start and exclusive at the end
    ori[:, 512:] = ori[:, 512:] - sky2

    # Crop the image
    orif = ori[12:979, 1:1023]  # Adjusted indices for Python's 0-based indexing

    # Write the processed data to a new FITS file
    dir_out = '/home/yymx2/HDD/data6/GFA/image2/procimg'
    newname='proc'+flname
    data_file_path = os.path.join(dir_out, newname)    
    fits.writeto(data_file_path, orif, hdu_list[0].header, overwrite=True)

    # Close the original FITS file
    hdu_list.close()

    # Astrometry
    dir_output="/home/yymx2/HDD/data6/GFA/image2/tempfiles"
    input_command="solve-field "+"--dir "+dir_output+" --scale-units degwidth --scale-low 0.03 --scale-high 0.16 --no-verify --no-plots --crpix-center -O --ra "+ra_in+" --dec "+dec_in+" --radius 5 "+dir_out+"/"+newname 
    os.system(input_command)
    #subprocess.call('export PATH="$PATH:/usr/local/astrometry/bin"', shell=True)

    #move the images and remove temp files
    savedname=newname[0:len(newname)-4] + 'new'
    dir_final='/home/yymx2/HDD/data6/GFA/image2/astroimg/'
    allnew='astro_'+newname
    os.system('cp '+dir_output+'/'+savedname+' '+dir_final+allnew)
    #os.system('cp '+dir_output+'/'+newname[0:len(newname)-4] + 'corr'+' '+'/home/yymx2/HDD/data6/GFA/image2/astroimg/starcatalog/'+'C'+allnew)
    os.system('rm '+dir_output+'/*')
