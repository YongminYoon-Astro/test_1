from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from astropy.io import ascii
import numpy as np
import math
import photutils.detection as pd
from astropy.stats import sigma_clip
import sys
import warnings
from astropy.utils.exceptions import AstropyWarning

#default input for deriving offsets
boxsize=40
crit_out=0.25
peakmax=30000#30000
peakmin=150#150

def load_image_and_wcs(image_file): 
    #loading image    
    image_data_p, header = fits.getdata(image_file, ext=0, header=True)
    #loading WCS info
    wcs = WCS(header)
    # Extracting CRVAL1 and CRVAL2 from the header
    #crval1 = header['CRVAL1']
    #crval2 = header['CRVAL2']
    return image_data_p, header, wcs

def load_only_image(image_file):
    # Ignore FITS header warning
    warnings.filterwarnings('ignore', category=AstropyWarning)
    #loading image    
    image_data_p = fits.getdata(image_file, ext=0)
    return image_data_p

def background(image_data_p):
    #sigma clipping to derive background and its standard deviation
    image_data=np.zeros_like(image_data_p, dtype=float)
    x_split = 511 

    region1 = image_data_p[:, :x_split]   
    sigclip1 = sigma_clip(region1, sigma=3, maxiters=False, masked=False)
    avg1 = np.mean(sigclip1)
    image_data[:, :x_split] =  region1 - avg1

    region2 = image_data_p[:, x_split:]
    sigclip2 = sigma_clip(region2, sigma=3, maxiters=False, masked=False)
    avg2 = np.mean(sigclip2)
    image_data[:, x_split:] = region2 - avg2

    sigclip=sigma_clip(image_data, sigma=3, maxiters=False, masked=False)
    stddev=np.std(sigclip)

    return image_data, stddev

#loading guide star catalog
def load_star_catalog(file_name, crval1, crval2):
    with fits.open(file_name) as hdul:
        data = hdul[1].data
        ra_p = data['index_ra']
        dec_p = data['index_dec']
        flux = data['FLUX']
    ra1_rad = np.radians(crval1)
    dec1_rad = np.radians(crval2)
    ra2_rad = np.radians(ra_p)
    dec2_rad = np.radians(dec_p)
    return ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux

def select_stars(ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux):
    # Calculate the angular distance
    delta_sigma = np.arccos(np.sin(dec1_rad) * np.sin(dec2_rad) + np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(ra1_rad - ra2_rad))
    # Convert back to degrees
    angular_distance_degrees = np.degrees(delta_sigma)
    # Filter coordinates whose angular distances are less than 0.06 degree
    mask = (angular_distance_degrees < 0.06) & (flux > peakmin)
    ra = ra_p[mask]
    dec = dec_p[mask]
    return ra, dec


def radec_to_xy_stars(ra, dec, wcs):
    #converting ra dec of guide star into x y position in the image
    dra=np.zeros(len(ra), dtype=float)
    ddec=np.zeros(len(ra), dtype=float)
    dra_f=np.zeros(len(ra), dtype=float)
    ddec_f=np.zeros(len(ra), dtype=float)
    for ii in range(len(ra)):
        dra_p, ddec_p = wcs.world_to_pixel_values(ra[ii], dec[ii])    
        dra[ii] = np.round(dra_p)+1
        ddec[ii] = np.round(ddec_p)+1
        dra_f[ii] = dra_p+1
        ddec_f[ii] = ddec_p+1
    return dra, ddec, dra_f, ddec_f


def cal_centroid_offset(dra, ddec, dra_f, ddec_f, stddev):
    dx=[]
    dy=[]
    peakc=[]
    for jj in range(len(ra)):
        try: 
            #finding peak in the cutout image of each guide star
            cutout=image_data[int(ddec[jj]-1-boxsize/2) : int(ddec[jj]-1+boxsize/2+1), int(dra[jj]-1-boxsize/2) :int(dra[jj]-1+boxsize/2+1)]

            thres=np.zeros((cutout.shape[0], cutout.shape[1]), dtype=float)+5*stddev
            peak=pd.find_peaks(cutout,thres, box_size=boxsize/4, npeaks=1)       

            x_peak=peak['x_peak'][0]
            y_peak=peak['y_peak'][0]
            peakv=peak['peak_value'][0]

            peakc.append(peakv)

            nra=int(dra[jj]-(0.5*boxsize-x_peak))
            ndec=int(ddec[jj]-(0.5*boxsize-y_peak))

            #calculating the center of light of each guide star in a smaller cutout image
            cutout2=image_data[int(ndec-1-boxsize/4) : int(ndec-1+boxsize/4+1), int(nra-1-boxsize/4) :int(nra-1+boxsize/4+1)]

            xcs=0
            ycs=0

            for kk in range(cutout2.shape[0]):
                for ll in range(cutout2.shape[1]):
                    xcs += cutout2[kk, ll] * ll
                    ycs += cutout2[kk, ll] * kk

            xc=xcs/np.sum(cutout2)
            yc=ycs/np.sum(cutout2)

            fra=(nra-(boxsize/4-xc))
            fdec=(ndec-(boxsize/4-yc))

            #deriving ra dec offsets corresponding to 1 pixel offset along the 'x-axis'
            x1 = fra  
            y1 = fdec 
            x2 = fra+1  
            y2 = fdec
            ra1, dec1 = wcs.pixel_to_world_values(x1, y1)
            ra2, dec2 = wcs.pixel_to_world_values(x2, y2)
            x1d=(ra2-ra1)*3600
            x2d=(dec2-dec1)*3600
            #print(x1d)
            #print(x2d)
            #print( x1d*math.cos(cdec*math.pi/180.) )

            #deriving ra dec offsets corresponding to 1 pixel offset along the 'y-axis'
            x1 = fra
            y1 = fdec 
            x2 = fra   
            y2 = fdec+1 
            ra1, dec1 = wcs.pixel_to_world_values(x1, y1)
            ra2, dec2 = wcs.pixel_to_world_values(x2, y2)
            y1d=(ra2-ra1)*3600
            y2d=(dec2-dec1)*3600
            #print(y1d)
            #print(y2d)

            #converting the x y offset of guide star locations into ra dec offsets
            dx.append((fra -dra_f[jj]) * x1d + (fdec - ddec_f[jj]) * x2d)
            dy.append((fra -dra_f[jj]) * y1d + (fdec - ddec_f[jj]) * y2d)

        except Exception as e:
            # Handle errors during peak finding
            dx.append(0)
            dy.append(0)
            peakc.append(-1)
            print(f"Error finding peaks: {e}")
            
    return dx, dy, peakc


def peak_select(dx, dy, peakc):
    #only using guide stars whose peak values are between min and max, in order to avoid cosmic rays and too small S/N (due to weather effects)
    peakn=np.array(peakc)
    pind=np.where((peakn>peakmin) & (peakn<peakmax))
    pindn=pind[0]

    dxn=np.array([dx[i] for i in pindn])
    dyn=np.array([dy[i] for i in pindn])
    return dxn, dyn, pindn

def cal_final_offset(dxn, dyn, pindn):
    if len(pindn) > 0.5:
        distances=np.sqrt(dxn**2+dyn**2)
        clipped_data = sigma_clip(distances, sigma=3, maxiters=False)
        cdx = dxn[~clipped_data.mask]
        cdy = dyn[~clipped_data.mask]

        #discarding maximum and minimum offsets when the number of guide stars are larger than four
        if len(cdx) > 4:
            filtered_distances = np.sqrt(cdx**2 + cdy**2)
            max_dist_index = np.argmax(filtered_distances)
            min_dist_index = np.argmin(filtered_distances)
            cdx = np.delete(cdx, [min_dist_index, max_dist_index])
            cdy = np.delete(cdy, [min_dist_index, max_dist_index])

        #averaging the offsets
        fdx=np.mean(cdx)
        fdy=np.mean(cdy)    
        #print(fdx)
        #print(fdy)
        return fdx, fdy
    # if the number of available guide star is zero, there is no offset.
    else:
        return 0, 0


def offset_crit(final_dx, final_dy):
    #mean values for offsets, excluding zero offsets
    ffdx=np.mean([x for x in final_dx if abs(x) > 0])
    ffdy=np.mean([y for y in final_dy if abs(y) > 0])
    print(ffdx)
    print(ffdy)
    #only using the derived offset when offset is larger than crit_out (0.25 arcsec)
    if abs(ffdx) > crit_out or abs(ffdy) > crit_out:
        return ffdx, ffdy
    else:
        return 0, 0



    
if __name__ == "__main__":

    mode = int(sys.argv[1])

    dirr = 'images/'
    astro_files = [
        dirr+'astro_procKMTNge.20231211T101128.0001.fits',
        dirr+'astro_procKMTNgn.20231211T101128.0001.fits',
        dirr+'astro_procKMTNgs.20231211T101128.0001.fits',
        dirr+'astro_procKMTNgw.20231211T101128.0001.fits'
    ]

    proc_files = [
        dirr+'procKMTNge.20231211T101128.0001.fits',
        dirr+'procKMTNgn.20231211T101128.0001.fits',
        dirr+'procKMTNgs.20231211T101128.0001.fits',
        dirr+'procKMTNgw.20231211T101128.0001.fits'
    ]

    final_dx = []
    final_dy = []
    
    if mode == 0:
        for astro_file in astro_files:
            image_file = get_pkg_data_filename(astro_file)        
            image_data_p, header, wcs = load_image_and_wcs(image_file)        
            crval1, crval2 = header['CRVAL1'], header['CRVAL2']       
            image_data, stddev=background(image_data_p)
            ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux = load_star_catalog('combined.fits', crval1, crval2)
            ra, dec = select_stars(ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux)
            dra, ddec, dra_f, ddec_f = radec_to_xy_stars(ra, dec, wcs)
            dx, dy, peakc = cal_centroid_offset(dra, ddec, dra_f, ddec_f, stddev)
            dxn, dyn, pindn = peak_select(dx, dy, peakc)        
            fdx, fdy = cal_final_offset(dxn, dyn, pindn)

            final_dx.append(fdx)
            final_dy.append(fdy)

        ffdx, ffdy = offset_crit(final_dx, final_dy)
        print(ffdx)
        print(ffdy)
        
            
    elif mode == 1:
        for astro_file, proc_file in zip(astro_files, proc_files):
            image_file = get_pkg_data_filename(astro_file)        
            image_data_p, header, wcs = load_image_and_wcs(image_file)        
            crval1, crval2 = header['CRVAL1'], header['CRVAL2']       

            image_file = get_pkg_data_filename(proc_file)
            image_data_p = load_only_image(image_file)
            
            image_data, stddev=background(image_data_p)
            ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux = load_star_catalog('combined.fits', crval1, crval2)
            ra, dec = select_stars(ra1_rad, dec1_rad, ra2_rad, dec2_rad, ra_p, dec_p, flux)
            dra, ddec, dra_f, ddec_f = radec_to_xy_stars(ra, dec, wcs)
            dx, dy, peakc = cal_centroid_offset(dra, ddec, dra_f, ddec_f, stddev)
            dxn, dyn, pindn = peak_select(dx, dy, peakc)        
            fdx, fdy = cal_final_offset(dxn, dyn, pindn)

            final_dx.append(fdx)
            final_dy.append(fdy)

        ffdx, ffdy = offset_crit(final_dx, final_dy)
        print(ffdx)
        print(ffdy)
        

       
        
