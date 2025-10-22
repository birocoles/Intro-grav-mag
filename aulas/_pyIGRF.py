import numpy as np
from scipy import interpolate
import _igrf_utils as iut
from tqdm import tqdm



def pyIGRF(longitude, latitude, height, date, geodetic=True):
    '''
    Function based on that presented by Ciaran Beggan (British Geological Survey),
    available at https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field,
    for evaluating the geomagnetic field components from IGRF coefficients.

    parameters
    ----------
    longitude : numpy array 1D
        Longitude values (in degrees) of the evaluation points.
    latitude : numpy array 1D
        Latitude values (in degrees) of the evaluation points with respect
        to the WGS84 reference elipsoid.
    height : numpy array 1D
        Geometric height values (in meters) of the evaluation points with respect
        to the WGS84 reference elipsoid.
    date : numpy array 1D
        Date in decimal year.
    geodetic : boolean
        If True, the function returns the field components in the geocentric
        geodetic coordinate system. Otherwise, in the geocentric, spherical 
        coordinate system. Default is True.

    returns
    -------
    northward, eastward, inward : numpy arrays 1D
        Northward, eastward and inward field components in geodetic or spherical 
        coordinates (see the parameter 'geodetic').

    '''

    # Load the coefficients file
    IGRF_FILE = "IGRF14.SHC"
    igrf = iut.load_shcfile(IGRF_FILE, None)

    # Parse the inputs for computing the main field values. 
    # Convert geodetic to geocentric coordinates if required 
    try:
        latitude = np.asarray(latitude, dtype='float')
        longitude = np.asarray(longitude, dtype='float')
    except ValueError:
        raise ValueError('Could not convert latitude or longitude to float.')
    
    if np.any(latitude < -90) or np.any(latitude > 90):
        raise ValueError('Latitude out of bounds [-90, 90].')
    if np.any(longitude < -360) or np.any(longitude > 360):
        raise ValueError('Longitude out of bounds [-360, 360].')

    colat = 90-latitude

    # Coordinate system: geodetic (shape of Earth using the WGS-84 ellipsoid)
    # Convert from geodetic to spherical
    try:
        # Convert from meter to km
        h = np.asarray(height*1e-3, dtype='float')
    except ValueError:
        raise ValueError('Could not convert height to float.')
    radius, colat, sd, cd = iut.gg_to_geo(h, colat)

    # Get date in decimal date in years 1900-2030
    if np.any(date < 1900) or np.any(date > 2035):
        raise ValueError('Date out of bounds [1900, 2035].')

    # Compute the interpolator for date
    f = interpolate.interp1d(
        igrf.time, igrf.coeffs, fill_value='extrapolate'
        )

    # Iterate over data points
    B = list()
    for (r, col, lon, d, sdi, cdi) in tqdm(zip(radius, colat, longitude, date, sd, cd), total=radius.size):
        # Interpolate the geomagnetic coefficients to the desired date(s)
        coeffs = f(d)    
        
        # Compute the main field B_r, B_lat and B_lon value for the location
        B_r, B_lat, B_lon = iut.synth_values(
            coeffs.T, r, col, lon, igrf.parameters['nmax']
            )

        # Rearrange to northward, eastward and inward components 
        northward = B_lat
        eastward = B_lon
        inward = -B_r

        if geodetic is True:
            # Rotate back to geodetic coords if needed
            t = northward
            X = northward*cdi + inward*sdi
            Z = inward*cdi - t*sdi
        
        B.append([northward, eastward, inward])

    B = np.vstack(B)

    # Northward, eastward and inward field components
    return B[:,0], B[:,1], B[:,2]