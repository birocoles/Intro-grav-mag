"""
Auxiliary functions based on those presented by Ciaran Beggan (British Geological Survey),
available at https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field,
for evaluating the geomagnetic field components from IGRF coefficients.
"""

import os
import numpy as np
from numpy import degrees, radians
from math import pi
import warnings
from coordinates import transform, geodetic_elipsoids, utils


class igrf: # A simple class to put the igrf file values into
  def __init__(self, time, coeffs, parameters):
     self.time = time
     self.coeffs = coeffs
     self.parameters = parameters
        

def load_shcfile(filepath, leap_year=None):
    """
    Load shc-file and return coefficient arrays.

    Parameters
    ----------
    filepath : str
        File path to spherical harmonic coefficient shc-file.
    leap_year : {True, False}, optional
        Take leap year in time conversion into account (default). Otherwise,
        use conversion factor of 365.25 days per year.

    Returns
    -------
    time : ndarray, shape (N,)
        Array containing `N` times for each model snapshot in modified
        Julian dates with origin January 1, 2000 0:00 UTC.
    coeffs : ndarray, shape (nmax(nmax+2), N)
        Coefficients of model snapshots. Each column is a snapshot up to
        spherical degree and order `nmax`.
    parameters : dict, {'SHC', 'nmin', 'nmax', 'N', 'order', 'step'}
        Dictionary containing parameters of the model snapshots and the
        following keys: ``'SHC'`` shc-file name, `nmin` minimum degree,
        ``'nmax'`` maximum degree, ``'N'`` number of snapshot models,
        ``'order'`` piecewise polynomial order and ``'step'`` number of
        snapshots until next break point. Extract break points of the
        piecewise polynomial with ``breaks = time[::step]``.

    """
    leap_year = True if leap_year is None else leap_year
    
    read_values_flag = False

    with open(filepath, 'r') as f:

        data = np.array([])
        for line in f.readlines():

            if line[0] == '#':
                continue

            read_line = np.fromstring(line, sep=' ')
            if (read_line.size == 7) and not(read_values_flag):
                name = os.path.split(filepath)[1]  # file name string
                values = [name] + read_line.astype(int).tolist()
                read_values_flag = True
            else:
                data = np.append(data, read_line)

        # unpack parameter line
        keys = ['SHC', 'nmin', 'nmax', 'N', 'order', 'step', 'start_year', 'end_year']
        parameters = dict(zip(keys, values))
        
        time = data[:parameters['N']]
        coeffs = data[parameters['N']:].reshape((-1, parameters['N']+2))
        coeffs = np.squeeze(coeffs[:, 2:])  # discard columns with n and m


    return igrf(time, coeffs, parameters)


def check_lat_lon_bounds(latd, lond):
    
    """ Check the bounds of the given lat, long are within -90 to +90 and -180 
    to +180 degrees 
    
    Paramters
    ---------
    latd, lond : int or float
    
    An exception is raised if the input is invalid.
    
    """
    
    if latd < -90 or latd > 90:
        raise ValueError(f'Latitude {latd} out of bounds [-90, 90].')
    if lond < -360 or lond > 360:
        raise ValueError(f'Longitude {lond} out of bounds [-360, 360].')


def gg_to_geo(h, gdcolat):
    """
    Compute geocentric colatitude and radius from geodetic colatitude and
    height.

    Parameters
    ----------
    h : ndarray, shape (...)
        Altitude in kilometers.
    gdcolat : ndarray, shape (...)
        Geodetic colatitude

    Returns
    -------
    radius : ndarray, shape (...)
        Geocentric radius in kilometers.
    theta : ndarray, shape (...)
        Geocentric colatitude in degrees.
    
    sd : ndarray shape (...) 
        rotate B_X to gd_lat 
    cd :  ndarray shape (...) 
        rotate B_Z to gd_lat 

    References
    ----------
    Equations (51)-(53) from "The main field" (chapter 4) by Langel, R. A. in:
    "Geomagnetism", Volume 1, Jacobs, J. A., Academic Press, 1987.
    
    Malin, S.R.C. and Barraclough, D.R., 1981. An algorithm for synthesizing 
    the geomagnetic field. Computers & Geosciences, 7(4), pp.401-405.

    """
    # Use WGS-84 ellipsoid parameters

    eqrad = 6378.137 # equatorial radius
    flat  = 1/298.257223563
    plrad = eqrad*(1-flat) # polar radius
    ctgd  = np.cos(np.deg2rad(gdcolat))
    stgd  = np.sin(np.deg2rad(gdcolat))
    a2    = eqrad*eqrad
    a4    = a2*a2
    b2    = plrad*plrad
    b4    = b2*b2
    c2    = ctgd*ctgd
    s2    = 1-c2
    rho   = np.sqrt(a2*s2 + b2*c2)
    
    rad   = np.sqrt(h*(h+2*rho) + (a4*s2+b4*c2)/rho**2)

    cd    = (h+rho)/rad
    sd    = (a2-b2)*ctgd*stgd/(rho*rad)
    
    cthc  = ctgd*cd - stgd*sd           # Also: sthc = stgd*cd + ctgd*sd
    thc   = np.rad2deg(np.arccos(cthc)) # arccos returns values in [0, pi]
    
    return rad, thc, sd, cd


def synth_values(coeffs, radius, theta, phi, \
                 nmax=None, nmin=None, grid=None):
    """
    Based on chaosmagpy from Clemens Kloss (DTU Space, Copenhagen)
    Computes radial, colatitude and azimuthal field components from the
    magnetic potential field in terms of spherical harmonic coefficients.
    A reduced version of the DTU synth_values chaosmagpy code

    Parameters
    ----------

    coeffs : ndarray, shape (..., N)
        Coefficients of the spherical harmonic expansion. The last dimension is
        equal to the number of coefficients, `N` at the grid points.
    radius : float or ndarray, shape (...)
        Array containing the radius in kilometers.
    theta : float or ndarray, shape (...)
        Array containing the colatitude in degrees
        :math:`[0^\\circ,180^\\circ]`.
    phi : float or ndarray, shape (...)
        Array containing the longitude in degrees.
    nmax : int, positive, optional
        Maximum degree up to which expansion is to be used (default is given by
        the ``coeffs``, but can also be smaller if specified
        :math:`N` :math:`\\geq` ``nmax`` (``nmax`` + 2)
    nmin : int, positive, optional
        Minimum degree from which expansion is to be used (defaults to 1).
        Note that it will just skip the degrees smaller than ``nmin``, the
        whole sequence of coefficients 1 through ``nmax`` must still be given
        in ``coeffs``.
        Magnetic field source (default is an internal source).
    grid : bool, optional
        If ``True``, field components are computed on a regular grid. Arrays
        ``theta`` and ``phi`` must have one dimension less than the output grid
        since the grid will be created as their outer product (defaults to
        ``False``).

    Returns
    -------
    B_radius, B_theta, B_phi : ndarray, shape (...)
        Radial, colatitude and azimuthal field components.

    Notes
    -----
    The function can work with different grid shapes, but the inputs have to
    satisfy NumPy's `broadcasting rules \\
    <https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html>`_
    (``grid=False``, default). This also applies to the dimension of the
    coefficients ``coeffs`` excluding the last dimension.

    The optional parameter ``grid`` is for convenience. If set to ``True``,
    a singleton dimension is appended (prepended) to ``theta`` (``phi``)
    for broadcasting to a regular grid. The other inputs ``radius`` and
    ``coeffs`` must then be broadcastable as before but now with the resulting
    regular grid.

    Examples
    --------
    The most straight forward computation uses a fully specified grid. For
    example, compute the magnetic field at :math:`N=50` grid points on the
    surface.

    .. code-block:: python

      import igrf_utils as iut
      import numpy as np

      N = 13
      coeffs = np.ones((3,))  # degree 1 coefficients for all points
      radius = 6371.2 * np.ones((N,))  # radius of 50 points in km
      phi = np.linspace(-180., 180., num=N)  # azimuth of 50 points in deg.
      theta = np.linspace(0., 180., num=N)  # colatitude of 50 points in deg.

      B = iut.synth_values(coeffs, radius, theta, phi)
      print([B[num].shape for num in range(3)])  # (N,) shaped output

    Instead of `N` points, compute the field on a regular
    :math:`N\\times N`-grid in azimuth and colatitude (slow).

    .. code-block:: python

      radius_grid = 6371.2 * np.ones((N, N))
      phi_grid, theta_grid = np.meshgrid(phi, theta)  # regular NxN grid

      B = iut.synth_values(coeffs, radius_grid, theta_grid, phi_grid)
      print([B[num].shape for num in range(3)])  # NxN output

    But this is slow since some computations on the grid are executed several
    times. The preferred method is to use NumPy's broadcasting rules (fast).

    .. code-block:: python

      radius_grid = 6371.2  # float, () or (1,)-shaped array broadcasted to NxN
      phi_grid = phi[None, ...]  # prepend singleton: 1xN
      theta_grid = theta[..., None]  # append singleton: Nx1

      B = iut.synth_values(coeffs, radius_grid, theta_grid, phi_grid)
      print([B[num].shape for num in range(3)])  # NxN output

    For convenience, you can do the same by using ``grid=True`` option.

    .. code-block:: python

      B = iut.synth_values(coeffs, radius_grid, theta, phi, grid=True)
      print([B[num].shape for num in range(3)])  # NxN output

    Remember that ``grid=False`` (or left out completely) will result in
    (N,)-shaped outputs as in the first example.

    """

    # ensure ndarray inputs
    coeffs = np.array(coeffs, dtype=float)
    radius = np.array(radius, dtype=float) / 6371.2  # Earth's average radius
    theta = np.array(theta, dtype=float)
    phi = np.array(phi, dtype=float)

    if np.amin(theta) <= 0.0 or np.amax(theta) >= 180.0:
        if np.amin(theta) == 0.0 or np.amax(theta) == 180.0:
            warnings.warn('The geographic poles are included.')
        else:
            raise ValueError('Colatitude outside bounds [0, 180].')

    if nmin is None:
        nmin = 1
    else:
        assert nmin > 0, 'Only positive nmin allowed.'

    # handle optional argument: nmax
    nmax_coeffs = int(np.sqrt(coeffs.shape[-1] + 1) - 1)  # degree
    if nmax is None:
        nmax = nmax_coeffs
    else:
        assert nmax > 0, 'Only positive nmax allowed.'

    if nmax > nmax_coeffs:
        warnings.warn('Supplied nmax = {0} and nmin = {1} is '
                      'incompatible with number of model coefficients. '
                      'Using nmax = {2} instead.'.format(
                        nmax, nmin, nmax_coeffs))
        nmax = nmax_coeffs

    if nmax < nmin:
        raise ValueError(f'Nothing to compute: nmax < nmin ({nmax} < {nmin}.)')

    # initialize radial dependence given the source
    r_n = radius**(-(nmin+2))

    # compute associated Legendre polynomials as (n, m, theta-points)-array
    Pnm = legendre_poly(nmax, theta)

    # save sinth for fast access
    sinth = Pnm[1, 1]

    # calculate cos(m*phi) and sin(m*phi) as (m, phi-points)-array
    phi = radians(phi)
    cmp = np.cos(np.multiply.outer(np.arange(nmax+1), phi))
    smp = np.sin(np.multiply.outer(np.arange(nmax+1), phi))

    # allocate arrays in memory
    B_radius = 0.
    B_theta = 0.
    B_phi = 0.

    num = nmin**2 - 1
    for n in range(nmin, nmax+1):
        B_radius += (n+1) * Pnm[n, 0] * r_n * coeffs[..., num]

        B_theta += -Pnm[0, n+1] * r_n * coeffs[..., num]

        num += 1

        for m in range(1, n+1):
            B_radius += ((n+1) * Pnm[n, m] * r_n
                             * (coeffs[..., num] * cmp[m]
                                + coeffs[..., num+1] * smp[m]))
            
            B_theta += (-Pnm[m, n+1] * r_n
                        * (coeffs[..., num] * cmp[m]
                           + coeffs[..., num+1] * smp[m]))

            with np.errstate(divide='ignore', invalid='ignore'):
                # handle poles using L'Hopital's rule
                div_Pnm = np.where(theta == 0., Pnm[m, n+1], Pnm[n, m] / sinth)
                div_Pnm = np.where(theta == degrees(pi), -Pnm[m, n+1], div_Pnm)

            B_phi += (m * div_Pnm * r_n
                      * (coeffs[..., num] * smp[m]
                         - coeffs[..., num+1] * cmp[m]))

            num += 2

        r_n = r_n / radius  # equivalent to r_n = radius**(-(n+2))
 
    # Set B_theta negative to get the value along latitude instead colatitude
    # return B_radius, B_theta, B_phi
    return B_radius, -B_theta, B_phi


def legendre_poly(nmax, theta):
    """
    Returns associated Legendre polynomials `P(n,m)` (Schmidt quasi-normalized)
    and the derivative :math:`dP(n,m)/d\\theta` evaluated at :math:`\\theta`.

    Parameters
    ----------
    nmax : int, positive
        Maximum degree of the spherical expansion.
    theta : ndarray, shape (...)
        Colatitude in degrees :math:`[0^\\circ, 180^\\circ]`
        of arbitrary shape.

    Returns
    -------
    Pnm : ndarray, shape (n, m, ...)
          Evaluated values and derivatives, grid shape is appended as trailing
          dimensions. `P(n,m)` := ``Pnm[n, m, ...]`` and `dP(n,m)` :=
          ``Pnm[m, n+1, ...]``

    """

    costh = np.cos(radians(theta))
    sinth = np.sqrt(1-costh**2)

    Pnm = np.zeros((nmax+1, nmax+2) + costh.shape)
    Pnm[0, 0] = 1  # is copied into trailing dimenions
    Pnm[1, 1] = sinth  # write theta into trailing dimenions via broadcasting

    rootn = np.sqrt(np.arange(2 * nmax**2 + 1))

    # Recursion relations after Langel "The Main Field" (1987),
    # eq. (27) and Table 2 (p. 256)
    for m in range(nmax):
        Pnm_tmp = rootn[m+m+1] * Pnm[m, m]
        Pnm[m+1, m] = costh * Pnm_tmp

        if m > 0:
            Pnm[m+1, m+1] = sinth*Pnm_tmp / rootn[m+m+2]

        for n in np.arange(m+2, nmax+1):
            d = n * n - m * m
            e = n + n - 1
            Pnm[n, m] = ((e * costh * Pnm[n-1, m] - rootn[d-e] * Pnm[n-2, m])
                         / rootn[d])

    # dP(n,m) = Pnm(m,n+1) is the derivative of P(n,m) vrt. theta
    Pnm[0, 2] = -Pnm[1, 1]
    Pnm[1, 2] = Pnm[1, 0]
    for n in range(2, nmax+1):
        Pnm[0, n+1] = -np.sqrt((n*n + n) / 2) * Pnm[n, 1]
        Pnm[1, n+1] = ((np.sqrt(2 * (n*n + n)) * Pnm[n, 0]
                       - np.sqrt((n*n + n - 2)) * Pnm[n, 2]) / 2)

        for m in np.arange(2, n):
            Pnm[m, n+1] = (0.5*(np.sqrt((n + m) * (n - m + 1)) * Pnm[n, m-1]
                           - np.sqrt((n + m + 1) * (n - m)) * Pnm[n, m+1]))

        Pnm[n, n+1] = np.sqrt(2 * n) * Pnm[n, n-1] / 2

    return Pnm


def igrf_GGS2TCS(
    longitude,
    latitude,
    height,
    northward,
    eastward,
    inward,
    longitude0,
    latitude0,
    height0):
    """
    Convert the IGRF field components from the geocentric geodetic system (GGS) to the topocentric Cartesian system (TCS).

    parameters
    ----------
    longitude, latitude, height : numpy arrays 1d
        Spatial coordinates of the IGRF points referred to the geocentric geodetic system (GGS).
    northward, eastward, inward : numpy arrays 1d
        Field components of the IGRF in the geocentric geodetic system (GGS).
    longitude0, latitude0, height0 : scalars
        Coordinates of the origin of the TCS referred to the GGS.

    returns
    -------
    x, y, z : numpy arrays 1d
        Field components of the IGRF referred to the TCS.
    """

    if type(height) != np.ndarray:
        raise ValueError("height must be a numpy array")
    if type(latitude) != np.ndarray:
        raise ValueError("latitude must be a numpy array")
    if type(longitude) != np.ndarray:
        raise ValueError("longitude must be a numpy array")
    if (height.ndim != 1) or (latitude.ndim != 1) or (longitude.ndim != 1):
        raise ValueError("height, latitude and longitude must be numpy arrays 1d")
    if (height.size != latitude.size) or (height.size != longitude.size):
        raise ValueError("height, latitude and longitude must have the same size")
    if type(northward) != np.ndarray:
        raise ValueError("northward must be a numpy array")
    if type(eastward) != np.ndarray:
        raise ValueError("eastward must be a numpy array")
    if type(inward) != np.ndarray:
        raise ValueError("inward must be a numpy array")
    if (northward.ndim != 1) or (eastward.ndim != 1) or (inward.ndim != 1):
        raise ValueError("northward, eastward and inward must be numpy arrays 1d")
    if (northward.size != eastward.size) or (northward.size != inward.size):
        raise ValueError("northward, eastward and inward must have the same size")
    if isinstance(height0, (int, float)) is False:
        raise ValueError("height0 must be float or int")
    if isinstance(latitude0, (int, float)) is False:
        raise ValueError("latitude0 must be float or int")
    if isinstance(longitude0, (int, float)) is False:
        raise ValueError("longitude0 must be float or int")
    
    # get major and minor semiaxes
    a, f = geodetic_elipsoids.WGS84()
    b = (1-f)*a

    # compute origin coordinates in the geocentric Cartesian system (GCS)
    X0, Y0, Z0 = transform.GGC2GCC(
        height = np.asarray(height0), 
        latitude = np.asarray(latitude0), 
        longitude = np.asarray(longitude0), 
        major_semiaxis = a,
        minor_semiaxis = b
    )

    # Compute the local basis of the geocentric geodetic system (GGS) at the origin of 
    # the topocentric Cartesian system (TCS) and arrange terms in an 1d array
    h0_X, h0_Y, h0_Z = utils.unit_vector_normal(
        latitude = latitude0,
        longitude = longitude0
    )
    varphi0_X, varphi0_Y, varphi0_Z = utils.unit_vector_latitude(
        latitude = latitude0,
        longitude = longitude0
    )
    lambda0_X, lambda0_Y = utils.unit_vector_longitude(
        longitude = longitude0
    )

    # define the unit vectors forming the TCS
    x_hat = np.array([varphi0_X, varphi0_Y, varphi0_Z])
    y_hat = np.array([lambda0_X, lambda0_Y])
    z_hat = -np.array([h0_X, h0_Y, h0_Z])

    # Elements of the rotation matrix associated with the local basis of the geocentric geodetic system (GGS) at the igrf points
    # The routine 'coordinates.utils.rotation_NED' returns a matrix with 8 columns. 
    # Columns 0, 3 and 6 contain the X, Y and Z components of the unit vector along the latitude direction. 
    # Columns 1 and 4 contain the X and Y (Z is always null) components of the unit vector alonf the longitude direction. 
    # Columns 2, 5 and 8 contain the X, Y and Z components of the unit vector along the inward direction. 
    # The components X, Y and Z are referred to the geocentric Cartesian system (GCS). 
    # Each row of this 8-column matrix contains the elements components of the unit vectors at a given point.
    R_NED = utils.rotation_NED(
        latitude = latitude, 
        longitude = longitude
        )

    # Elements of the resultant matrix C transforming the IGRF field components.
    # Each row of matrix C contains the elements 00, 01, 02, 10, 11, 12, 20, 21 and 22 of the matrix given by
    # the product R_TCS.T R_NED, where R_TCS is a constant rotation matrix whose columns are defined by the unit
    # vectors x_hat, y_hat and z_hat defining the axes of the TCS and R_NED is a rotation matrix whose columns are 
    # defined by the unit vectors pointing northwar5d, eastward and inward in the GGS at a given point.
    C00 = np.dot(R_NED[:,[0,3,6]], x_hat)
    C10 = np.dot(R_NED[:,[0,3]], y_hat)
    C20 = np.dot(R_NED[:,[0,3,6]], z_hat)
    C01 = np.dot(R_NED[:,[1,4]], x_hat[:2])
    C11 = np.dot(R_NED[:,[1,4]], y_hat)
    C21 = np.dot(R_NED[:,[1,4]], z_hat[:2])
    C02 = np.dot(R_NED[:,[2,5,7]], x_hat)
    C12 = np.dot(R_NED[:,[2,5]], y_hat)
    C22 = np.dot(R_NED[:,[2,5,7]], z_hat)

    # IGRF field components in the TCS
    x = (C00*northward + C01*eastward + C02*inward)
    y = (C10*northward + C11*eastward + C12*inward)
    z = (C20*northward + C21*eastward + C22*inward)

    return x, y, z


def compute_declination_inclination(northward, eastward, inward):
    """
    Given the field components along the northward, eastward and inward directions, 
    compute the declination and inclination.

    parameters
    ----------
    northward, eastward, inward : numpy arrays 1d
        Field components along the northward, eastward and inward directions.

    returns
    -------
    declination, inclination : numpy arrays 1d
        Pointwise declination and inclination of the field (in degrees).
    """
    if type(northward) != np.ndarray:
        raise ValueError("northward must be a numpy array")
    if type(eastward) != np.ndarray:
        raise ValueError("eastward must be a numpy array")
    if type(inward) != np.ndarray:
        raise ValueError("inward must be a numpy array")
    if (northward.ndim != 1) or (eastward.ndim != 1) or (inward.ndim != 1):
        raise ValueError("northward, eastward and inward must be numpy arrays 1d")
    if (northward.size != eastward.size) or (northward.size != inward.size):
        raise ValueError("northward, eastward and inward must have the same size")

    declination = np.rad2deg(np.arctan2(eastward,northward))
    inclination = np.rad2deg(np.arctan(inward/np.sqrt(northward**2 + eastward**2)))

    return declination, inclination