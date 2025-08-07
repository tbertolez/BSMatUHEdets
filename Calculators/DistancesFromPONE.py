# Script by Alba Burgos, revised by Toni Bertólez

# Uses a topographic file with the elevation of the seabed and adjacent surfaces 
# to compute the layers crossed by a particle from the horizontal plane, to P-ONE.
# Write in a file a table (dwater1, drock, dwater2) for every (varphi,theta)
import os
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import time

R_earth_m = 6371.0 * 1000  #in meters
arca_latitude = 47.742345 # Cascadia Basin, as in https://www.oceannetworks.ca/observatories/locations/, also https://cdn.onc-prod.intergalactic.space/NC_Information_for_Mariners_CB_1_b8624a7b93.pdf
arca_longitude = -127.729272
arca_depth = -2662  # Depth in meters 

def spherical_to_cartesian(lon, lat, elevation):
    # Convert latitude and longitude to Cartesian coordinates for the topography
    X = (R_earth_m + elevation) * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    Y = (R_earth_m + elevation) * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    Z = (R_earth_m + elevation) * np.sin(np.radians(lat))

    return np.array([X, Y, Z])

# Function to calculate azimuth and elevation relative to the local horizontal plane at ARCA 
def calculate_angles(azimuth_deg, elevation_deg, local_vertical_vector):
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    
    # Normal vector to local horizontal plane at ARCA
    local_horizontal_normal = local_vertical_vector / np.linalg.norm(local_vertical_vector)
    
    # Create a horizontal plane at P0, with azimuthal rotation (around the z-axis)
    horizontal_x = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0])  # Initial horizontal direction
    horizontal_x = horizontal_x - np.dot(horizontal_x, local_horizontal_normal) * local_horizontal_normal
    horizontal_x /= np.linalg.norm(horizontal_x)
    
    # Elevation will adjust the direction out of the local horizontal plane
    d_x = horizontal_x * np.cos(elevation_rad)
    d_z = local_horizontal_normal * np.sin(elevation_rad)
    
    # Resultant direction vector (normalized)
    ray_direction = d_x + d_z
    return ray_direction

def compute_direction_vector(azimuth_deg, elevation_deg, latitude_deg = arca_latitude, longitude_deg = arca_longitude):
    # Convert angles to radians
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    lat_rad = np.deg2rad(latitude_deg)
    lon_rad = np.deg2rad(longitude_deg)

    # 1. Compute local "up" vector (radial direction)
    local_up = np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])

    # 2. Compute local "North" (project North Pole onto tangent plane)
    north_pole = np.array([0, 0, 1])  # ECEF North Pole direction
    local_north = north_pole - np.dot(north_pole, local_up) * local_up
    local_north /= np.linalg.norm(local_north)

    # 3. Compute local "East" (cross product, now guaranteed correct)
    local_east = np.cross(local_up, local_north) # I think this is west :(
    local_east = np.cross(local_north, local_up)

    # 4. Compute horizontal direction (azimuth)
    horizontal_dir = (
        np.cos(azimuth_rad) * local_north +
        np.sin(azimuth_rad) * local_east
    )

    # 5. Combine with elevation
    direction = (
        np.cos(elevation_rad) * horizontal_dir +
        np.sin(elevation_rad) * local_up
    )

    return direction / np.linalg.norm(direction)

# Here we load a file with the topography around the detector
# Can be obtained at https://download.gebco.net/, choosing wide enough boundaries around P-ONE, and downloading the "2D netCDF Grid".
homedir = os.path.realpath(__file__)[:-len('Calculators/DistancesFromPONE.py')]
file_path = homedir+"Data/P-ONE/gebco_2024_n60.0_s17.0_w-157.0_e-97.0.nc"
ds = xr.open_dataset(file_path)

# fine controls how fine we use the (lon,lat) grid. Decrease for more precision
fine = 100 # 10 is very good precision, 100 is more chill
lons = ds['lon'].values[::fine]
lats = ds['lat'].values[::fine]
elevation = ds['elevation'].values[::fine, ::fine]
lon_grid, lat_grid = np.meshgrid(lons, lats)

# position of the ARCA detector
P0 = spherical_to_cartesian(arca_longitude, arca_latitude, arca_depth)

coords = np.zeros((*elevation.shape, 3))
coords_spherical = np.zeros((*elevation.shape, 3))

for i in range(lon_grid.shape[0]):
    for j in range(lon_grid.shape[1]):
        lon = lon_grid[i, j]
        lat = lat_grid[i, j]
        elev = elevation[i, j]
        
        coords[i, j] = spherical_to_cartesian(lon, lat, elev)
        coords_spherical[i, j] = spherical_to_cartesian(lon, lat, 0) 

X, Y, Z = coords[..., 0], coords[..., 1], coords[..., 2]
X1, X2, X3 = coords_spherical[..., 0], coords_spherical[..., 1], coords_spherical[..., 2]

def get_distance(azimuth_degrees, elevation_degrees):

    if elevation_degrees < -8.5:
         raise ValueError("The elevation angle asked is too steep for the width of the topography grid. Get a wider grid or reduce the elevation angle.")
    # here I begin defining an observation
    # azimuth_degrees =  0 # Example azimuth angle (from north)
    # elevation_degrees = 7.0#2.1 # Example elevation angle (above horizontal)

    ray_direction = compute_direction_vector(azimuth_degrees, elevation_degrees)

    # Create ray path
    t_values = np.linspace(0, 2000000, 1000)
    ray_x = P0[0] + t_values * ray_direction[0]
    ray_y = P0[1] + t_values * ray_direction[1]
    ray_z = P0[2] + t_values * ray_direction[2]

    distance_xy = np.sqrt((ray_x - P0[0])**2 + (ray_y - P0[1])**2)

    points = np.array([X.flatten(), Y.flatten()]).T
    values = Z.flatten()

    points2 = np.array([X1.flatten(), X2.flatten()]).T
    values2 = X3.flatten()
    ray_elevation = griddata(points, values, (ray_x, ray_y), method='linear')
    ray_elevation_spherical = griddata(points2, values2, (ray_x, ray_y), method='linear')

    intersection_distances = []
    intersection_elevations = []
    waterA_distance = 0
    rockA_distance = 0
    waterB_distance = 0

    # Calculate the difference between the ray's elevation and terrain's Z-values
    difference = ray_elevation - ray_z
    sign_changes = np.where(np.diff(np.sign(difference)))[0]  # Find where the sign changes (indicating intersections)

    if len(sign_changes) > 0:
        for i, idx in enumerate(sign_changes):
            f = interp1d(difference[idx:idx + 2], distance_xy[idx:idx + 2])  # Local interpolation between two points
            intersection_distance = f(0)  # Distance at the intersection
            intersection_distances.append(intersection_distance)

            intersection_elevation = np.interp(intersection_distance, distance_xy, ray_elevation)
            intersection_elevations.append(intersection_elevation)

    ## Now do the same for the Spherical surface (water-level line)
    difference2 = ray_elevation_spherical - ray_z
    sign_changes2 = np.where(np.diff(np.sign(difference2)))[0]  # Find where the sign changes (indicating intersections)

    if len(sign_changes2) > 0:
        for i, idx in enumerate(sign_changes2):
            f2 = interp1d(difference2[idx:idx + 2], distance_xy[idx:idx + 2])  # Local interpolation between two points
            intersection_distance2 = f2(0)  # Distance at the intersection
            intersection_distances.append(intersection_distance2)

            intersection_elevation2 = np.interp(intersection_distance2, distance_xy, ray_elevation_spherical)
            intersection_elevations.append(intersection_elevation2)

    valid_distances = np.array(intersection_distances)[np.isfinite(intersection_distances)]
    valid_elevations = np.array(intersection_elevations)[np.isfinite(intersection_distances)]

    # Process the intervals between intersection points
    # print(valid_distances)
    if len(valid_distances) ==1: 
                waterA_distance = (valid_distances[0]**2+ (valid_elevations[0]-P0[2])**2)**0.5
                rockA_distance = 0
                waterB_distance = 0

    elif len(valid_distances) ==2: 
        waterA_distance = (valid_distances[0]**2+ (valid_elevations[0]-P0[2])**2)**0.5
        waterB_distance = 0

        interval_distanceB = valid_distances[1] - valid_distances[0]
        interval_heightB = valid_elevations[1]-valid_elevations[0]
        distanceB = (interval_distanceB**2 + interval_heightB**2)**(0.5)

        rockA_distance = distanceB

    else: 
        waterA_distance = (valid_distances[0]**2+ (valid_elevations[0]-P0[2])**2)**0.5

        interval_distanceB = valid_distances[1] - valid_distances[0]
        interval_heightB = valid_elevations[1]-valid_elevations[0]
        distanceB = (interval_distanceB**2 + interval_heightB**2)**(0.5)

        rockA_distance = distanceB

        interval_distanceC = valid_distances[-1] - valid_distances[1]
        interval_heightC = valid_elevations[-1]-valid_elevations[1]
        distanceC = (interval_distanceC**2 + interval_heightC**2)**(0.5)

        waterB_distance = distanceC

    # Convert distances to kilometers (if required)

    waterA_distance /= 1000
    rockA_distance /= 1000
    waterB_distance/=1000

    return np.array([waterA_distance,rockA_distance,waterB_distance])

#####################
#  OUTPUT REGION    #
#####################

# We define at which angles we want to compute the layers
azimuth_grid = np.arange(0, 360, 1)  # in degrees
elevation_grid = np.arange(-7.0, 7.0, 0.05)  # in degrees

begin = time.time()
outdata = np.empty((azimuth_grid.shape[0]*elevation_grid.shape[0],5))
for idx_az, az in enumerate(azimuth_grid):
    for idx_th, th in enumerate(elevation_grid):
        dists = get_distance(az,th)
        outdata[idx_az*elevation_grid.shape[0]+idx_th] = np.concatenate((np.array([az,th]),dists))
    print("Done azimuthal angle = {:.3f}º. Cumulative time = {:.2f}".format(az,time.time()-begin))

np.savetxt(homedir+"Data/P-ONE/distances_from_P-ONE_fine.dat",outdata, fmt = ['%.3f','%.3f','%.5e','%.5e','%.5e'])