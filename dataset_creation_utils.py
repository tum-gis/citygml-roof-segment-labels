import os
from shutil import copy2
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import rasterio.features
import shapely
import cv2
import time


###################################################################
# Create dataset of images and labels
###################################################################

def get_image_crop(crop_geometry_json, in_tif):
    """
    Crops a given raster picture in_tif with a given crop window given as json geometry.
    """
    coords = [crop_geometry_json]
    
    # Crop the image by the given geometry
    out_img, out_transform = rasterio.mask.mask(dataset = in_tif, shapes = coords, crop = True, all_touched = True)
    out_meta = in_tif.meta.copy()
    out_crs = in_tif.crs.wkt
    
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform,
                     "crs": out_crs
                    })
    
    return out_img, out_meta

def save_png_image(img_rio, dirpath, filename):
    """
    Takes an RGB image as numpy.ndarray with shape (3, w, h) as output by rasterio (rio)
    and saves it as PNG image using cv2.
    """
    # add file extension if necessary
    if filename[-4:].lower() != ".png": filename += ".png"
    filepath = os.path.join(dirpath, filename)
    
    # Change image shape from (3, w, h) to (w, h, 3) as required by cv2
    img_cv2 = np.moveaxis(img_rio, 0, -1)
    
    # Change image color channels from RGB to BGR as required by cv2
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
    
    return cv2.imwrite(filepath, img_cv2)

def save_tif_image(img_rio, img_meta, dirpath, filename):
    """
    Takes an RGB image as numpy.ndarray with shape (3, w, h) as output by rasterio (rio)
    and its meta data such as crs and transform and saves it as GeoTIFF.
    """
    # add file extension if necessary
    if filename[-4:].lower() != ".tif": filename += ".tif"
    filepath = os.path.join(dirpath, filename)
    
    with rasterio.open(filepath, "w", **img_meta) as dest:
        dest.write(img_rio)

def get_label_segments(segments, crop_geometry, crop_crs):
    """
    Joins a GeoDataFrame of segment polygons with a crop window given as shapely polygon
    by intersection and returns a GeoDataFrame of segment polygons that intersect the crop.
    """
    crop_gdf = gpd.GeoDataFrame({"geometry": crop_geometry}, index = [0])
    crop_gdf.crs = crop_crs
    return segments.sjoin(crop_gdf, how = "inner", predicate = "intersects")

def get_label_segments_sindex(segments, crop_geometry):
    """
    Joins a GeoDataFrame of segment polygons with a crop window given as shapely polygon
    by intersection and returns a GeoDataFrame of segment polygons that intersect the crop.
    """
    return segments.iloc[segments.sindex.query(crop_geometry)]

def save_png_labels(segments_geometry, segments_orientation, shape, transform, dirpath, filename, fill = 0, raster_multiplier = 1):
    """
    Takes two corresponding GeoSeries of (1) segment geometries and (2) their orientation
    classes and turns them into a raster image using given shape and transform, finally
    saves this as PNG image.
    """
    
    # set up raster variable
    r = None
    
    # rasterize given geometries assigning pixel values
    # from corresponding segments_orientation
    try:
        r = rasterio.features.rasterize(
            zip(segments_geometry, segments_orientation),
            out_shape = shape,
            transform = transform,
            fill = fill
        )
        
    except ValueError:
        print("No valid geometry objects found for rasterize. b_id: {}".format(filename))
    
    else:
        # add file extension if necessary
        if filename[-4:].lower() != ".png": filename += ".png"
        filepath = os.path.join(dirpath, filename)

        return cv2.imwrite(filepath, r * raster_multiplier)


###################################################################
# Perform data split
###################################################################

def relocate_samples(ids, filetype, dirpath_src, dirpath_dst):
    n = len(ids)
    for i, b_id in enumerate(ids):
        filename = str(b_id) + filetype
        
        filepath_src = os.path.join(dirpath_src, filename)
        filepath_dst = os.path.join(dirpath_dst, filename)
        
        try:
            copy2(src = filepath_src, dst = filepath_dst)
        except FileNotFoundError:
            print("No such file or directory, skipping: {}".format(filepath_src))
        
        print("Processed " + str(i+1) + " of " + str(n) + " files.", end = "\r")
    
    print("Processed " + str(i+1) + " of " + str(n) + " files.")

def remove_overlapping_buildings(buildings, buildings_subset):
    """
    Finds all buildings that intersect any buildings_subset buffers, which are therefore depicted
    in training samples created from buildings_subset. Removes these intersecting buildings and
    returns only those that do not intersect buildings_subset buffers.
    """
    
    # set different geometry column to be used for sjoin
    buildings_subset = buildings_subset.set_geometry("buffer")
    
    num_buildings_subset = buildings_subset.shape[0]
    
    # remove those buildings that are contained in the subset
    buildings_except_subset = buildings[~buildings["b_id"].isin(buildings_subset["b_id"])]
    
    num_bld_except_subset = buildings_except_subset.shape[0]
    
    print(str(num_bld_except_subset) + " buildings are not contained in the subset.")
    
    # determine all buildings that intersect the image buffers around subset buildings
    buildings_intersect_subset_buffer = buildings.sjoin(buildings_subset, how = "inner", predicate = "intersects")
    
    num_bld_intersect_subset_buffer = len(buildings_intersect_subset_buffer["b_id_left"].unique())
    
    print(str(num_bld_intersect_subset_buffer - num_buildings_subset) + " of these buildings intersect buffers of subset buildings.")
    
    # remove those buildings that overlap any buffers of subset buildings
    buildings_except_subset_overlap = buildings[~buildings["b_id"].isin(buildings_intersect_subset_buffer["b_id_left"].unique())]
    
    num_bld_except_subset_overlap = buildings_except_subset_overlap.shape[0]
    
    print(str(num_bld_except_subset_overlap) + " buildings remain that do not intersect buffers of subset buildings.\n")
    
    return buildings_except_subset_overlap

def get_data_subset(points, numbers, buildings):
    """
    Determines *len(points)* subsets from *buildings*, where around each point in *points*,
    the corresponding number in *numbers* of buildings is selected. The found subsets are then merged.
    Returns therefore a single subset of buildings.
    
    At the moment, there is no mechanism that ensures that the separate subsets don't contain
    the same buildings if the positions (points) are too close to each other. This would lead to the total
    number of subset buildings being smaller than *sum(numbers)*. It could be avoided by introducing
    a mechanism that, after finding a subset, subtracts the buildings contained in this subset from
    the buildings-dataframe that is used in determining the next subset. In the order in which *points* are given.
    """
    
    if type(numbers) == pd.core.series.Series: numbers = numbers.tolist()
    
    subsets = []
    areas = []
    radii = []
    
    for i, point in enumerate(points.geometry):
        
        subset, area_gdf, radius = get_buildings_around_location_by_number(buildings = buildings,
                                                                           location = point,
                                                                           number = numbers[i])
        subsets.append(subset)
        areas.append(area_gdf)
        radii.append(radius)
        
    subset = gpd.GeoDataFrame(pd.concat(subsets, ignore_index=True), crs = subsets[0].crs)
    subset = subset.drop("index_right", axis = 1)
    
    areas = gpd.GeoDataFrame(pd.concat(areas, ignore_index=True), crs = subsets[0].crs)
    
    return subset, areas, radii

def get_buildings_around_location_by_number(buildings, location, number):
    """
    Iteratively determines the radius of a circle that includes exactly *number* building geometries
    from *buildings* around point coordinate *location*. If necessary, tries two different approaches.
    
    Approach 1: Tries to determine the necessary radius by computing the ratio of the target *number* of buildings
    with the number of buildings included within the current radius, then using this ratio to estimate
    the required radius by multiplying it with the current radius.
    Sometimes, this approach may fail to deliver a radius that works. Then the search radius will jiggle around
    within a certain interval around the target radius. If this happens (if the standard deviation of the past
    *mode_switch_threshold_tries* radiuses drops below the value *mode_switch_threshold_std*), the function
    tries the second approach.
    
    Approach 2: Iteratively increases the search radius in an interval starting from the mininum of the
    last *mode_switch_threshold_tries* radiuses up to the maximum of these values. First divides this interval
    into 10 steps, then interatively decreases the step size (*radius_increment*) by a factor of 2 and
    starts anew if necessary.
    """
    
    print("Attempting to find {} buildings around location {}.".format(str(number), str(location)))
    print("Search mode: 0")
    
    radius = 100
    
    buildings_within_radius, area_gdf = get_buildings_around_location_by_radius(buildings, location, radius)
    current_number = buildings_within_radius.shape[0]
    
    print("- Found {} buildings within radius {}".format(str(current_number), str(radius)))

    # make sure that more than 0 buildings are contained in the first circle, otherwise
    # radius estimation based on this number will fail
    while current_number == 0:
        radius *= 10
        buildings_within_radius, area_gdf = get_buildings_around_location_by_radius(buildings, location, radius)
        current_number = buildings_within_radius.shape[0]
        print("- Found {} buildings within radius {}".format(str(current_number), str(radius)))
    
    radius_history = [radius]
    density_history = []
    
    counter = 0
    search_mode = 0
    mode_switch_threshold_tries = 30
    mode_switch_threshold_std = 5
    iteration_limit = 5000
    
    while current_number != number:   
        
        if counter == iteration_limit: break
        
        # compute value used as threshold standard deviation of last radiusses to switch search mode
        # as 1% of their mean
        mode_switch_threshold_std = 0.01 * np.mean(radius_history[-mode_switch_threshold_tries:])
        
        # checking whether to switch search mode
        if (
            search_mode == 0
            and len(radius_history) >= mode_switch_threshold_tries
            and np.std(radius_history[-mode_switch_threshold_tries:]) < mode_switch_threshold_std
        ):
            search_mode = 1
            radius_range_min = min(radius_history[-mode_switch_threshold_tries:])
            radius_range_max = max(radius_history[-mode_switch_threshold_tries:])
            radius_range = radius_range_max - radius_range_min
            radius = radius_range_min
            radius_increment = radius_range / 10
            first_radius_increment = radius_increment
            increment_decrease_counter = 0
            print("Switching search mode.")
            print("Search mode: 1")
            print("Radius range min: " + str(radius_range_min))
            print("Radius range max: " + str(radius_range_max))
            print("Radius increment: " + str(radius_increment))
        
        # change search radius depending on search mode
        if search_mode == 0:

            area_numeric = math.pi * radius**2
            
            density = current_number / area_numeric
            density_history.append(density)
            mean_density = np.mean(density_history[-3:])
            
            # DEPRECATED
            # using the ratio of target number to current number, estimate the
            # size of the target area containing the target number of buildings
            # (assumes constant building density, which actually breaks the algorithm)
            # area_estimation = number / current_number * area_numeric
            
            area_estimation = number / mean_density
            
            radius = math.sqrt(area_estimation / math.pi)

        elif search_mode == 1:
            
            radius += radius_increment
            
            if radius > radius_range_max:
                # the first time the increment is decreased, it actually stays the same
                # (because the denominator becomes 0) and only the first radius value is
                # shifted by half the increment size. every later time the search radius is
                # effectively divided by 2 compared to the previous time. again, the first
                # radius value is shifted by half the increment size.
                
                radius_increment = first_radius_increment / 2**(increment_decrease_counter)
                radius = radius_range_min + radius_increment/2
                increment_decrease_counter += 1
                
                print("Decreasing radius increment.")
                print("Radius increment: " + str(radius_increment))

        # determine number of buildings within current radius
        buildings_within_radius, area_gdf = get_buildings_around_location_by_radius_sindex(buildings, location, radius)
        current_number = buildings_within_radius.shape[0]

        print("- Found {} buildings within radius {}".format(str(current_number), str(radius)))
        
        radius_history.append(radius)

        counter += 1
    
    if current_number == number:
        print("Finished successfully after " + str(counter) + " iterations.\n")
    else:
        print("Aborted search: No success after " + str(iteration_limit) + " iterations.\n")
    
    return buildings_within_radius, area_gdf, radius
        
def get_buildings_around_location_by_radius(buildings, location, radius):
    """
    This function does not use as spatial index and is therefore slow for large numbers of buildings.
    See function get_buildings_around_location_by_radius_sindex.
    """
    
    if type(location) == shapely.geometry.point.Point:
        location = gpd.GeoSeries([location])
        location.crs = buildings.crs
    else:
        raise TypeError("Location must be of type shapely.geometry.point.Point!")
    
    buffer = location.buffer(radius)
    area_gdf = gpd.GeoDataFrame({'geometry': buffer[0]}, index = [0])
    area_gdf.crs = buildings.crs
        
    buildings_within_radius = buildings.sjoin(area_gdf, how = "inner", predicate = "intersects")
    
    return buildings_within_radius, area_gdf

def get_buildings_around_location_by_radius_sindex(buildings, location, radius):
    
    if type(location) == shapely.geometry.point.Point:
        location = gpd.GeoSeries([location])
        location.crs = buildings.crs
    else:
        raise TypeError("Location must be of type shapely.geometry.point.Point!")
    
    buffer = location.buffer(radius)
    area_gdf = gpd.GeoDataFrame({'geometry': buffer[0]}, index = [0])
    area_gdf.crs = buildings.crs   
    
    buildings_within_radius_envelope = buildings.iloc[buildings.sindex.query(buffer[0])]
    
    buildings_within_radius = buildings_within_radius_envelope.sjoin(area_gdf, how = "inner", predicate = "intersects")
    
    return buildings_within_radius, area_gdf


###################################################################
# Scale down manually labeled Wartenberg dataset
###################################################################

# Functions using CV2

def rescale_and_relocate_images(filenames, dirpath_src, dirpath_dst, target_size = (256, 256), convert_labels = False, interpolation = cv2.INTER_AREA):
    n_files = len(filenames)
    for i, filename in enumerate(filenames):
        filepath_src = os.path.join(dirpath_src, filename)
        filepath_dst = os.path.join(dirpath_dst, filename)
        
        image = cv2.imread(filepath_src)
        image = cv2.resize(image, target_size, interpolation = interpolation)
        if convert_labels: image = convert_label_classes_pixelwise(image)
        cv2.imwrite(filepath_dst, image)
        
        print("Processed " + str(i+1) + " of " + str(n_files) + " files.", end = "\r")
    
    print("Processed " + str(i+1) + " of " + str(n_files) + " files.")

def convert_label_classes_pixelwise(image):
    # add value 1 to all classes / pixels
    image += 1
    # set pixels with value 18 to value 0 (background)
    image = image - (image == 18)*18
    return image