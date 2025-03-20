import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Geod
from shapely.geometry import Point, LineString, Polygon
from math import acos, sin, cos, radians, pi
from sklearn.metrics import r2_score


def calculate_distance(x1, y1, x2, y2):
    """
    input: lon1, lat1, lon2, lat2
    output: distance
    """ 
    line_string = LineString([Point(x1, y1), Point(x2, y2)])
    geod = Geod(ellps="WGS84")
    d = geod.geometry_length(line_string)
    return d


def top(x1, y1, x2, y2, x_test, y_test):
    """
    input: lat1, lon1, lat2, lon2, lat_test, lon_test
    output: True(top), False(bottom)
    """
    a = (y1-y2) / (x1-x2)
    b = (x1*y2 - x2*y1) / (x1-x2)
    y_online = a*x_test + b
    if y_test > y_online :
        return True
    else:
        return False


def dat2csv(input_filename, ouput_filename, lat_epicenter, lon_epicenter):
    """
    To output csv file in the current folder
    """
    f_in = open(input_filename, 'r')
    f_out = open(ouput_filename, 'w')
    lines = f_in.readlines()
    f_in.close()

    for line in lines:
        year = int(line[0:4])
        month = int(line[4:6])
        day = int(line[6:8])
        lat = int(line[18:20]) + float(line[20:25])/60
        lon = int(line[25:28]) + float(line[28:33])/60
        depth = float(line[33:39])
        mag = float(line[39:43])
        dist = np.round(calculate_distance(lat, lon, lat_epicenter, lon_epicenter) ,3)

        f_out.write("{:5d}\t{:5d}\t{:5d}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(year, month, day, lat, lon, depth, mag, dist))
    f_out.close()


def count_in_magnitude(m_group, m_interval=1):
    """
    input: list(magnitude)
    output: dict(magnitude, number)
    instruction: 
    1. Use magnitude_interval = 1 -> floor magnitude
    2. Use magnitude_interval = 0.1 -> magnitude
    """
    def inside(m1, m2, m):
        if m1<= m < m2:
            return True
        else:
            return False

    if m_interval == 1:
        m_group = np.floor(m_group)
    m_indexs = set(m_group)
    number_group = np.zeros(len(m_indexs))

    for i, m_index in enumerate(m_indexs):
        mask = [inside(m_index, m_index+m_interval, m) for m in m_group]
        number = len(m_group[mask])
        number_group[i] = number
    return dict(zip(m_indexs, number_group.astype(int)))


def count_in_magnitude_cumulative(m_group, m_interval=0.1):
    """
    input: list(magnitude), magnitude_interval
    output: dict(magnitude, number)
    instruction: 
    1. Use magnitude_interval = 1 -> floor magnitude
    2. Use magnitude_interval = 0.1 -> magnitude
    """
    if m_interval == 1:
        m_group = np.floor(m_group)  
    m_indexs = set(m_group)
    tmp =  dict((m_index, m_group.count(m_index)) for m_index in m_indexs)
    m_numbers_sort = sorted(tmp.items())
    m_numbers = dict(m_numbers_sort)
    return m_numbers


def get_area(*border_point):
    """
    input: border_points
    output: poly
    """
    border_points = list(border_point)
    poly = Polygon(border_points)
    return poly


def get_area_point_mask(area_poly, x, y):
    """
    input: poly, lon, lat
    output: mask
    """
    mask = np.zeros(len(x))
    for i, point in enumerate(zip(x, y)):
        mask[i] = area_poly.contains(Point(point))
    return mask


def get_GRlaw(year, rate):
    """
    input: year, rate
    output: poly1d, r2
    """
    rate_log = np.log10(rate)
    coefficients = np.polyfit(year, rate_log,1)
    p = np.poly1d(coefficients)
    r2 = r2_score(rate_log, p(year))
    return p, r2



