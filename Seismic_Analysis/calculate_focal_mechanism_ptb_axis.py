import numpy as np

def get_slip_vector(strike, dip, rake):
    # Calculate components of the slip vector
    n = np.cos(np.radians(rake)) * np.cos(np.radians(strike)) + np.sin(np.radians(rake)) * np.cos(np.radians(dip)) * np.sin(np.radians(strike))
    e = np.cos(np.radians(rake)) * np.sin(np.radians(strike)) - np.sin(np.radians(rake)) * np.cos(np.radians(dip)) * np.cos(np.radians(strike))
    u = np.sin(np.radians(rake)) * np.sin(np.radians(dip))

    # Combine components into a single array if only one output is needed
    slip_vector = np.array((n, e, u))
    return slip_vector

def calculate_normal_vector(dip, strike):
    # Calculate north, east, and up components
    n = -np.sin(np.radians(dip)) * np.sin(np.radians(strike))
    e = np.sin(np.radians(dip)) * np.cos(np.radians(strike))
    u = np.cos(np.radians(dip))

    # Combine components into a single array if only one output is needed
    normal_vector = np.array((n, e, u))
    return normal_vector

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    return azimuth, elevation, r

def convert_to_vpa(component):
    n, e, u = component
    # In numpy, the azimuth is the angle in the xy-plane counted in radians from the positive x-axis.
    azimuth, plunge, value = cart2sph(e, n, u)
    azimuth = 90 - azimuth
    plunge = - plunge
    vpa_vector = np.array((azimuth, value, plunge))
    return vpa_vector

def sdr2ptb(strike, dip, rake):
  s = get_slip_vector(strike, dip, rake)
  n = calculate_normal_vector(dip, strike)
  t = convert_to_vpa(n + s)
  p = convert_to_vpa(n - s)
  return p, t
