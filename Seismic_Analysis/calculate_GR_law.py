import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Geod
from shapely.geometry import Point, Polygon
from sklearn.metrics import r2_score

year_group = np.arange(1980, 2022, 1)
area = 'S04'


def get_area(*border_points):
    """
    input: border_points
    output: poly, area
    """
    geod = Geod(ellps="WGS84")
    poly = Polygon(border_points)
    area = geod.geometry_area_perimeter(poly)[0]
    return poly, area


def get_area_point_mask(area_poly, x, y):
    """
    input: poly, lon, lat
    output: mask
    """
    mask = np.zeros(len(x))
    for i, point in enumerate(zip(x, y)):
        mask[i] = area_poly.contains(Point(point))
    return mask


def count_in_magnitude(m_group, m_interval=1):
    """
    input: list(magnitude)
    output: dict(magnitude, number)
    instruction:
    1. Use magnitude_interval = 1 -> floor magnitude
    2. Use magnitude_interval = 0.1 -> magnitude
    """
    def inside(m1, m2, m):
        return m1 <= m < m2

    if m_interval == 1:
        m_group = np.floor(m_group)
    m_indices = set(m_group)
    number_group = np.zeros(len(m_indices))

    for i, m_index in enumerate(m_indices):
        mask = [inside(m_index, m_index + m_interval, m) for m in m_group]
        number = len(m_group[mask])
        number_group[i] = number
    return dict(zip(m_indices, number_group.astype(int)))


def get_GR_law(m, rate):
    """
    input: magnitude, rate
    output: poly1d, r2
    """
    rate_log = np.log10(rate)
    coefficients = np.polyfit(m, rate_log, 1)
    p = np.poly1d(coefficients)
    r2 = r2_score(rate_log, p(m))
    return p, r2


def plot_map(x, y, title):
    plt.figure()
    plt.scatter(x, y, s=0.1)
    plt.title(title)
    plt.savefig('area_map.jpg')


def get_annual_rate(df, mag_group):
    plt.figure()
    global year_group
    for mag in mag_group:
        tmp = df[mag - 1 <= df['mag']]
        Y = []
        for year in year_group:
            df_year = tmp[tmp['year'] == year]
            N = len(df_year)
            Y.append(N)
        plt.scatter(year_group, Y, label=f'{mag}M+')
    plt.yscale('log')
    plt.ylabel('Annual Rate (N)')
    plt.xlabel('Year')
    plt.legend()
    plt.grid()
    plt.savefig("annual_rate_good.jpg")
    plt.show()


def plot_GR_law(mag_group, annual_rate_group, p, title):
    plt.figure()
    index = range(1, 5, 1)
    M_predict = p(index)
    a = p[0]
    b = -p[1]
    text_regression = f'log(N) = {a:.2f} - {b:.2f}M'
    annual_rate_group = np.log10(annual_rate_group)
    fig, ax = plt.subplots()
    ax.scatter(mag_group, annual_rate_group, color='r')
    ax.plot(index, M_predict, color='b')
    ax.text(0.05, 0.1, text_regression, ha='left', va='center', transform=ax.transAxes, fontsize=13)
    ax.set_title(title)
    ax.set_xlabel("M")
    ax.set_ylabel("log(N)")
    plt.savefig("GR-law.jpg")


if __name__ == '__main__':
    nuclear4 = (121.925, 25.039)
    df = pd.read_csv("tmp.csv", sep='\t', names=['year', 'month', 'day', 'lat', 'lon', 'dep', 'mag', 'dist'], header=None)
    df = df[df['year'] >= year_group[0]]

    x = df['lon']
    y = df['lat']
    if area == 'S04':
        # ============= S04 ===================
        A = (121.182568, 25.157801)
        B = (121.871427, 25.503781)
        C = (122.206815, 25.107782)
        D = (121.867727, 24.933576)
        E = (121.860178, 24.926317)
        F = (121.637366, 24.807973)
        G = (121.416511, 24.687554)
        H = (121.414364, 24.689529)
        I = (121.182568, 25.157801)
        area_poly, area_surface = get_area(A, B, C, D, E, F, G, H, I)
    elif area == 'S14A':
        # ============ S14A ====================
        A = (121.416511, 24.687554)
        B = (121.637366, 24.807973)
        C = (121.860178, 24.926317)
        D = (121.867727, 24.933576)
        E = (121.90067, 24.496784)
        F = (121.587151, 24.375046)
        G = (121.416511, 24.687554)
        area_poly, area_surface = get_area(A, B, C, D, E, F, G)
    elif area == 'S14B':
        # ========= S14B ======================
        A = (121.867727, 24.933576)
        B = (122.206815, 25.107782)
        C = (122.459702, 24.893829)
        D = (122.225845, 24.249776)
        E = (121.90067, 24.496784)
        F = (121.867727, 24.933576)
        area_poly, area_surface = get_area(A, B, C, D, E, F)

    mask = get_area_point_mask(area_poly, x, y)
    df_area = pd.DataFrame()
    for i, m in enumerate(mask):
        if m == 1.0:
            df_area = df_area.append(df.iloc[i])

    m_numbers = count_in_magnitude(list(df_area['mag']))
    total_years = len(year_group)
    mag_group = np.array(list(m_numbers.keys()))
    annual_rate_group = np.array(list(m_numbers.values())) / total_years
    p, R2 = get_GR_law(mag_group[2:5], annual_rate_group[2:5])

    title = f'GR-law for {area}'

    get_annual_rate(df_area, mag_group)
    plot_GR_law(mag_group[2:5], annual_rate_group[2:5], p, title)

    a = p[0]
    b = -p[1]

    M_predict = p(np.arange(3, 9, 1))
    annual_rate_group = 10 ** M_predict
    prob = 1 - np.exp(-1 * annual_rate_group * 10000)
    print(prob)
