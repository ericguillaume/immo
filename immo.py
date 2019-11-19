import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix, parallel_coordinates, andrews_curves
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from data import notaire_sales


'''
    todo:
        * modeliser le tout avec des fonctions mathematiques forcees en simplifiant les dependences
            et faire un RN qui apprend, par exemple price_m2 = f(surface) * f(etage) * f(age)
        * faire features sur age du logement et prix du m2
        * trouver moyen de mettre SOM dessus
        
        date of construction can be nan
'''

'''
    analysis   
    
    correlations for 3 pieces
                     price       age      cave    garage       rdc   surface  price_per_m2
price         1.000000 -0.306803 -0.089401  0.262358  0.056982  0.305977      0.721050
age          -0.306803  1.000000  0.387971 -0.386826  0.175740 -0.498756      0.126960
cave         -0.089401  0.387971  1.000000 -0.006432 -0.020026 -0.161926      0.015464
garage        0.262358 -0.386826 -0.006432  1.000000  0.166053  0.438814     -0.055991
rdc           0.056982  0.175740 -0.020026  0.166053  1.000000 -0.248432      0.243387
surface       0.305977 -0.498756 -0.161926  0.438814 -0.248432  1.000000     -0.420341
price_per_m2  0.721050  0.126960  0.015464 -0.055991  0.243387 -0.420341      1.000000
    correlations with price for 3 pieces
price           1.000000
price_per_m2    0.721050
surface         0.305977
garage          0.262358
rdc             0.056982
cave           -0.089401
age            -0.306803


'''

# additionnal features # "rdc", "garage", "cave"]
LINEAR_REG_CITY_FEATURES = {
    "default": ["garage"],
    "issy": ["garage", "rdc", "cave"],
}

LINEAR_REG_CITY_KWARGS = {
    "default": {"age_min": 10.0, "age_max": 20.0}
}


def linear_reg(city, **kwargs):
    kwargs.update(LINEAR_REG_CITY_KWARGS.get(city, LINEAR_REG_CITY_KWARGS["default"]))

    data = notaire_sales(city, **kwargs)
    # print(data)

    features = ["surface_under_30", "surface_between_30_and_45", "surface_above_45"]
    features += LINEAR_REG_CITY_FEATURES.get(city, LINEAR_REG_CITY_FEATURES["default"])

    x = []
    y = []
    for index, row in data.iterrows():
        x.append([row[feat] for feat in features])
        y.append(row["price"])
    x = np.array(x)
    y = np.array(y)

    reg = LinearRegression(fit_intercept=False).fit(x, y)
    print("{} linear_reg       --".format(city, kwargs))
    print("features = {}".format(features))
    print("reg.coef_ = {}".format(reg.coef_))


def histogram_old_prices_for_all_pieces(data):
    data = data[(data["age"] >= 10) & (data["age"] <= 40)]
    for pieces_count in [1, 2, 3, 4, 5]:
        data_old_per_piece = data[(data["new"] == 0) & (data["pieces"] == pieces_count)]
        data_old_per_piece.hist("price_per_m2", bins=40)
        plt.show()


def old_2_pieces_price_sorted_by_date(data):
    data = data[(data["pieces"] == 2) & (data["new"] == 0)]
    print(data.sort_values(by="date_construction"))


# compares globally old and new real estate
def compare_old_and_new(city):
    # pandas.DataFrame.hist() aller voir code pour comprendre pk affiche pas clairement
    old_data = notaire_sales(city, dropna=True, old_only=True, split_surface=False)
    print("{} histogram_features_for_old_and_new".format(city))
    print(old_data.groupby("pieces").mean())

    new_data = notaire_sales(city, dropna=True, new_only=True, split_surface=False)
    print("{} histogram_features_for_new".format(city))
    print(new_data.groupby("pieces").mean())

    # display histograms prices
    fig, (old_ax, new_ax) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)

    old_ax.set_title("{}: histogram old prices".format(city))
    old_data.hist(["price"], bins=30, ax=old_ax, label="ttt")

    new_ax.set_title("{}: histogram new prices".format(city))
    new_data.hist(["price"], bins=30, ax=new_ax)
    plt.show()


def price_old_3_pieces_by_age(city):
    print("{} prices of 3 pieces by age".format(city))
    data = notaire_sales(city, dropna=True, old_only=True, pieces_count=3)
    data.groupby("date_construction")[["price_per_m2"]].mean().plot()  # mettre dans plot
    plt.show()


def ds_basic_visus(sales):
    print(sales.head())
    print(sales.info())

    price_correlations_features = sales[["price", "price_per_m2", "surface", "age", "garage", "cave", "rdc"]]
    price_correlations = price_correlations_features.corr()
    print(price_correlations)
    print()
    price_correlations = price_correlations["price"]
    print(price_correlations.sort_values(ascending=False))
    print()

    # plt.style.use('classic')  # 'ggplot'
    scatter_matrix(price_correlations_features)
    plt.show()


def ds_multi_visus(sales):
    '''
        andrews_curves works fine with small number of normalized features, we see that we are missing some important
        features for more precision
    '''
    features_names = ["price", "age", "cave", "garage"]  # , "rdc", "surface", "price_per_m2"]   AJOUTER surface??????
    price_correlations_features = sales[features_names]  # ENLEVER CORRELATIO NDE NAME

    price_correlations_features_scaled = MinMaxScaler().fit_transform(price_correlations_features.values)
    price_correlations_features_scaled = pd.DataFrame(price_correlations_features_scaled, columns=features_names)

    indexes_bins_count = 4
    indexes = np.array(list(range(1, indexes_bins_count, 1))) / float(indexes_bins_count)
    price_correlations_features_scaled["price"] = np.searchsorted(indexes,
                                                                  price_correlations_features_scaled["price"].values)

    parallel_coordinates(price_correlations_features_scaled, "price", colormap=plt.get_cmap("jet"))
    plt.show()
    andrews_curves(price_correlations_features_scaled, "price", colormap=plt.get_cmap("jet"))  # jet is cool red => blue
    plt.show()


def ds_visu_3d(sales):
    sales_scaled = MinMaxScaler().fit_transform(sales.values)
    sales_scaled = pd.DataFrame(sales_scaled, columns=list(sales.columns))

    indexes_bins_count = 4
    indexes = np.array(list(range(1, indexes_bins_count, 1))) / float(indexes_bins_count)
    print("indexes = {}".format(indexes))
    # sales_scaled["price"] = np.searchsorted(indexes, sales_scaled["price"].values)

    fig = plt.figure(figsize=(12, 12))
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')

    x = sales_scaled["surface"].values
    y = sales_scaled["rdc"].values
    z = sales_scaled["price"].values * 20
    ax.set_xlabel("surface")
    ax.set_ylabel("rdc")
    ax.set_zlabel("price")
    print("x[0:10] = {}".format(x[0:10]))
    ax.scatter(x, y, z, s=z)
    plt.tight_layout()
    plt.show()


# etudier pieces  vs prix pour same size
# voir si surface price per m2 influe bcp sur le prix ou non
def ds_visu(city):
    sales = notaire_sales(city, dropna=True, old_only=True, pieces_count=3)

    # ds_basic_visus(sales)
    # ds_multi_visus(sales)
    ds_visu_3d(sales)


def run(city):
    # price_old_3_pieces_by_age(city)
    # compare_old_and_new(city)
    # linear_reg(city, dropna=True, old_only=True)

    ds_visu(city)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default="sannois")
    args = parser.parse_args()
    run(args.city)
