import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from data import notaire_sales


'''
    todo:
        * modeliser le tout avec des fonctions mathematiques forcees en simplifiant les dependences
            et faire un RN qui apprend, par exemple price_m2 = f(surface) * f(etage) * f(age)
        * faire features sur age du logement et prix du m2
        * trouver moyen de mettre SOM dessus
        
        date of construction can be nan
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


def run(city):
    # price_old_3_pieces_by_age(city)
    # compare_old_and_new(city)
    linear_reg(city, dropna=True, old_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default="sannois")
    args = parser.parse_args()
    run(args.city)
