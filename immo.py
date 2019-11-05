import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from tools import load_data, surface_under_30, surface_between_30_and_45, surface_above_45


'''
    todo:
        * faire features sur age du logement et prix du m2
        * trouver moyen de mettre SOM dessus
        
        date of construction can be nan
'''


def linear_reg(data):
    data = data.dropna()
    data = data[data["new"] == 0]
    # data = data[data["age"] <= 2020 - 1996]
    data = data[(data["age"] >= 10) & (data["age"] <= 40)]
    print(data)

    # test
    x = []
    y = []
    features = ["surface_under_30",
                "surface_between_30_and_45",
                "surface_above_45",
                "garage"]
    # "rdc",
    # "cave"]  # "rdc", "garage", "cave"]
    for index, row in data.iterrows():
        new_row = [row[feat] for feat in features]
        x.append(new_row)
        y.append(row["price"])
    x = np.array(x)
    y = np.array(y)

    reg = LinearRegression(fit_intercept=False).fit(x, y)
    print("features = {}".format(features))
    print("reg.coef_ = {}".format(reg.coef_))
    print("reg.intercept_ = {}".format(reg.intercept_))


def histogram_old_prices_for_all_pieces(data):
    data = data[(data["age"] >= 10) & (data["age"] <= 40)]
    for pieces_count in [1, 2, 3, 4, 5]:
        data_old_per_piece = data[(data["new"] == 0) & (data["pieces"] == pieces_count)]
        data_old_per_piece.hist("price_per_m2", bins=40)
        plt.show()


def price_3_pieces_by_age(data):
    data = data[data["pieces"] == 3]
    data.groupby("date_construction")[["price_per_m2"]].mean().plot()
    plt.show()


def old_2_pieces_price_sorted_by_date(data):
    data = data[(data["pieces"] == 2) & (data["new"] == 0)]
    print(data.sort_values(by="date_construction"))


def run():
    data = load_data()
    data = pd.DataFrame(data)
    data["price_per_m2"] = data["price"] / data["surface"]  # faire average de age par categorie.... ajouter age
    data["surface_under_30"] = data["surface"].apply(surface_under_30)
    data["surface_between_30_and_45"] = data["surface"].apply(surface_between_30_and_45)
    data["surface_above_45"] = data["surface"].apply(surface_above_45)
    data = data.dropna()

    # data_old = data[data["new"] == 0]
    # print(data_old.groupby("pieces").mean())
    #
    # data_new = data[data["new"] == 1]
    # print(data_new.groupby("pieces").mean())

    # data.hist(["price"], bins=30)
    # plt.show()

    # data.hist("surface", bins=30)
    # plt.show()

    # data = data.drop("surface", axis=1)

    linear_reg(data)


if __name__ == "__main__":
    run()
