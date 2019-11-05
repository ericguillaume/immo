import csv
import re

import numpy as np


def surface_under_30(surface):
    return surface_between(surface, 0, 30)


def surface_between_30_and_45(surface):
    return surface_between(surface, 30, 45)


def surface_above_45(surface):
    return surface_between(surface, 45, 1e3)


def surface_between(surface, lower_bound, upper_bound):
    surface = surface - lower_bound
    return max(min(surface, upper_bound - lower_bound), 0)


def load_data():
    rows = []
    with open("data/notaires/Ventes-de--Sannois--95-03112019.csv") as f:
        reader = csv.reader(f, delimiter=";")

        for idx, row in enumerate(reader):
            if idx <= 2 or len(row[0]) == 0:
                continue
            try:
                row = {"price": float(numeric_only(row[0])),
                       "surface": float(numeric_only(row[1])),
                       "pieces": int(numeric_only(row[2])),
                       "new": 1 if row[3] == "2011 / 2020" else 0,
                       "date_construction": csv_string_to_age(row[3]),
                       "garage": 1 if row[4] == "Oui" else 0,
                       "rdc": 1 if int(numeric_only(row[7])) == 0 else 0,
                       "cave": 1 if row[8] == "Oui" else 0}  # cando ajouter les balcons exclue les -
                if row["surface"] == 0.0:
                    continue  # trouver toutes ces extremess
                row["age"] = 2020 - row["date_construction"]
            except ValueError as e:
                # print(row)
                # print(e)
                continue

            rows.append(row)
    print("len(rows) = {}".format(len(rows)))
    exit()
    return rows


def alphanum(string):
    pattern = re.compile(r"[\W_]+", re.ASCII)
    return pattern.sub("", string)


def numeric_only(string):
    pattern = re.compile(r"[a-z\W_]+", re.ASCII)
    return pattern.sub("", string)


def csv_string_to_age(value):
    # if value == "-" return nan et exclue
    if value == "-":
        return np.nan
    if value == "2011 / 2020":
        return 2016
    elif value == "2001 / 2010":
        return 2006
    elif value == "1992 / 2000":
        return 1996
    elif value == "1981 / 1991":
        return 1986
    elif value == "1970 / 1980":
        return 1975
    elif value == "1948 / 1969":
        return 1958
    elif value == "1914 / 1947":
        return 1930
    elif value == "1850 / 1913":
        return 1881
    elif value == "Avant 1850":
        return 1830
    else:
        print("csv_string_to_age dont support {}".format(value))
        exit()



# def test_intercept():
#     x = np.array([[0.0, 1.0], [2.0, 0.0]]) # un chacun plus un # plus un static
#     y = np.array([2.0, 3.0])
#     reg = LinearRegression(fit_intercept=True).fit(x, y)
#     print("reg.coef_ = {}".format(reg.coef_))
#     print("reg.intercept_ = {}".format(reg.intercept_))


# histogram price old houses
#     data = data[data["new"] == 0]
#     data.hist(column="price_per_m2", bins=100, figsize=(10, 8))
#     plt.show()

# histogram dates of constructions
#     data = data[data["new"] == 0]
#     data.hist(column="date_construction", bins=100, figsize=(10, 8))
#     plt.show()