import csv

import numpy as np
import pandas as pd

from tools import numeric_only, surface_under_30, surface_between_30_and_45, surface_above_45


NOTAIRE_SALES_PER_CITY_DICT = {}


# manages RAM caching
def notaire_sales(city,
                  dropna=True,
                  old_only=False,
                  new_only=False,
                  age_min=None,
                  age_max=None,
                  pieces_count=None,
                  split_surface=True):
    data = NOTAIRE_SALES_PER_CITY_DICT.get(city, None)
    if data is None:
        data = _do_load_notaires_sales(city)
        NOTAIRE_SALES_PER_CITY_DICT[city] = data

    if dropna:
        data = data.dropna()
    if old_only:
        data = data[data["new"] == 0]
    if new_only:
        data = data[data["new"] == 1]
    if age_min:
        data = data[data["age"] >= age_min]
    if age_max:
        data = data[data["age"] >= age_max]
    if pieces_count:
        data = data[data["pieces"] == pieces_count]

    if split_surface:
        data["surface_under_30"] = data["surface"].apply(surface_under_30)
        data["surface_between_30_and_45"] = data["surface"].apply(surface_between_30_and_45)
        data["surface_above_45"] = data["surface"].apply(surface_above_45)
    return data


def _do_load_notaires_sales(city):
    rows = []
    with open(_notaires_sales_filepath(city)) as f:
        reader = csv.reader(f, delimiter=";")

        for idx, row in enumerate(reader):
            if idx <= 2 or len(row[0]) == 0:
                continue
            try:
                row = {"price": float(numeric_only(row[0])),
                       "surface": float(numeric_only(row[1])),
                       "pieces": int(numeric_only(row[2])),
                       "new": 1 if row[3] == "2011 / 2020" else 0,
                       "date_construction": _csv_string_to_age(row[3]),
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
    data = pd.DataFrame(rows)

    data["price_per_m2"] = data["price"] / data["surface"]  # faire average de age par categorie.... ajouter age
    return data


def _csv_string_to_age(value):
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


def _notaires_sales_filepath(city):
    if city == "issy":
        return "data/notaires/Ventes-d--Issy-les-Moulineaux--92-03112019.csv"
    elif city == "sannois":
        return "data/notaires/Ventes-de--Sannois--95-03112019.csv"
    elif city == "ermont":
        return "data/notaires/Ventes-d--Ermont--95-03112019.csv"
