import re


def surface_under_30(surface):
    return surface_between(surface, 0, 30)


def surface_between_30_and_45(surface):
    return surface_between(surface, 30, 45)


def surface_above_45(surface):
    return surface_between(surface, 45, 1e3)


def surface_between(surface, lower_bound, upper_bound):
    surface = surface - lower_bound
    return max(min(surface, upper_bound - lower_bound), 0)


def alphanum(string):
    pattern = re.compile(r"[\W_]+", re.ASCII)
    return pattern.sub("", string)


def numeric_only(string):
    pattern = re.compile(r"[a-z\W_]+", re.ASCII)
    return pattern.sub("", string)



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