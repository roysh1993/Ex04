import numpy as np
import pandas as pd
# from sklearn import
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors.kneighbors


def preprocessing_step(data):
    """
    This step preproccesses the data before the training step
    :param data:
    :return: data after preprocessing
    """
    # change categorical data to 1 hot @TODO find something better (maybe)
    # @TODO maybe select features according to correlation
    data = pd.get_dummies(data)

    Y = data["Y"]
    T = data["T"]
    X = data.drop(["Y","T"],axis= 1)
    column_names = X.columns
    X_norm = StandardScaler().fit_transform(X)
    new_df = pd.DataFrame(X_norm,columns=column_names)
    new_df["T"] = T
    new_df["Y"] = Y
    return new_df


def learning_step(X,y,neighbors = 5, distance = None):

    KN_model = NearestNeighbors(n_neighbors= neighbors)
    KN_model.fit(X,y)
    return KN_model


def find_knn(model,points, k = 5):
    """
    :param model: trained model
    :param points: the poins who we want to query the knn
    :param k: number of neighbors
    :return: a tuple with the knn to every point and the distance of every neighbor
    """

    return model.kneighbors(points,k)


def estimate_y_by_knn(distance_arr,neighbors_indices_arr,y_arr ):
    """

    :param distance_arr: a 2d array where for every sample x we have its distance from each of the k neighbors
    :param neighbors_indices_arr: a 2d array where for every sample x we the k indices of the nearest neighbors
    :param y_arr: the y value of all the neighbors
    :return: a prediction of the y value for every sample by a weighted average of its neighbors
    """
    # get y value from indices
    neigh_shape = neighbors_indices_arr.shape
    y_values = y_arr[neighbors_indices_arr.flatten()].reshape(neigh_shape)
    weighted_average = (1/distance_arr) / ((1/distance_arr).sum(axis = 1)[:,None])
    y_hat = (weighted_average*y_values).sum(axis = 1)

    return y_hat

def calculate_ITE(y, y_hat,t):
    if t == 1:
        ite = y - y_hat
    else:
        ite = y_hat - y
    return ite

def calculate_ATT(ite_vec):

    return np.mean(ite_vec)


def main(data):
    k = 5
    processed_data = preprocessing_step(data)
    T_0 = processed_data[processed_data["T"] == 0].reset_index()
    Y_0 = T_0["Y"].values
    X_0 = T_0.drop(["T"],axis= 1 )
    T_1 = processed_data[processed_data["T"] == 1].reset_index()
    Y_1 = T_1["Y"].values
    X_1 = T_1.drop(["T"],axis= 1)
    model = learning_step(X_0,Y_0,neighbors= k)
    distance,neighbors_indices = find_knn(model,X_1,k=k)
    y_hat = estimate_y_by_knn(distance,neighbors_indices,Y_0)
    ITE = calculate_ITE(Y_1,y_hat,1)
    ATT = calculate_ATT(ITE)
    return ATT





if __name__ == '__main__':
    print("HELLO")
    data = pd.read_csv("data/data1.csv",index_col = 0)
    # data = preprocessing_step(data)
    # T_0 = data[data["T"] == 0].reset_index()
    # Y_0 = T_0["Y"].values
    # X_0 = T_0.drop(["T"],axis= 1 )
    # T_1 = data[data["T"] == 1].reset_index()
    # Y_1 = T_1["Y"].values
    # X_1 = T_1.drop(["T"],axis= 1 )
    print(main(data))







