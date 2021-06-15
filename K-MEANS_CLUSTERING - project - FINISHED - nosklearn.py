# K-MEANS CLUSTERING ALGORITHM WITHOUT USING SKLEARN !!!

# FINISHED !!!


import numpy as np
import pandas as pd

# custom made, made it myself, just for plot analysis and customizing dataset, can be found in github plot analysis and dataset customizing
from DATASET_CUSTOMIZING.DATASET_READING_CUSTOMIZING_INPUT import make_dataset
from PLOT_ANALYSIS.PLOT_ANALYSIS_INPUT import make_analysis

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')



def K_Means():
    global ask_IVs

    # read or make some customization for the dataset
    df = make_dataset()

    # make plot analysis if user wishes to do so
    make_analysis(df)

    while True:

        # check for any input error
        try:
            print('')
            print('THESE ARE YOUR COLUMNS NAMES:', list(df.columns))
            print('')
            # ask for column(s) which will be all X values, which will be our features
            ask_IVs = input('Type all column names in the dataset, you want for X (Independent variables / features) (split with comma + space): ').split(', ')

            # if that column that was input does not occur in the dataset, raise an error
            for col_X in ask_IVs:
                if col_X not in list(df.columns):
                    raise Exception


            # ask if want to have default parameters or want to specify them
            print('')
            ask_specify_default = input('Wanna specify the n_clusters(k)/maximum_iterations/tolerance or want to leave it all in default (default k=2; max_iter=300; tol=0.001) --> Specify/Default: ').upper()

            if ask_specify_default == 'SPECIFY':

                # ask for n_clusters and assign it to k
                print('')
                ask_parameters = input('OK, give me the specifications in this order !!! --> k/max_iter/tol (split with slash): ').split('/')

                if len(ask_parameters) != 3:
                    raise Exception

                k = int(ask_parameters[0])
                max_iter = int(ask_parameters[1])
                tol = float(ask_parameters[2])


            # default n_cluster is equal to 3
            elif ask_specify_default == 'DEFAULT':
                k = 2
                max_iter = 300
                tol = 0.001

            else:
                print('')
                print('Wrong input Specify/Default, try again !!!')
                raise Exception


        except ValueError:
            print('')
            print('Oops, something went wrong, an input must be number, try again !!!')

        except Exception:
            print('')
            print('''Oops, something wrong with your input, or at least one the columns you have input does not appear in the dataset
or you have input more than one column, or the label column is not categorized (True/False --> 1/0)
or the number of parameters does not match with your input, try again !!!''')

        # if everything seems to be working and no error has been raised, break the loop
        else:
            break

    # an array of feature sets
    X = np.array(df[ask_IVs])


    # ask if user wants to standardize the data
    print('')
    ask_scale = input('Do you also wanna scale your feature sets? Yes/No: ').upper()

    if ask_scale == 'YES':

        # STANDARDIZATION --> X_scaled = (X - X_mean) / X_std
        x_means = np.mean(X, axis=0)
        x_stds = np.std(X, axis=0)
        X = (X - x_means) / x_stds

    else:
        pass

    # starting random centroids, meaning first k centroids that are in X
    centroids = {}
    for i_centroid in range(k):
        centroids[i_centroid] = X[i_centroid]


    # go in range of maximum iterations
    for i in range(max_iter):

        # this is where classification is going to occur, determination of which point belongs to what centroid (key = centroid, value = point)
        classifications = {}

        # each centroid is called like --> 1st_cen = 0, 2nd_cen = 1, etc...
        # so to centroid will have its list for later classification
        for i_centroid in range(k):
            classifications[i_centroid] = []


        # go through every feature_set in X
        for feature_set_X in X:

            # calculate the distance of the feature_set for EVERY centroid and make a list of those distances
            distances = [np.sqrt(np.sum((feature_set_X - centroids[centroid]) ** 2)) for centroid in centroids]

            # so every centroid is called by index (by range), we take the minimum distance of that list and take its index where it occurs in that distances list
            # index will be equal to the centroid
            classification = distances.index(min(distances))
            # then add that feature_set to its nearest centroid
            classifications[classification].append(feature_set_X)


        # make a copy of previous centroids, because in the next for loop we will overwrite the centroids by taking the mean of feature_sets of each centroid
        previous_centroids = dict(centroids)

        # go for every centroid name
        for classification in classifications:
            # new centroid is equal to the mean value of all the coordinates of that specific centroid
            new_centroid = np.mean(classifications[classification], axis=0)
            # assign that new centroid to centroid dictionary
            centroids[classification] = new_centroid


        # optimized is True
        optimized = True

        # go through every centroid
        for centroid in centroids:

            # previous centroid is equal to the centroid which is previous_entroid dictionary
            prev_centroid = previous_centroids[centroid]
            # this current changed centroid (mean of its feature sets) is now in centroid dictionary, so take that centroid
            current_centroid = centroids[centroid]

            # let's compare the 2 centroids (previous, new)
            # to make a comparison sum(( (new - old) / old * 100 ))
            # if that's gonna be more than our tolerance, optimized is False
            # if the centroid are moving significantly then don;t stop the iterations, but if it's moving very slowly, and it's not likely to change anything, then optimized stays true
            if np.sum((current_centroid - prev_centroid) / prev_centroid * 100) > tol:
                optimized = False

        # if optimized stays True, when the centroid haven't moved significantly (moved less than our tolerance) then break the loop
        if optimized:
            break



    # this will be the array of all feature sets in X but with their additional classification/cluster they belong in, that's why n elements + 1, +1 is the cluster
    X_classifications = np.array([[x for x in range(len(X[1, :]) + 1)]])


    # go through each cluster_name, or classification key in classification dictionary
    for classification_key in classifications:

        # go through every feature set in that specific centroid, those feature sets belon in
        for feature_set_x in classifications[classification_key]:
            # make a 2d array of that feature set and its classification --> [[10 10 0]] (0 --> 1st cluster) (10, 10 --> feature set)
            feature_set_class = np.array(list(feature_set_x) + [classification_key]).reshape(1, -1)

            # add that feature set with its class into that numpy array but in axis=0, vertically, by columns, not next to each other (by rows)
            X_classifications = np.append(X_classifications, feature_set_class, axis=0)

    # delete that first element which was set to be [0, 0, 0], and it will also return the array with that deleted array
    X_classifications = np.delete(X_classifications, 0, 0)



    while True:

        try:
            print('')
            # ask for one single column which will be our label Y, for making accuracy check (DV)
            ask_DV = input('Type column name in the dataset you want for label Y, must be categorical column, and must have the same amount of unique values as k (n_clusters) !!!: ').split(', ')

            # if there will be more than one column for label input, raise an error
            if len(ask_DV) != 1:
                raise Exception


            for col_y in ask_DV:

                # and again if that label column does not occur in the dataset, raise an error
                if col_y not in list(df.columns):
                    raise Exception

                # if that column has more or less unique values than k (n_clusters) raise an error
                if len(df[col_y].unique()) != k:
                    raise Exception


        except Exception:
            print('')
            print('Oops, something went wrong, must be only ONE label, must be in that dataset, and must have the same amount of unique values as k (n_clusters), try again !!!')

        # if everything occurs without any error, break the loop
        else:
            break



    # an array of label, where we have true data not predicted by clustering
    y = np.array(df[''.join(ask_DV)])

    # horizontal stack real data for X and their real label y, label y is the last
    X_classifications_with_real_y = np.hstack([X, y.reshape((len(y), 1))])



    # !!! ----------------------------------------- ACCURACY CALCULATION -------------------------------------------- !!!
    correct = 0

    # go through every array in X_classifications, where we have all data for X and the last element is clustering prediction for centroid
    for X_centroid_arr in X_classifications:

        # go through every array in X_classifications_with_real_y, where we have all data for X and the last element is the true value for y
        for X_arr_real_y in X_classifications_with_real_y:

            # find the same data in both array, but without any predictions elements (without last element)
            if all(X_centroid_arr[:len(X_arr_real_y) - 1] == X_arr_real_y[:len(X_arr_real_y) - 1]):

                # if those arrays are the same, check if they have the same label/prediction --> cluster_pred to true_y
                if X_centroid_arr[-1] == X_arr_real_y[-1]:
                    # if the prediction is the same as the true label of that data X, correct + 1
                    correct += 1

    # accuracy in percents, all correct ones divided by the number amount of all X data, and in percentage * 100
    accuracy = np.round(correct / len(X) * 100, 2)

    print('')
    print('!!!')
    print('ACCURACY:', str(accuracy) + '%')
    print('!!!')
    # !!! ----------------------------------------- ACCURACY CALCULATION -------------------------------------------- !!!



    # !!! ------------------------------------------- CENTROID COORDINATES ------------------------------------------ !!!
    # final cluster centers coordinates in 2D numpy array
    cluster_centers = np.array([centroids[centroid] for centroid in centroids])
    # !!! ------------------------------------------- CENTROID COORDINATES ------------------------------------------ !!!



    # !!! ----------------------------------- PLOTS GRAPH (when 2 elements in feature set) -------------------------------- !!!

    # if you got 2 dimensional data, plot it, because it's easy to plot that, with more dimensions, it's almost impossible
    if len(ask_IVs) == 2:

        print('')
        print('CHECK THE GRAPH FIRST AND CLOSE IT!!!')
        print('')

        # colors full of 1s and 0s, because 3rd column in that array is the class each feature set belong in, and also set c=colors (the same colors for same centroids)
        colors = X_classifications[:, 2]
        plt.scatter(X_classifications[:, 0], X_classifications[:, 1], s=30, marker='o', c=colors, edgecolors='k',
                    linewidths=1)

        # so scatter those centroids, c=[0, 1]
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=1000, marker='*', c=[x for x in range(k)],
                    edgecolors='k', linewidths=2, label='Cluster Centroids')

        plt.legend(loc='best')
        plt.xlabel(ask_IVs[0])
        plt.ylabel(ask_IVs[1])
        plt.title('k-means (k=' + str(k) + ')')

        plt.show()
    # !!! ----------------------------------- PLOTS GRAPH (when 2 elements in feature set) -------------------------------- !!!



    # !!! ------------------------------------------ PREDICTION MAKING AS Y------------------------------------------------ !!!
    def predict_new_data():
        global ask_IVs

        while True:

            print('')
            # ask the user if wants to make prediction for new data that will be input
            ask_new_data = input('Do you wanna make some predictions on new data, which you wanna type here? Yes/No: ').upper()

            if ask_new_data == 'YES':

                # make an empty array which will include all new data for X
                new_data_array = np.array([])

                # ask for each IV, feature that is in the model for the new data
                for independent_variable in ask_IVs:
                    ask_new_data_IV = input('New Data (split with comma + space) | ' + independent_variable + ' : ').split(', ')

                    # if the data can't be converted into floats, raise an error
                    try:
                        for new_data in ask_new_data_IV:
                            _ = float(new_data)

                    except ValueError:
                        print('')
                        print('Oops something went wrong with the new data you have input, they must be numeric and well separated, try again !!!')
                        predict_new_data()


                    # make a numpy array of it and in float dtype
                    ask_new_data_IV_array = np.array(ask_new_data_IV, dtype=float)

                    # append that array with new data into the empty numpy array
                    new_data_array = np.append(new_data_array, ask_new_data_IV_array)


                # but all data are all in one row, the array is not sorted and not in 2D
                # so to reshape we have to first reshape that each row will be for each feature
                # and we want it to be that each column will be for each feature soo make a transpose
                new_data_array = new_data_array.reshape((len(ask_IVs), len(ask_new_data_IV_array))).transpose()


                # if the algorithm was run with scaled data, scale also the new ones
                if ask_scale == 'YES':
                    # STANDARDIZATION --> X_scaled_new = (X_new - X_new_mean) / X_new_std
                    X_new_means = np.mean(new_data_array, axis=0)
                    X_new_std = np.std(new_data_array, axis=0)
                    new_data_array = (new_data_array - X_new_means) / X_new_std

                    ask_IVs = [column + ' (scaled)' for column in ask_IVs]

                else:
                    pass

                # a prediction lists empty so far
                predictions = []

                # go through every new data feature set
                for new_data_pred in new_data_array:

                    # calculate the distance between that new feature set and every centroid
                    distances_pred = [np.sqrt(np.sum((new_data_pred - centroid) ** 2)) for centroid in cluster_centers]

                    # take the min distance and the index of that distance in that list, that's our prediction a i'th centroid
                    prediction = distances_pred.index(min(distances_pred))

                    # add that prediction to predictions list
                    predictions.append(prediction)

                # make a numpy array of that predictions lists
                predictions = np.array(predictions)

                # make a horizontal stack with new data feature_sets and with their predictions
                new_data_array_with_predictions = np.hstack([new_data_array, predictions.reshape((len(predictions), 1))])

                # make a dataframe out of that array, last element is the prediction
                new_data_pred_df = pd.DataFrame(new_data_array_with_predictions, columns=ask_IVs + ['Pred'])

                print('')
                print('')
                print('NEW OBSERVATIONS WITH THEIR PREDICTIONS')
                print('')
                print(new_data_pred_df.to_string(index=False))
                print('')


                # ask if wants to try again !!!
                print('')
                ask_again = input('Do you wanna try predicting new values again? Yes/No: ').upper()

                if ask_again == 'YES':
                    print('OK, starting again.')

                elif ask_again == 'NO':
                    print('OK, no more predicting new values.')
                    break

                else:
                    print('Oops, something went wrong with your input (Yes/No), try again !!!')


            elif ask_new_data == 'NO':
                print('')
                print('OK, there will be no prediction making.')
                break

            else:
                print('')
                print('Oops, something went wrong with your input (Yes/No), try again !!!')


    predict_new_data()
    # !!! ------------------------------------------ PREDICTION MAKING AS Y------------------------------------------------ !!!


K_Means()
