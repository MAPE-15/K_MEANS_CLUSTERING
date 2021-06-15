# K-MEANS CLUSTERING ALGORITHM WITH USING SKLEARN !!!

# FINISHED !!!


import numpy as np
import pandas as pd

# custom made, made it myself, just for plot analysis and customizing dataset, can be found in github plot analysis and dataset customizing
from DATASET_CUSTOMIZING.DATASET_READING_CUSTOMIZING_INPUT import make_dataset
from PLOT_ANALYSIS.PLOT_ANALYSIS_INPUT import make_analysis

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib import style

import pickle


# classes for some specific error
class LabelError(Exception):
    pass

class InputError(Exception):
    pass

class Exceptions(Exception):
    pass

# show all columns in the dataframe while printing it
pd.set_option('display.max_columns', None)


def K_Means(df):

    # make plot analysis if user wishes to do so
    make_analysis(df)


    while True:

        # check for any input error
        try:
            print(df)

            print('')
            print('THESE ARE YOUR COLUMNS NAMES:', list(df.columns))
            print('')

            # ask for column(s) which will be all X values, which will be our features
            ask_IVs = input('Type all column names in the dataset, you want for X (Independent variables / features) (split with comma + space): ').split(', ')

            # if that column that was input does not occur in the dataset, raise an error
            for col_X in ask_IVs:
                if col_X not in list(df.columns):
                    raise Exceptions


            # make a 2D numpy array of those feature sets in that dataset
            X = np.array(df[ask_IVs])


            # ask if user wants to standardize the data
            print('')
            ask_scale = input('Do you also wanna scale your feature sets? Yes/No: ').upper()

            if ask_scale == 'YES':
                # make a classifier for scaling the data
                scale = StandardScaler()
                # train the scaler for data X, we'll get the means and stds
                scale.fit(X)
                # and finally scale tha data for X
                X = scale.transform(X)

            elif ask_scale == 'NO':
                pass

            else:
                raise InputError


            print('')
            print('''!!! CHECK THE INERTIA PLOT TO DETERMINE THE BEST NUMBER OF CLUSTERS FOR THE MODEL !!!
Inertia Plot (Elbow Method) --> The point where the drop stops to be significant and more linear if the best point for the number of clusters !!!''')


            # !!! -------------------------------------- INERTIA PLOT (ELBOW METHOD) -------------------------------- !!!
            # a list of inertias
            inertia = []

            # run this inertia for number of clusters from 1 to 11
            for i in range(1, 11):
                kmeans_inertia = KMeans(n_clusters=i)
                kmeans_inertia.fit(X)
                # get the inertia and append to the list
                inertia.append(kmeans_inertia.inertia_)


            # plot the Inertia Plot Elbow Method
            plt.plot(np.arange(1, 11), inertia, marker='o', color='#050311')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            plt.title('Inertia Plot (Elbow Method)\nTake the one where the decrease has started to decrease slowly !!!')
            plt.show()
            # !!! -------------------------------------- INERTIA PLOT (ELBOW METHOD) -------------------------------- !!!


            # ask if want to have default parameters or want to specify them
            print('')
            ask_specify_default = input('Wanna specify the n_clusters(k)/maximum_iterations/tolerance or want to leave it all in default (default k=2; max_iter=300; tol=0.001) --> Specify/Default: ').upper()

            if ask_specify_default == 'SPECIFY':

                # ask for n_clusters and assign it to k
                print('')
                ask_parameters = input('OK, give me the specifications in this order !!! --> k/max_iter/tol (split with slash): ').split('/')

                if len(ask_parameters) != 3:
                    raise Exceptions

                k = int(ask_parameters[0])
                max_iter = int(ask_parameters[1])
                tol = float(ask_parameters[2])


            # default n_cluster is equal to 3
            elif ask_specify_default == 'DEFAULT':
                k = 2
                max_iter = 300
                tol = 0.001

            else:
                raise InputError


            print('')
            # ask for one single column which will be our label Y, for making accuracy check (DV)
            ask_DV = input('Type column name in the dataset you want for label Y, must be categorical column, and must have the same amount of unique values as k (n_clusters) !!!: ').split(', ')

            # if there will be more than one column for label input, raise an error
            if len(ask_DV) != 1:
                raise LabelError

            for col_y in ask_DV:

                # and again if that label column does not occur in the dataset, raise an error
                if col_y not in list(df.columns):
                    raise LabelError

                # if that column has more or less unique values than k (n_clusters) raise an error
                if len(df[col_y].unique()) != k:
                    raise LabelError

            # y label in 1d numpy array
            y = np.array(df[''.join(ask_DV)])


        except ValueError:
            print('')
            print('Oops, something went wrong, an input must be number, try again !!!')


        except InputError:
            print('')
            print('Oops, something went wrong with your input, try again !!!')


        except Exceptions:
            print('')
            print('''At least one the columns you have input does not appear in the dataset 
or the number of parameters does not match with your input, try again !!!''')


        except LabelError:
            print('')
            print('''Oops, something went wrong, must be only ONE label, label MUST be categorized, 
and MUST be in that dataset, and MUST have the same amount of unique values as k (n_clusters), try again !!!''')


        # if everything seems to be working and no error has been raised, break the loop
        else:
            break



    while True:

        # the maximum accuracy the model can get, only to accuracy poles are there, when the centroid are exchanged or not, the bigger percentage wins
        # f.e. 1st acc = 80%, 2nd acc = 20%, the model with acc 20% had probably centroids exchanged (0 centroid had to be 1, and 1 had to be 0), so we take in account the model with 80%
        # we take the models best accuracy, and with the best accuracy we take its best cluster centroids coordinated, and predicted values

        accuracy_max = 0
        cluster_centers_final = []
        predicted_values = []


        print('')
        # ask the user if wants to save or load the model
        ask_save_load = input('Do you want to save or load your best model? Save/Load: ').upper()

        # if wants to save the best model
        if ask_save_load == 'SAVE':

            print('')
            # ask for the name you want your best model to be saved
            ask_save_name = input('OK, give me the name of your best model you want to have in your safe, it will be saved in this directory: ')

            for _ in range(10):

                # instantiate the model
                model = KMeans(n_clusters=k, max_iter=max_iter, tol=tol)
                # train the model with X
                model.fit(X)

                # predict the values for X
                predicted = model.predict(X)

                correct = 0
                for i in range(len(y)):
                    # compare the i'th y (original) and the i'th predicted, if they match, correct += 1
                    if y[i] == predicted[i]:
                        correct += 1

                # accuracy percentage --> correct_amount / overall_amount * 100
                accuracy = np.round(correct / len(X) * 100, 2)


                # take the biggest/best accuracy and assign it to accuracy_max, and assign the best cluster_centroids coords and predicted values
                if accuracy > accuracy_max:
                    accuracy_max = accuracy

                    cluster_centers_final.clear()
                    cluster_centers_final.append(model.cluster_centers_)

                    predicted_values.clear()
                    predicted_values.append(predicted)

                    # also save that best model using open with() function and using pickle.dump(instance, f)
                    with open(ask_save_name + '.pickle', 'wb') as f:
                        pickle.dump(model, f)


            # break the loop if everything seems to be working
            break


        # if user wants to load the model
        elif ask_save_load == 'LOAD':

            # ask for the model name
            ask_load_name = input('OK, give me the name of the saved model you want to load in (pickle file, f.e., name1.pickle): ')

            # check for any errors
            try:

                # to read that model, first open it in read binary mode and with open('name') function
                pickle_in = open(ask_load_name, 'rb')
                # and to make, to load that file and make an linear regression instance --> pickle.load()
                model = pickle.load(pickle_in)

                # predict the values for X
                predicted = model.predict(X)

                # append those cluster centers coords and predicted values to those lists
                cluster_centers_final.append(model.cluster_centers_)
                predicted_values.append(predicted)


                correct = 0
                for i in range(len(y)):
                    # compare the i'th y (original) and the i'th predicted, if they match, correct += 1
                    if y[i] == predicted[i]:
                        correct += 1

                # calculate the accuracy of that model
                accuracy_max = np.round(correct / len(y) * 100, 2)

                # break the loop if everything seems to be working
                break

            except Exception:
                print('')
                print('Oops, model name does not exist, try all over again !!!')

        else:
            print('')
            print('Oops, wrong input Save/Load, try again !!!')



    # just take the cluster centroids numpy array from that list, the same with predicted values
    cluster_centers_final = cluster_centers_final[0]
    predicted_values = predicted_values[0]



    print('')
    print('''IT CAN HAPPEN THAT CENTROIDS ARE EXACT OPPOSITE IN COMPARISON OF THE VALUES IN YOUR LABEL, SO HERE YOU HAVE 2 ACCURACY POLES
THE BIGGER PERCENTAGE SHOWS THE ACCURACY OF THE MODEL, BOTH TOGETHER ADD UP TO 100% !!! 

Higher Percentage Accuracy pole:''', str(accuracy_max) + '%', '''
Lower Percentage Accuracy pole:''', str(100 - accuracy_max) + '%')



    # !!! ----------------------------------- PLOTS GRAPH (when 2 elements in feature set) -------------------------------- !!!

    # if you got 2 dimensional data, plot it, because it's easy to plot that, with more dimensions, it's almost impossible
    if len(ask_IVs) == 2:
        style.use('ggplot')

        print('')
        print('CHECK THE GRAPH FIRST AND CLOSE IT!!!')
        print('')

        # plot the x1 and x2, each color belongs to specific cluster/group
        plt.scatter(X[:, 0], X[:, 1], s=30, marker='o', c=predicted_values, edgecolors='k', linewidths=1)

        # so scatter those centroids, c=[0, 1]
        plt.scatter(cluster_centers_final[:, 0], cluster_centers_final[:, 1], s=1000, marker='*', c=[x for x in range(k)],
                    edgecolors='k', linewidths=2, label='Cluster Centroids')


        plt.legend(loc='best')
        plt.xlabel(ask_IVs[0])
        plt.ylabel(ask_IVs[1])
        plt.title('k-means (k=' + str(k) + ')')

        plt.show()
    # !!! ----------------------------------- PLOTS GRAPH (when 2 elements in feature set) -------------------------------- !!!



    # !!! ------------------------------------------ PREDICT NEW DATA ---------------------------------------------------- !!!
    while True:

        try:

            print('')
            # ask the user if wants to make prediction for new data that will be input
            ask_new_data = input('Do you wanna make some predictions on new data, which you wanna type here? Yes/No: ').upper()

            if ask_new_data == 'YES':

                # make an empty array which will include all new data for X
                new_data_array = np.array([])

                # ask for each IV, feature that is in the model for the new data
                for independent_variable in ask_IVs:
                    ask_new_data_IV = input('New Data (split with comma + space) | ' + independent_variable + ' : ').split(
                        ', ')

                    # if the data can't be converted into floats, raise an error
                    for new_data in ask_new_data_IV:
                        _ = float(new_data)


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
                    # we already got our scaler train (fitted) so just transform the new data and standardize/scale them
                    new_data_array = scale.transform(new_data_array)

                    # to each column name add 'scaled' to its name for the user to know that it scaled his/hers new data
                    ask_IVs = [column + ' (scaled)' for column in ask_IVs]

                else:
                    pass


                # 1d numpy array with predicted valued for new data
                predictions_new = model.predict(new_data_array)


                # make a horizontal stack with new data feature_sets and with their predictions
                new_data_array_with_predictions = np.hstack([new_data_array, predictions_new.reshape((len(predictions_new), 1))])

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


        except ValueError:
            print('')
            print('Oops something went wrong with the new data you have input, they must be numeric and well separated, try again !!!')
    # !!! ------------------------------------------ PREDICT NEW DATA ---------------------------------------------------- !!!



# read or make some customization for the dataset
# df = make_dataset()

# K_Means(df)
