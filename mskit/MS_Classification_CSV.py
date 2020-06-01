import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import plotly.offline as py
import plotly.figure_factory as ff
from plotly import tools
import dash_bio as dashbio
import pandas as pd
import lxml as lx
from pyteomics import mzxml, pylab_aux
from scipy.stats import mode
import scipy.integrate as integrate
from scipy.signal import medfilt
from scipy.linalg import norm
from sklearn import svm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import pairwise_distances
from itertools import combinations
import multiprocessing
import pickle
import csv
import os
import copy
import time

# Global variables
outputs = []
file_obj = {}
areas = []
stdevs = []
csv_data = []
time_span = 120
time_step = 1
num_cores = multiprocessing.cpu_count()

start = time.time()

# import mzXML files to read in
# filepath: path to directory containing mzXML files to be read in
# populates outputs: list of pyteomics objects
# populates file_obj: dict linking pyteomics objects (key) to file names(value)
def import_files(filepath):
    os.chdir(filepath)
    directory = os.listdir(os.getcwd())
    global outputs
    global file_obj
    i = 0

    for file in directory:
        if file.endswith("mzXML"):
            obj = mzxml.read(file)
            file_obj[obj] = file
            outputs.append(obj)
            i = i + 1

def import_file(file):
    if file.endswith("mzXML"):
        obj = mzxml.read(file)
        file_obj_tup = (obj, file)
        outputs.append(obj)
    return file_obj_tup

def import_csv(filepath):
    os.chdir(filepath)
    directory = os.listdir(os.getcwd())
    global outputs
    global file_obj
    global csv_data

    for file in directory:
        if file.endswith("csv"):
            csv_data.append(np.loadtxt(file, delimiter = ','))
            dict_reader = csv.DictReader(file, fieldnames = ['retentionTime', 'intensity array'])
            outputs.append(dict_reader)
            file_obj[dict_reader] = file

# calculate area and stdev for all files
# obj: pyteomics object of spectrum to be analyzed
# populates areas: integrates to find area under spectrum
# populates stdevs: calculates average of stdev in chunks of 500 timesteps
# returns file_area: dict of file name (key) and area (value)
def calc_area_stdev(intensities):

    start = time.time()
    global areas
    global stdevs
    global csv_data

    # intensities /= ((np.amax(intensities)) / 100.0)
    
    chunk_stdev = []
    for y in range(0, intensities.size, 500):
        temp = np.std(intensities[y:y + 500])
        chunk_stdev.append(temp)

    stdev = np.average(chunk_stdev)
    stdev = np.around(stdev, 8)
    stdevs.append(stdev)

    area = integrate.simps(intensities, dx = 3)
    area = np.around(area, 8)
    areas.append(area)

    tup = (area, stdev)
        
    end = time.time()
    return tup

# plot area vs stdev for each spectrum
# displays plot of area vs stdev with file name labels
def plot_area_stdev():

    fig2 = plt.figure(figsize = (20, 20))
    ax2 = fig2.add_subplot(111)

    for i, op in enumerate(file_obj.keys()):
        y = stdevs[i]
        x = areas[i]
        ax2.scatter(x, y)
        ax2.text(x + 0.03, y + 0.03, file_obj[op])
    plt.show()


# SVM for area and stdev
# labels: list of correct labels for plots; 0 bad, 1 good
# returns clf: SVM classifier fit to given data and labels
def make_svm(labels):

    X = np.array([areas, stdevs])
    X = np.transpose(X)

    X = X / np.average(X, axis = 0)

    Y = labels

    clf = svm.SVC(kernel = "linear", C = 150.0)
    clf.fit(X, Y)
    print("Classifier labels:", clf.predict(X))
    saved_clf = pickle.dump(clf, open('svm_clf', 'wb'))
    return saved_clf

# setup to generate plot of SVM; found in scikit docs

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    # x_ = np.arange(x_min, x_max, h)
    # y_ = np.arange(y_min, y_max, h)
    # xx, yy = np.meshgrid(x_, y_)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(x, y, ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    x_ = np.arange(x_min, x_max, 0.02)
    y_ = np.arange(y_min, y_max, 0.02)
    # p1 = go.Contour(x=xx, y=yy, z=Z, 
    #                 colorscale='Bluered',
    #                 showscale=False)
    p1 = dict(
        type = 'contour', x = x_, y = y_, z = Z, x0 = -2, y0 = -2, contours = dict(coloring = 'lines'),
        colorscale = [[0, 'rgb(0, 0, 0)'], [1, 'rgb(0, 0, 0)']], showscale = False)
    # p1 = go.Scatter(x = Z[:,0], y = Z[:,1], mode = 'lines', marker = dict(color = 'black'))
    # out = ax.contourf(xx, yy, Z, **params)
    return p1

# generates color-coded classified plot
# blue region - classified bad; red region - classified good
# x - labeled bad; o - labeled good
# clf: classifier made by make_svm
# labels: list of correct labels for plots; 0 bad, 1 good
# returns good_obj: sublist of outputs labeled 1 by classifier
def plot_svm(clf, labels):
    traces = []

    X = np.array([areas, stdevs])
    X = np.transpose(X)

    X = X / np.average(X, axis = 0)
    
    # Y = labels
    
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    fig = tools.make_subplots(rows=2, cols=2,
                          print_grid=False)
    
    fig3 = plt.figure(figsize = (15, 15))
    ax3 = fig3.add_subplot(111)

    p1 = plot_contours(X0, X1, ax3, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.5)
    predictions = clf.predict(X)

    trace0 = go.Scatter(x = X0, y = X1, mode = 'markers',
        marker = dict(color = predictions, colorscale = [[0, 'rgb(173, 10, 59)'], [1, 'rgb(33, 168, 78)']], size = 15),
        line = dict(color = 'rgb(0, 0, 0)', width = 6), text = labels)
    # trace0 = go.Scatter(x = X0, y = X1, mode = 'markers',
    #     marker = dict(color = clf.predict(X), colorscale = 'Viridis'), text = labels)
    traces.append(p1)
    traces.append(trace0)

    return traces, predictions

def predict_X(clf, a, sd):
    X = np.array([a, sd])
    X = np.transpose(X)

    X = X / np.average(X, axis = 0)
    X = X.reshape(1, -1)

    classes = clf.predict(X)
    return classes

# convert outputs into list of tuples(time, signal intensity)
# obj: pyteomics object to be converted to time, intensity
# returns time_inten: 2D array of retention time values matched with intensity
def parse_outputs(obj):
    time = []
    inten = []
    temp_time_inten = []

    for signal in obj:
        time.append((signal['retentionTime']))
        inten.append(np.sum(signal['intensity array']))
    time = np.array(time)
    inten = np.array(inten)
    temp_time_inten = np.vstack((time, inten))
    temp_time_inten = temp_time_inten.transpose()
    return temp_time_inten

# finds average intensity value over time step to normalize intensity values over timespan
# calls parse_outputs to get 2D array
# returns norm_intensities: list of np.float64 of normalized intensity values for spectra
def cluster_intensities(time_inten):
    global csv_data

    # Normalize then cluster 

    norm_intensities = []
    
    zero = (time_inten[:,0] < 1)
    norm_intensities.append(np.average(time_inten[zero, 1]))
    
    for t in range(1, time_span - 1, time_step):
        indices = (time_inten[:,0] > (t - 1)) & (time_inten[:,0] < (t + 1))
        if len(indices) > 0:
            norm_intensities.append(np.average(time_inten[indices, 1]))
        else:
            norm_intensities.append(0.0)
    
    last = (time_inten[:,0] > (time_span - time_step))
    norm_intensities.append(np.average(time_inten[last, 1]))
    print(len(norm_intensities))
    first = len(norm_intensities) * 0.20
    last = len(norm_intensities) * 0.80
    cut_intensities = [i for i in norm_intensities[int(first): int(last): 1]]
    print(len(cut_intensities))
            
    return cut_intensities

# generates plot of clusters using k-means
# clustered_intensities: resulting list of running all objects through cluster_intensities
# filename: string of name to be used to save plot
# n_clusters: int of number of clusters to be formed by k-means
# prints cluster labels generated by k-means
def plot_kmeans_pca_all(clustered_intensities, filename, n_clusters):
    filtered_data = medfilt((clustered_intensities))
    clustered_data = np.array(filtered_data)

    clustered_data = np.nan_to_num(clustered_data, copy = False)

    pca_data = PCA(n_components = 2).fit_transform(clustered_data)

    kmeans = KMeans(n_clusters = n_clusters).fit(pca_data)
    kmeans.labels_
    print("Cluster Labels:", kmeans.labels_)

    trace0 = go.Scatter(x = pca_data[:,0], y = pca_data[:,1], mode = 'markers',
        marker = dict(color = kmeans.labels_, colorscale = 'Viridis'), text = [file_obj[x] for x in outputs])
    data = [trace0]

    layout = go.Layout(xaxis = dict(zeroline = False, zerolinewidth = 1), yaxis = dict(zeroline = False, zerolinewidth = 1))
    py.plot(data, layout)

    timestr = time.strftime("%Y%m%d-%H%M%S")

# generates labeled plot of clusters separated by Ward clustering using input distance threshold
# clustered_intensities: resulting list of running all objects through cluster_intensities
# string of name to be used to save plot
# dt: distance threshold determines how separated clusters should be
# prints legend for plot labels
def ward_clustering(clustered_intensities, labels, dt = 0.001):
    filtered_data = medfilt((clustered_intensities))  
    clustered_data = np.array(filtered_data)
    # clustered_data = np.around(clustered_data, 8)
    clustered_data = np.nan_to_num(clustered_data, copy = False)
    d_matrix = get_distance_matrix(clustered_data)
    print(d_matrix.shape)
    d_matrix = squareform(d_matrix)

    ward_cl = AgglomerativeClustering(linkage = 'complete', affinity = 'precomputed', 
        n_clusters = None, distance_threshold = dt).fit(d_matrix)
    pca_data = PCA(n_components = 2).fit_transform(d_matrix)

    print("Ward Cluster Labels:", ward_cl.labels_)

    trace0 = go.Scatter(x = pca_data[:,0], y = pca_data[:,1], mode = 'markers',
        marker = dict(color = ward_cl.labels_, colorscale = 'Viridis'), text = labels)
    data = [trace0]

    layout = go.Layout(xaxis = dict(zeroline = False, zerolinewidth = 1), yaxis = dict(zeroline = False, zerolinewidth = 1))
    return trace0

def polysnap_correlation(pattern, other):
    """
    pattern: 1-D np.array
    other: 1-D np.array
    
    """
    assert len(pattern) == len(other)
    r = pearsonr(pattern, other)[0]
    rho = spearmanr(pattern, other)[0]

    dist = 1 + (0.5 * r + 0.5 * rho)
    
    return dist

def get_distance_matrix(patterns):

    NPATTERNS = len(patterns)
    idx = {}
    for i, p in enumerate(patterns):
        idx[tuple(p)] = i
    
    iter_pairs = combinations(patterns, 2)
    
    c_matrix = np.zeros((NPATTERNS, NPATTERNS))
    
    for p_i, p_j in iter_pairs:
        coeff = polysnap_correlation(p_i, p_j)
        c_matrix[idx[tuple(p_i)], idx[tuple(p_j)]] = coeff
        c_matrix[idx[tuple(p_j)], idx[tuple(p_i)]]  = coeff
    
    for n in range(NPATTERNS):
        c_matrix[n, n] = 1.0
    
    d_matrix = 0.5 * (1.0 - c_matrix)
    d_matrix = squareform(d_matrix)
    return d_matrix

def plot_clustergram(patterns, NPATTERNS, linkage_method):
    PAD_SIZE = int(np.log10(NPATTERNS)) + 1
    PAD_FMT = '%0' + str(PAD_SIZE) + 'd'

    labels = [ PAD_FMT % (n + 1) for n in range(NPATTERNS) ]
    d_matrix = get_distance_matrix(patterns)
    print(patterns.shape)

    cutoff = np.amax(patterns) * (NPATTERNS * 0.08)
    # cutoff = 1600000000
    print(cutoff)

    clustergram = dashbio.Clustergram(data = patterns, cluster = 'row', row_dist = 'euclidean',
     display_ratio = [0.75, 0.5], row_labels = labels, color_map = 'Viridis', width = 1200, height = 1200,
     symmetric_value = False, color_threshold = {'row' : cutoff})

    cluster_table = print_clusters(clustergram)

    return clustergram, cluster_table

def print_clusters(clustergram):
    cluster_table = {}
    tickvals = clustergram['layout']['yaxis5']['tickvals']
    ticktext = clustergram['layout']['yaxis5']['ticktext']
    for att in clustergram['data']:
        den_vals = att['x']
        indices = []
        for i, z in enumerate(den_vals):
            if z == 0:
                indices.append(i)
        y = att['y']
        y_vals = [y[i] for i in indices]
        indices = []
        for val in y_vals:
            for i, tick in enumerate(tickvals):
                if val == tick:
                    indices.append(i)
                    break
        cluster = att['name']
        if cluster is None: continue
        cluster_comps = [ticktext[i] for i in indices]
        if len(cluster_comps) == 0: continue
        cluster_table[cluster] = cluster_comps
    # df = pd.DataFrame.from_dict(cluster_table, orient = 'index')
    # df.transpose()
    # df.replace(to_replace = [None], value = '', inplace = True)
    # return df.to_dict()
    # print(cluster_table)
    return cluster_table


if __name__ == "__main__":
    # gather input files (csv format)
    import_csv("/home/flynne03/wip/Projects/Data/TestSet/trial_set/trial_set")

    # part_clustered = [cluster_intensities(i) for i, obj in enumerate(outputs)]
    # ward_clustering(part_clustered, "clusterpca_alldata", 10000000000)

    # # plot based on kmeans clusters for all data
    # full_clustered = Parallel(n_jobs = num_cores)(delayed(cluster_intensities)(x, i) for i, x in enumerate(outputs))
    # plot_kmeans_pca_all(full_clustered, "clusterpca_alldata", 2)

    # # plot based on ward clusters for all data
    # full_clustered = Parallel(n_jobs = num_cores)(delayed(cluster_intensities)(i) for i in range(0, len(outputs), 1))
    # ward_clustering(full_clustered, "wardcluster_noiseless10000000000", 10000000000)

    # calculate area and stdev for each spectrum; populates areas and stdevs
    a_sd = [calc_area_stdev(i) for i in range(0, len(outputs), 1)]
    areas = [i[0] for i in a_sd]
    print(areas)
    stdevs = [i[1] for i in a_sd]
    print(stdevs)

    # input correct labels for spectra to feed into classifier
    labels = [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]

    # constructs svm classifier clf based on given labels
    saved_clf = make_svm(labels)
    # clf = pickle.loads(saved_clf)


    # plots classification of full data based on clf predictions
    good_obj = plot_svm(clf, labels)

    # plots based on kmeans clusters for good spectra only
    # good_clustered = Parallel(n_jobs = num_cores)(delayed(cluster_intensities)(i) for i in range(0, len(good_obj), 1))
    good_clustered = [cluster_intensities(i) for i in range(0, len(good_obj), 1)]
    ward_clustering(good_clustered, "clusterpca_noiselessdata", 10000000000)

    end = time.time()
    print('Runtime:', (end - start))
