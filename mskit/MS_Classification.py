import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import pandas as pd
import lxml as lx
from pyteomics import mzxml, pylab_aux
import scipy.integrate as integrate
from scipy.signal import medfilt
from scipy.linalg import norm
from sklearn import svm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import multiprocessing
import os
import copy
import time

#plotly.tools.set_credentials_file(username='EF14', api_key='YVWK6BPHXnCG9GDmTAz6')

# Global variables
outputs = []
file_obj = {}
areas = []
stdevs = []
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

def import_file(filename):
    if file.endswith("mzXML"):
        obj = mzxml.read(file)
        file_obj_tup = (obj, file)
        outputs.append(obj)
    return file_obj_tup

# calculate area and stdev for all files
# obj: pyteomics object of spectrum to be analyzed
# populates areas: integrates to find area under spectrum
# populates stdevs: calculates average of stdev in chunks of 500 timesteps
# returns file_area: dict of file name (key) and area (value)
def calc_area_stdev(obj):

    start = time.time()
    file_area = {}
    global areas
    global stdevs
    
    intensities = []

    title, end = file_obj[obj].split(".")

    for att in obj:
        intensities.append(np.sum(att['intensity array']))

    intensities /= ((np.amax(intensities)) / 100.0)

    chunk_stdev = []
    for y in range(0, len(intensities), 500):
        temp = np.std(intensities[y:y + 500])
        chunk_stdev.append(temp)

    stdev = np.average(chunk_stdev)
    stdevs.append(stdev)

    area = integrate.simps(intensities, dx = 3)
    file_area[title] = area
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

    timestr = time.strftime("%Y%m%d-%H%M%S")
    # fig2.savefig("/home/flynne03/wip/Projects/MS_Outlier/Plots/areavstdev" + timestr + ".png")

# SVM for area and stdev
# labels: list of correct labels for plots; 0 bad, 1 good
# returns clf: SVM classifier fit to given data and labels
def make_svm(labels):

    X = np.array([areas, stdevs])
    X = np.transpose(X)

    X = X / np.average(X, axis = 0)

    Y = labels

    clf = svm.SVC(kernel = "linear", C = 200.0)
    clf.fit(X, Y)
    return clf

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
    p1 = dict(type = 'contour', x = x_, y = y_, z = Z, x0 = -2, y0 = -2,
        colorscale = 'Jet')
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
    
    Y = labels
    
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    fig = tools.make_subplots(rows=2, cols=2,
                          print_grid=False)
    
    fig3 = plt.figure(figsize = (15, 15))
    ax3 = fig3.add_subplot(111)

    goodX0= []
    goodX1 = []
    badX0 = []
    badX1 = []
    badl = []
    goodl = []
    
    good_obj = []

    for i, data in enumerate(Y):
        if data == 0:
            badX0.append(X0[i])
            badX1.append(X1[i])
            badl.append(file_obj[outputs[i]])
            ax3.text(X0[i] + (0.01), X1[i] + 0.01, i)
        else:
            goodX0.append(X0[i])
            goodX1.append(X1[i])
            goodl.append(file_obj[outputs[i]])
            good_obj.append(outputs[i])
            ax3.text(X0[i] + (0.01), X1[i] + 0.01, i)

    p1 = plot_contours(X0, X1, ax3, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.5)
    ax3.scatter(badX0, badX1, marker = "x", s=20, edgecolors='k')
    ax3.scatter(goodX0, goodX1, marker = "o", s=20, edgecolors='k')

    trace0 = go.Scatter(x = X0, y = X1, mode = 'markers',
        marker = dict(color = clf.predict(X), colorscale = 'Picnic'), text = [file_obj[x] for x in outputs])
    traces.append(p1)
    traces.append(trace0)
    py.plot(traces)

    print("Classifier labels:", clf.predict(X))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    return good_obj

# convert outputs into list of tuples(time, signal intensity)
# obj: pyteomics object to be converted to time, intensity
# returns time_inten: 2D array of retention time values matched with intensity
def parse_outputs(obj):
    time = []
    inten = []
    time_inten = []

    for signal in obj:
        time.append((signal['retentionTime']))
        inten.append(np.sum(signal['intensity array']))
    time = np.array(time)
    inten = np.array(inten)
    time_inten = np.vstack((time, inten))
    time_inten = time_inten.transpose()
    return time_inten

# finds average intensity value over time step to normalize intensity values over timespan
# calls parse_outputs to get 2D array
# returns norm_intensities: list of np.float64 of normalized intensity values for spectra
def cluster_intensities(obj):
    
    time_inten = parse_outputs(obj)
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
            
    return norm_intensities

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
def ward_clustering(clustered_intensities, filename, dt = 10000000000):
    filtered_data = medfilt((clustered_intensities))
    clustered_data = np.array(filtered_data)
    clustered_data = np.nan_to_num(clustered_data, copy = False)
    dist = norm(clustered_data[0:5])
    
    
    ward_cl = AgglomerativeClustering(n_clusters = None, distance_threshold = dt).fit(clustered_data)
    pca_data = PCA(n_components = 2).fit_transform(clustered_data)
    
    fig5 = plt.figure(figsize = (10, 10))
    ax5 = fig5.add_subplot(111)
    
    for i, coord in enumerate(pca_data):
        ax5.text(coord[0] + (0.08), coord[1] + 0.08, i)
    
    ax5.scatter(pca_data[:,0], pca_data[:,1], c = ward_cl.labels_)

    print("Ward Cluster Labels:", ward_cl.labels_)

    trace0 = go.Scatter(x = pca_data[:,0], y = pca_data[:,1], mode = 'markers',
        marker = dict(color = ward_cl.labels_, colorscale = 'Viridis'), text = [file_obj[x] for x in outputs])
    data = [trace0]

    layout = go.Layout(xaxis = dict(zeroline = False, zerolinewidth = 1), yaxis = dict(zeroline = False, zerolinewidth = 1))
    py.plot(data, layout)

    timestr = time.strftime("%Y%m%d-%H%M%S")

# gather input files (mzXML format)
import_files("/home/flynne03/wip/Projects/Data/TestSet/trial_set/trial_set")

# time and intensity 2D array for plotting spectra
plot_data = Parallel(n_jobs = num_cores)(delayed(parse_outputs)(x) for x in outputs)

# plot based on kmeans clusters for all data
full_clustered = Parallel(n_jobs = num_cores)(delayed(cluster_intensities)(x) for x in outputs)
plot_kmeans_pca_all(full_clustered, "clusterpca_alldata", 2)

# plot based on ward clusters for all data
full_clustered = Parallel(n_jobs = num_cores)(delayed(cluster_intensities)(x) for x in outputs)
ward_clustering(full_clustered, "wardcluster_noiseless10000000000", 10000000000)

# calculate area and stdev for each spectrum; populates areas and stdevs
a_sd = Parallel(n_jobs = num_cores)(delayed(calc_area_stdev)(x) for x in outputs)
areas = [i[0] for i in a_sd]
stdevs = [i[1] for i in a_sd]

# input correct labels for spectra to feed into classifier
labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0]

# constructs svm classifier clf based on given labels
clf = make_svm(labels)

# plots classification of full data based on clf predictions
good_obj = plot_svm(clf, labels)

# plots based on kmeans clusters for good spectra only
good_clustered = Parallel(n_jobs = num_cores)(delayed(cluster_intensities)(x) for x in good_obj)
ward_clustering(good_clustered, "clusterpca_noiselessdata", 10000000000)

end = time.time()
print('Runtime:', (end - start))
