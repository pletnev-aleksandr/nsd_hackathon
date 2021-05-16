import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
from PIL import Image
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from pandas.core.common import flatten
from sklearn.cluster import DBSCAN
import pickle
def load_pickle(file_name):
    with open( file_name, 'rb') as reader:
        data = pickle.load(reader)
    return data
from sklearn.mixture import GaussianMixture
import langid
from matplotlib.colors import ListedColormap
from sklearn.cluster import DBSCAN#
from sklearn.cluster import KMeans
import pytesseract
from PIL import Image
import pickle
import os
import numpy as np
import pandas as pd
import xlsxwriter

def load_pickle(file_name):
    with open( file_name, 'rb') as reader:
        data = pickle.load(reader)
    return data


def create_local_path(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as e:
        pass
        #logging.error(e)



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def plot_cells_by_clusters(path_to_image,df_coordinates, output_filename = 'filename.png', label_column = 'labels'):
    """
    :params df_coordinates: dataframe where cells are represented as (x1,y1,x2,y2)
    """
    im = Image.open(r'input/'+path_to_image)
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15,15))

    # Display the image
    ax.imshow(im)
    palette = itertools.cycle(sns.color_palette('bright', df_coordinates[label_column].nunique()))
    for cluster in df_coordinates[label_column].unique():
        color =next(palette)
        for idx,row in df_coordinates.loc[df_coordinates[label_column]==cluster].iterrows():
            left_top = (row.x1,row.y1)
            x_delta = row.x2-row.x1
            y_delta = row.y2-row.y1
            # Create a Rectangle patch
            rect = patches.Rectangle(left_top, x_delta, y_delta, linewidth=2, edgecolor=color, facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
    plt.savefig(output_filename, dpi=300)
    plt.show()


def generate_small_cropped_cell_images(df_coordinates,original_filename,original_image):
    """

    :param df_coordinates:
    :param original_filename:
    :param original_image:
    :return:
    """
    create_local_path(r'output/cropped_cells_images/' + original_filename)
    for idx, cell in df_coordinates.iterrows():
        cut_cell = cell.copy(deep=True)
        cut_cell.x1 -= 4
        cut_cell.y1 -= 4
        cut_cell.x2 += 4
        cut_cell.y2 += 4
        cropped_cell = original_image.crop(cut_cell)
        name = r'output/cropped_cells_images/' + original_filename + '/' + f'{idx}' + '.png'
        cropped_cell.save(name, quality=95)


def run_tessaract_for_cut_cells(df_coordinates,original_filename):
    parsed_text = []
    for idx, row in df_coordinates.iterrows():
        name = r'output/cropped_cells_images/' + original_filename + '/' + f'{idx}' + '.png'
        cropped_image = Image.open(name)
        parsed_cell_rus = pytesseract.image_to_string(cropped_image, lang='rus',
                                    config='--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"')
        parsed_cell_eng = pytesseract.image_to_string(cropped_image, lang='eng',
                                                  config='--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"')
        # we need to determine what text is in this cell ( it may be English or Russian)
        try:
            lang_r, score_r = langid.classify(parsed_cell_rus)
            lang_e, score_e = langid.classify(parsed_cell_eng)
            if score_e>score_r:
                if lang_e.lang=='en':
                    parsed_cell = parsed_cell_eng
            else:
                parsed_cell = parsed_cell_rus
        except:
            # no features in text, most likely a number
            parsed_cell = parsed_cell_rus
        if parsed_cell.count('\n')==1:
            #Succesfull parse
            parsed_cell = parsed_cell.split('\n')[0]
        else:
            parsed_cell = parsed_cell
        parsed_text.append(parsed_cell)
    return parsed_text


def create_parsed_tabular_dataset(df_coordinates):
    x_centroids = df_coordinates.groupby(by='x_labels').mean()[['x_center']].sort_values(by='x_center').reset_index(drop=True)
    y_centroids = df_coordinates.groupby(by='y_labels').mean()[['y_center']].sort_values(by='y_center').reset_index(drop=True)
    filled_dataframe_cell_idx = pd.DataFrame(index=y_centroids.index, columns=x_centroids.index)
    filled_dataframe_cell_parses_text = pd.DataFrame(index=y_centroids.index, columns=x_centroids.index)
    # filled_dataframe_cell_idx = filled_dataframe_cell_idx.fillna(None)
    for idx, cell in df_coordinates.iterrows():
        x_cell_ind = x_centroids.sub(cell.x_center).abs().idxmin().values[0]
        y_cell_ind = y_centroids.sub(cell.y_center).abs().idxmin().values[0]
        if filled_dataframe_cell_idx.iloc[y_cell_ind, x_cell_ind] is np.nan:
            filled_dataframe_cell_idx.iloc[y_cell_ind, x_cell_ind] = idx
            filled_dataframe_cell_parses_text.iloc[y_cell_ind, x_cell_ind] = cell.parsed_text
        else:
            # ravel list of lists
            filled_dataframe_cell_idx.iloc[y_cell_ind, x_cell_ind] = list(
                flatten([filled_dataframe_cell_idx.iloc[y_cell_ind, x_cell_ind], idx]))
            filled_dataframe_cell_parses_text.iloc[y_cell_ind, x_cell_ind] = filled_dataframe_cell_parses_text.iloc[
                                                                                 y_cell_ind, x_cell_ind] + ' // ' + cell.parsed_text

    return filled_dataframe_cell_idx,filled_dataframe_cell_parses_text
