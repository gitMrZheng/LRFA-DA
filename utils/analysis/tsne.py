import torch
import matplotlib

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col


def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='#88c999', target_color='red'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)


    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 0 represents source while 1 represents target
    domains = np.concatenate((np.zeros(len(source_feature)), np.ones(len(target_feature))))

    # visualize using matplotlib
    fig,ax=plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    Scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='.', c=domains,
                         cmap=col.ListedColormap(
                             [source_color, target_color]), s=3)
    a, b = Scatter.legend_elements(alpha=0.6)
    b = ['$\\mathdefault {Source}$' + ' ' + 'domain',
         '$\\mathdefault {Target}$' + ' ' + 'domain']
    Legend = ax.legend(a, b, loc='best')
    ax.add_artist(Legend)
    plt.savefig(filename)
def visualize1(source_feature: torch.Tensor, target_feature: torch.Tensor,source_labels,target_labels,
              filename: str,source_benign_color='#88c999',source_malignant='red',target_benign_color='royalblue',target_malignant_color='hotpink'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    source_labels = source_labels.numpy()
    target_labels = target_labels.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)
    labels=np.concatenate([source_labels,target_labels],axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)


    # visualize using matplotlib
    fig,ax=plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    Scatter=ax.scatter(X_tsne[:, 0], X_tsne[:, 1],marker='.',c=labels,
                cmap=col.ListedColormap([source_benign_color, source_malignant,target_benign_color,target_malignant_color]), s=3)
    a,b=Scatter.legend_elements(alpha=0.6)
    b=['$\\mathdefault {Source}$'+' '+'benign'+' '+'class',
       '$\\mathdefault {Source}$'+' '+'maglignant'+' '+'class',
       '$\\mathdefault {Target}$'+' '+'benign'+' '+'class',
       '$\\mathdefault {Target}$'+' '+'maglignant'+' '+'class']
    Legend=ax.legend(a,b,loc='best')
    ax.add_artist(Legend)
    plt.savefig(filename)

def visualize2(source_feature: torch.Tensor, target_feature: torch.Tensor,source_labels,target_labels,
              filename: str,source_color='royalblue',target_color='hotpink'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    source_labels = source_labels.numpy()
    target_labels = target_labels.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)
    labels=np.concatenate([source_labels,target_labels],axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)


    # visualize using matplotlib
    fig,ax=plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    Scatter=ax.scatter(X_tsne[:, 0], X_tsne[:, 1],marker='.',c=labels,
                cmap=col.ListedColormap([source_color, target_color]), s=3)
    a,b=Scatter.legend_elements(alpha=0.6)
    b=['$\\mathdefault {Benign}$',
       '$\\mathdefault {Malignant}$']
    Legend=ax.legend(a,b,loc='best')
    ax.add_artist(Legend)
    plt.savefig(filename)