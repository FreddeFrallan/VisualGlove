def _getNColors(numberOfColors):
    import colorsys
    N = numberOfColors
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return list(RGB_tuples)

def plotEmbeddings(points, labels, targetNames, title=""):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np

    print(len(points))
    print(len(points[0]))
    pca = PCA(n_components=2)
    X_r = pca.fit(points).transform(points)

    plt.figure()
    numberOfClasses = len(np.unique(labels))
    colors = _getNColors(numberOfClasses)
    lw = 0.5

    for color, i, target_name in zip(colors, list(range(numberOfClasses)), targetNames):
        plt.scatter(X_r[labels == i, 0], X_r[labels == i, 1], color=color, alpha=1, lw=lw, label=target_name)

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.show()