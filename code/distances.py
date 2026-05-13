import minirocket_multivariate_variable as mmv

def pca_mr_distance(x, y):
    return mmv.transform(x) - mmv.transform(y)