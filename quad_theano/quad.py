import numpy as np
import theano.tensor as tt
import theano


n_grid = 50

abisca, wg = np.polynomial.legendre.leggauss(n_grid)
ww = np.outer(wg,wg)
www = np.zeros((len(wg),len(wg),len(wg)))
for i, wi in enumerate(wg):
    for j, wj in enumerate(wg):
        for k, wk in enumerate(wg):
            
            www[i,j,k] = wi*wj*wk


wg = theano.shared(wg.reshape(n_grid,1))
ww = theano.shared(ww)
www = theano.shared(www)
 



def quad(f, x_min, x_max, * parameters):

    x_diff = 0.5 * (xmax-xmin)
    x_plus = 0.5 * (xmax+xmin)
    
    
    
    dx = (tg * x_diff).reshape((1,n_grid))
    
    eval_matrix = f(x_plus + dx, *parameters) + f(x_plus - dx, *parameters)
    
    val = wg.dot(eval_matrix)
    
    return .5 * x_diff * val


def double_quad(f, xmin, xmax, ymin, ymax, *parameters):
    
    
    x_diff = 0.5 * (xmax-xmin)
    x_plus = 0.5 * (xmax+xmin)
    
    y_diff = 0.5 * (ymax-ymin)
    y_plus = 0.5 * (ymax+ymin)
    
    dx = (tg * x_diff).reshape((1,n_grid))
    dy = (tg * y_diff).reshape((n_grid,1))
    
    eval_matrix = f(x_plus + dx, y_plus + dy, *parameters ) + f(x_plus - dx, y_plus - dy, *parameters )
    
    val = tt.mul(ww,eval_matrix)
    
    return .5 * y_diff * x_diff * val.sum()

def triple_quad(f, xmin, xmax, ymin, ymax, zmin, zmax, *parameters):
    
    
    x_diff = 0.5 * (xmax-xmin)
    x_plus = 0.5 * (xmax+xmin)
    
    y_diff = 0.5 * (ymax-ymin)
    y_plus = 0.5 * (ymax+ymin)
    
    z_diff = 0.5 * (zmax-zmin)
    z_plus = 0.5 * (zmax+zmin)
    
    dx = (tg * x_diff).reshape((n_grid,1,1))
    dy = (tg * y_diff).reshape((1,n_grid,1))
    dz = (tg * z_diff).reshape((1,1,n_grid))
    
    eval_matrix = f(x_plus + dx, y_plus + dy, z_plus + dz, *parameters ) + f(x_plus - dx, y_plus - dy,  z_plus - dz  *parameters)
    
    val = tt.mul(www,eval_matrix)
    
    return .5 * z_diff * y_diff * x_diff * val.sum()
