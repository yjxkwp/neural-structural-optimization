from IPython import display
from PIL import Image
import seaborn
import matplotlib.pyplot as plt
import xarray
import pandas as pd

from neural_structural_optimization import pipeline_utils
from neural_structural_optimization import problems
from neural_structural_optimization import models
from neural_structural_optimization import topo_api
from neural_structural_optimization import train


def train_all_p_1_modified(problem, max_iterations, cnn_kwargs=None):
    args = topo_api.specified_task(problem)
    if cnn_kwargs is None:
        cnn_kwargs = {}
    
    print("train pixcel...")
    model = models.PixelModel(args=args)
    ds_pix = train.train_lbfgs(model, max_iterations)
    
    print("train mma...")
    model = models.PixelModel(args=args)
    ds_mma = train.method_of_moving_asymptotes(model, max_iterations)
    
    print("train cnn...")
    model = models.CNNModel(args=args, **cnn_kwargs)
    ds_cnn = train.train_lbfgs(model, max_iterations)
    
    print("train fcn...")
    model = models.FCNModel(args=args)
    ds_fcn = train.train_lbfgs(model, max_iterations)
    
    print("train unet...")
    model = models.UNetModel(args=args)
    ds_unet = train.train_lbfgs(model, max_iterations)
    
    print("train oc...")
    model = models.PixelModel(args=args)
    ds_oc = train.optimality_criteria(model, max_iterations)
    
    dims = pd.Index(['cnn-lbfgs', 'fcn-lbfgs', 'unet-lbfgs', 'mma', 'oc', 'pixel-lbfgs'], name='model')
    return xarray.concat([ds_cnn, ds_fcn, ds_unet, ds_mma, ds_oc, ds_pix], dim=dims)
#     dims = pd.Index(['cnn-lbfgs', 'fcn-lbfgs', 'unet-lbfgs', 'mma', 'oc'], name='model')
#     return xarray.concat([ds_cnn, ds_fcn, ds_unet, ds_mma, ds_oc], dim=dims)
#     dims = pd.Index(['fcn-lbfgs', 'unet-lbfgs'], name='model')
#     return xarray.concat([ds_fcn, ds_unet], dim=dims)

if __name__ == '__main__':
    problem_p_1_modified = problems.mbb_beam(height=32, width=64)
    max_iterations = 40
    
    # can't upscale by exactly 8x for a 60x20 design region, so upscale by
    # only 4x instead
    ds_p_just_1_modified = train_all_p_1_modified(problem_p_1_modified, max_iterations,
                                                  cnn_kwargs=dict(resizes=(1, 1, 2, 2, 1)))