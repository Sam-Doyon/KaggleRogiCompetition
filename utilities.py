import pandas as pd
import numpy as np
import os, sys
from tqdm.notebook import tqdm
from pathlib import Path
from plotly import express as px
import plotly.graph_objects as go
from sklearn.linear_model import  LinearRegression

def plot_wellbore(well_data):
    pred_start = well_data.TVT_input.shift(1).dropna().index[-1]
    fig = px.scatter_3d(well_data, x="X", y="Y", z="Z", color="TVT").update_layout(height=800, width=800)
    fig.update_traces(marker=dict(size=2))
    start_point = well_data.loc[pred_start]
    fig.add_trace(go.Scatter3d(
        x=[start_point["X"]],
        y=[start_point["Y"]],
        z=[start_point["Z"]],
        mode="markers",
        marker=dict(color="red", size=8),
        name="Prediction Start",
        showlegend=False
    ))
    fig.show()


class RogiDataset():
    def __init__(self, path=Path("../../rogii-wellbore-geology-prediction/train",)):
        self.path = path
        self.well_files={
            f.name.replace("__horizontal_well.csv",""):{"Well":f,"TypeWell":f.parent/f.name.replace("__horizontal_well.csv","__typewell.csv")}
            for f in self.path.glob("*__horizontal_well.csv")
            }
        self.keys = list(self.well_files.keys())

    def __getitem__(self,key:str|int):
        if isinstance(key, int):
            key = self.keys[key]
        return pd.read_csv(self.well_files[key]["Well"]),pd.read_csv(self.well_files[key]["TypeWell"])
