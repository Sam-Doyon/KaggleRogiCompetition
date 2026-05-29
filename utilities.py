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
        self._index=0

    def __getitem__(self,key:str|int):
        if isinstance(key, int):
            key = self.keys[key]
        well_data, typewell_data = pd.read_csv(self.well_files[key]["Well"]),pd.read_csv(self.well_files[key]["TypeWell"])
        well_data, typewell_data = well_data.set_index("MD").sort_index(), typewell_data.set_index("TVT").sort_index()
        pred_start_idx = well_data.index[(~well_data.TVT_input.isna()).sum()]
        last_known_idx = well_data.index[~well_data.TVT_input.isna()][-1]
        well_data["Known"] = well_data.index<=last_known_idx
        return well_data, typewell_data

    def __len__(self):
        return len(self.keys)
    
    def __iter__(self):
        self._index=0
        return self

    def __next__(self):
        if self._index>= len(self):
            raise StopIteration
        val = self[self._index]
        self._index+=1
        return val
        
