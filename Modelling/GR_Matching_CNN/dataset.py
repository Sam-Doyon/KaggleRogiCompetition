import os, sys
sys.path.append("../..")
from utilities import plot_wellbore, RogiDataset
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
from plotly import express as px
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset



        


class MatchingDataset(Dataset):
    """Create local well/typewell window pairs for TVT-difference training."""

    def __init__(
            self, path=Path("../../rogii-wellbore-geology-prediction/train"), 
            well_wnd_size=100, 
            typewell_wnd_size=200, typewell_tvt_interval=0.5,
            ):
        """Initialize the dataset and precompute how many windows each well contributes."""
        self.rogi_dataset = RogiDataset(path)
        self.well_wnd_size = well_wnd_size
        self.typewell_wnd_size = typewell_wnd_size
        self.typewell_tvt_interval = typewell_tvt_interval
        self.typewell_wnd_tvt_range = self.typewell_tvt_interval*self.typewell_wnd_size/2

        well_num_idxs = []
        for i in tqdm(range(len(self.rogi_dataset))):
            w,t = self.rogi_dataset[i]
            well_num_idxs.append((~w.Known).sum()-(self.well_wnd_size-1))
        self.well_num_idxs = np.array(well_num_idxs, dtype=int)
        self.well_num_idxs_cumsum = self.well_num_idxs.cumsum()

    def __len__(self):
        """Return the total number of sliding windows across all wells."""
        return self.well_num_idxs.sum()
        
    def get_well_typewell_pair(self,idx) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load one raw well and its paired typewell by integer index."""
        return self.rogi_dataset[idx]
    
    def filter_srs(self, srs:pd.Series, wnd_size=3, mode="median") -> pd.Series:
        """Apply a light rolling filter while preserving the series length."""
        rll = srs.rolling(wnd_size, center=True,min_periods=1)
        if mode=="median":
            return rll.median()
        elif mode=="mean":
            return rll.mean()
        else:
            return srs
    
    def _process_typewell(self, typewell:pd.DataFrame)->pd.Series:
        typewell_gr = typewell.GR
        grid = np.arange(typewell_gr.index[0], typewell_gr.index[-1]+self.typewell_tvt_interval, self.typewell_tvt_interval)
        typewell_gr = pd.Series(
            np.interp(grid, typewell_gr.index.to_numpy(dtype=float), typewell_gr.to_numpy(dtype=float)),
            index=grid,
            name="GR"
            )
        typewell_gr = self.filter_srs(typewell_gr)
        return typewell_gr
    
    def _process_well(self, well:pd.DataFrame)->pd.DataFrame:
        """Filter the well GR and keep only the unknown interval used for training."""
        well = well.copy()
        well["GR"] = self.filter_srs(well.GR).interpolate()
        well = well.loc[~well.Known]
        return well
    
    def _get_well_selection(self,key):
        """Map a global sample index to a specific well and local window offset."""
        selected_well = int(np.searchsorted(self.well_num_idxs_cumsum, key, side="right"))
        selected_well_idx = int(key-self.well_num_idxs[:selected_well].sum())
        return selected_well, selected_well_idx

    def _get_typewell_window(self, typewell_gr: pd.Series, anchor_tvt: float) -> pd.Series:
        """Extract a fixed-width typewell window centered as closely as possible on the anchor TVT."""
        typewell_anchor = int(np.argmin(np.abs(typewell_gr.index - anchor_tvt)))
        half_window = self.typewell_wnd_size // 2
        start = min(max(typewell_anchor - half_window, 0), typewell_gr.shape[0] - self.typewell_wnd_size)
        stop = start + self.typewell_wnd_size
        return typewell_gr.iloc[start:stop]

    def __getitem__(self, key:int):
        """Return a well GR window, the aligned typewell window, and TVT offsets from the window start."""
        total_samples = len(self)
        if key<0:
            key = total_samples+key
        if key < 0 or key >= total_samples:
            raise IndexError("MatchingDataset index out of range")
        selected_well, selected_well_idx = self._get_well_selection(key)
        well, typewell = self.get_well_typewell_pair(selected_well)
        well, typewell_gr = self._process_well(well), self._process_typewell(typewell)
        well_wnd = well.iloc[selected_well_idx:selected_well_idx+self.well_wnd_size]
        anchor_tvt = well_wnd.TVT.iloc[0]
        typewell_wnd = self._get_typewell_window(typewell_gr, anchor_tvt)
        well_wnd_gr = well_wnd.GR
        tvt_diff = well_wnd.TVT - anchor_tvt
        return well_wnd_gr, typewell_wnd, tvt_diff
    
    
self = MatchingDataset()