# GR Matching CNN

## Problem Framing

The task is a local signal-registration problem. For each short GR segment from the lateral well, we want to estimate where that segment sits relative to a local TVT anchor in the paired typewell.

The current notebook formulation does not train against a full TVT heatmap. Instead, it constructs paired local windows and predicts a continuous TVT offset sequence:

$$
\Delta \mathrm{TVT}_i = \mathrm{TVT}_i - \mathrm{TVT}_{\text{anchor}}
$$

where $\mathrm{TVT}_{\text{anchor}}$ is the first TVT value in the selected well window.

## Current Dataset Design

`MatchingDataset` turns each raw well/typewell pair into many local training samples.

For one sample:

- A fixed-width window is taken from the unknown part of the lateral well.
- The well GR is lightly filtered and interpolated.
- The typewell GR is resampled onto a uniform TVT grid using `np.interp`.
- A fixed-width typewell window is extracted around the nearest sampled TVT to the well-window anchor.
- The target is the per-step TVT difference relative to the window anchor.

This means each training example is local and translation-friendly: the model does not have to infer absolute TVT from scratch, only the offset of each MD step within the current registration window.

## Sample Structure

Each `__getitem__` currently returns three objects:

1. `well_wnd_gr`
    A 1D GR sequence from the unknown interval of the lateral well.
2. `typewell_wnd`
    A 1D GR sequence from the typewell, resampled to a uniform TVT spacing and centered near the local anchor.
3. `tvt_diff`
    A 1D target sequence of TVT offsets from the anchor TVT.

Conceptually:

$$
	ext{sample} = (\text{well GR window},\; \text{typewell GR window},\; \Delta \mathrm{TVT}\text{ sequence})
$$

## Preprocessing

### Well

- Use only the interval where `Known == False`.
- Apply a light rolling filter to GR.
- Interpolate remaining GR gaps.
- Slice fixed-width MD windows.

### Typewell

- Use the raw GR indexed by TVT.
- Resample onto a uniform TVT grid with spacing `typewell_tvt_interval`.
- Apply a light rolling filter after resampling.
- Slice a fixed-width window centered as closely as possible on the local anchor TVT.

## Windowing Logic

Let `typewell_anchor` be the nearest sampled typewell index to the anchor TVT. The typewell window is centered around that location and clamped so the slice stays within bounds while preserving fixed width.

This gives a stable training input shape:

- Well window length: `well_wnd_size`
- Typewell window length: `typewell_wnd_size`

## Training Implication

This README reflects the current notebook state: a local regression-style target based on `tvt_diff`, not the earlier heatmap/classification formulation.

The natural training setup from here is:

- model inputs: well GR window and typewell GR window
- model output: predicted TVT-offset sequence
- loss: regression loss such as L1, Huber, or MSE on `tvt_diff`