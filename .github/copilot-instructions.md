# Kaggle ROGI Competition

We are doing the kaggle rogii competition.


The goal, is to predict the TVT along every foot of MD in the lateral of 700 well boreholes. In this competition, they are using a different definition of TVT, it is essentially stratigraphic depth, not a thickness at all. The grading metric is RMSE on the predicted TVT.

## Data 
The data, is 700 borehole trajectory (X,Y,Z), along with GR logs at each position, and the formation height of a few formations, at every X and Y borehole coordinate combination (you only get formation height for the path the borehole takes). For each well, they also provide a typewell, which shows the expected GR reading for all TVT values.

