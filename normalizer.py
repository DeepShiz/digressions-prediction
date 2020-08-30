import pandas as pd
import numpy as np


count=0

path_may_goodcasts = "datasets/filtered_data_goodcasts_May.xlsx"
path_may_breakouts = "datasets/filtered_data_breakouts_May.xlsx"
path_july_breakouts = "datasets/filtered_data_breakouts_july.xls"
path_aug_breakouts = "datasets/filtered_data_breakouts_aug.xls"
path_march_breakouts = "datasets/filtered_data_breakouts_march.xls"
path_april_breakouts = "datasets/filtered_data_breakouts_april.xls"

data_may_goodcasts = pd.read_excel(path_may_goodcasts)
data_may_breakouts = pd.read_excel(path_may_breakouts)
data_july_breakouts = pd.read_excel(path_july_breakouts)
data_aug_breakouts = pd.read_excel(path_aug_breakouts)
data_march_breakouts = pd.read_excel(path_march_breakouts)
data_april_breakouts = pd.read_excel(path_april_breakouts)

max_vals_goodcasts_may  =  data_may_goodcasts.max()
max_vals_breakouts_may =  data_may_breakouts.max()
#max_vals_goodcasts_june =  data_june_goodcasts.max()
#max_vals_breakouts_june =  data_june_breakouts.max()

min_vals_goodcasts_may  =  data_may_goodcasts.min()
min_vals_breakouts_may =  data_may_breakouts.min()
#min_vals_goodcasts_june =  data_june_goodcasts.min()
#min_vals_breakouts_june =  data_june_breakouts.min()


range_vals = []

max_val_feature = []
min_val_feature = []

for i in range(len(max_vals_breakouts_may)):

    max_val = max(max_vals_breakouts_may[i],max_vals_goodcasts_may[i])
    min_val = min(min_vals_breakouts_may[i],min_vals_goodcasts_may[i])

    range_vals.append(max_val-min_val)

    max_val_feature.append(max_val+ 0.2*(max_val-min_val))
    min_val_feature.append(min_val - 0.2*(max_val-min_val))

maximum = np.array(max_val_feature)
minimum = np.array(min_val_feature)


#applied_breakout_july = data_july_breakouts.apply(lambda x:(2*x -maximum-minimum)/(maximum-minimum),axis=1)
applied_breakout_march = data_march_breakouts.apply(lambda x:(2*x -maximum-minimum)/(maximum-minimum),axis=1)
applied_breakout_april = data_april_breakouts.apply(lambda x:(2*x -maximum-minimum)/(maximum-minimum),axis=1)



#applied_breakout_july.to_excel("normalized_breakouts_july.xls",index = False)
applied_breakout_aug.to_excel("normalized_breakouts_aug.xls",index = False)
