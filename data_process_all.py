import pandas as pd

month = "april"
df = pd.read_excel("datasets/data_" + month + ".xls" , sheet_name= 'breakouts')

#df_breakouts = pd.read_excel("data May 2010 - columns pruned and output added.xls" , sheet_name= 'breakouts')

#unwanted_cols_breakouts = ["date","time","type of break out"]

unwanted_cols = ["date","time","type of break out",'shift colour', 'LFC index',
       'water jacket ID north', 'water jacket ID south','type of SEN',
       'water jacket ID east', 'water jacket ID west','mould number', 'rest_standtijd_kms']

for name in unwanted_cols:
    #print(name)
    df.pop(name)


#for name in unwanted_cols_breakouts:
#    df_breakouts.pop(name)

col = 'steelgrade'
df.loc[df[col] == 'FN80', col] = 1
df.loc[df[col] == 'F12L', col] = 2
df.loc[df[col] == 'FN81', col] = 3
df.loc[df[col] == 'FV83', col] = 4
df.loc[df[col] == 'UT12', col] = 5
df.loc[df[col] == 'FV85', col] = 6
df.loc[df[col] == 'UT23', col] = 7
df.loc[df[col] == 'LGAB', col] = 8

df.loc[df["powder consumption"] == "Under Range", "powder consumption"] = 0
#df.loc[df["powder consumption"] == "Over Range", "powder consumption"] = 19.6215705871582*1.1

#df_breakouts.loc[df["powder consumption"] == "Under Range", "powder consumption"] = 0
#df_breakouts.loc[df["powder consumption"] == "Over Range", "powder consumption"] = 19.6215705871582*1.1

df.to_excel("datasets/filtered_data_breakouts_" + month + ".xls" , sheet_name = "breakouts", index=False)
#df_breakouts.to_excel("filtered_data_breakouts.xls" , sheet_name = "breakouts")
