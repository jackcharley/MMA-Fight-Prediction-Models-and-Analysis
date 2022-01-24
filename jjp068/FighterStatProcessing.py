import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

"""
References:

Title: matplotlib
Author: matplotlib Team
Availability: https://github.com/matplotlib/matplotlib
Version: 3.4.2

Title: numpy
Author: numpy Team
Availability: https://github.com/numpy/numpy
Version: 1.19.5

Title: pandas
Author: pandas Team
Availability: https://github.com/pandas-dev/pandas
Version:  1.2.4

"""

desired_width = 320
pd.set_option('display.width',desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)

df = pd.read_csv('practiceallfighterinfo.csv')





# convert career avg stats into floats

df[['SLpM', 'Str. Acc.', 'SApM', 'Str. Def', 'TD Avg.','TD Acc.',
       'TD Def.', 'Sub. Avg.']] = df[['SLpM', 'Str. Acc.', 'SApM', 'Str. Def', 'TD Avg.','TD Acc.','TD Def.', 'Sub. Avg.']].apply(pd.to_numeric, errors = 'coerce')




#convert record to wins,loss columns
#wins
df['Wins'] = df.Record.str.extract('(\d+)')
df['Losses'] = df.Record.str.extract('(-\d+)')
df['Losses'] = df['Losses'].map(lambda x: x.lstrip('-'))



#reformat df
df.drop(['Record'], 1, inplace= True)
titles = list(df.columns)
df = df[titles[0:4]+[titles[-2]]+[titles[-1]]+titles[4:15]]
df[['Wins','Losses']] = df[['Wins','Losses']].apply(pd.to_numeric, errors = 'coerce')

# convert height from feet inches string to cm int
df['Feet'] = df.Height.str.extract('(\d)')
df['Inches'] = df.Height.str.extract("('\s\d)")
df['Inches'] = df.Inches.str.extract("(\d)")
df[['Feet','Inches']] = df[['Feet','Inches']].apply(pd.to_numeric, errors = 'coerce')
df['Height_cm'] = (df['Feet']*30.48)+(df['Inches']*2.54)

#reformat df
del df['Height']
del df['Feet']
del df['Inches']
titles = list(df.columns)
df = df[titles[0:3]+[titles[-1]]+titles[3:16]]

#convert reach into int datatype
df['Reach'] = df.Reach.str.extract('(\d{2})')
df['Reach'] = df['Reach'].apply(pd.to_numeric, errors = 'coerce')


#convert DOB into Year of B int
df['DOB'] = df.DOB.str.extract('(\d{4})')
df['DOB'] = df['DOB'].apply(pd.to_numeric, errors = 'coerce')

# remove any records with 0 for all stats
df= df.loc[((df['SLpM']!= 0)|(df['Str. Acc.']!=0)| (df['SApM']!= 0) | (df['Str. Def']!=0) | (df['TD Avg.']!= 0) | (df['TD Acc.']!=0) | (df['TD Def.']!= 0)| (df['Sub. Avg.']!=0))]

# deleted  weight and set name as index

df.drop(['Weight','Fighter_iD'], 1,inplace = True)
df.set_index('Name', inplace = True)



# incorporate catgories with a hanful of rows into larger categories and name the the missing values 'unk' = unknown
df['STANCE'].replace({'Open Stance':'Orthodox', 'Sideways':'Orthodox'}, inplace= True)
df['STANCE'].fillna('unk',inplace=True)


#impute missing numeric values


#1% of records were missing height value - just removed those rows
df.dropna(axis = 0 , subset = ['Height_cm'], inplace = True)


#dealing with outliers
max_threshold = df['SLpM'].quantile(0.995)
df = df[(df.SLpM<max_threshold)]
df = df[(df['Str. Acc.']<df['Str. Acc.'].quantile(0.985)) & (df['Str. Acc.']>df['Str. Acc.'].quantile(0.03))]
df = df[(df['SApM']<df['SApM'].quantile(0.995)) & (df['SApM']>df['SApM'].quantile(0.005))]
df = df[(df['Str. Def']<df['Str. Def'].quantile(0.9955)) & (df['Str. Def']>df['Str. Def'].quantile(0.005))]
df = df[(df['TD Avg.']<=df['TD Avg.'].quantile(0.98))]
df = df[(df['TD Acc.']<df['TD Acc.'].quantile(0.93))]
df = df[(df['Sub. Avg.']<df['Sub. Avg.'].quantile(0.995))]
df = df[(df['Wins']<df['Wins'].quantile(0.99))]
df = df[(df['Losses']<df['Losses'].quantile(0.99))]


#Now the matchup info

matches_frame = pd.read_csv('newfightinfo.csv')

#remove useless rows
matches_frame = matches_frame.drop(['F1_Kd','F2_Kd','F1_Str','F2_Str','F1_Td','F2_Td','F1_Sub','F2_Sub','Time'], axis=1)
matches_frame.rename( columns = {'Unnamed: 0': 'matchid'}, inplace = True)
matches_frame.set_index('matchid', inplace= True)


#remove all nc and draws
matches_frame = matches_frame.loc[matches_frame['W/L']=='win']



#neaten up columns
matches_frame = matches_frame.drop('W/L',1)
matches_frame['Method'].replace({ 'U-DEC':'DEC', 'S-DEC':'DEC', 'M-DEC':'DEC','DQ':'DEC'}, inplace = True)

# due to the formatting of UFCstats.com Fighter 1 is always the winner, need to change the columns and values so fighter 2
#is winner for same ammount of fights

"""
combining data frames, the code was take from below source
Title: preprocessing.ipynb
Author: Yuan Tian
Date: 15/09/2020
Code Version : 1.0
Availability: https://github.com/naity/DeepUFC2/blob/master/preprocessing.ipynb
"""

matches_frame['winner'] = matches_frame['Fighter1']
swap_columns = np.random.choice(len(matches_frame), size  = len(matches_frame)//2, replace = False)
matches_frame.iloc[swap_columns,[0,1]]= matches_frame.iloc[swap_columns,[1,0]].values
matches_frame['winner'] = matches_frame['winner']==matches_frame['Fighter1']
matches_frame['winner']= matches_frame['winner'].astype(int)

# merging the dataframes

# checking if fights have fighters, if they dont remove corresponding match
fighters_list = df.index.tolist()
matches_mod = matches_frame.loc[(matches_frame['Fighter1'].isin(fighters_list)) &
                            (matches_frame['Fighter2'].isin(fighters_list))]

# reset index
matches_mod.reset_index(inplace = True, drop = True)
fighter1_stats = df.loc[matches_mod['Fighter1']]
fighter1_stats = fighter1_stats.add_prefix('f1_')
fighter2_stats = df.loc[matches_mod['Fighter2']]
fighter2_stats = fighter2_stats.add_prefix('f2_')

fighter1_stats.reset_index(drop = True , inplace= True)
fighter2_stats.reset_index(drop = True , inplace= True)



all_stats_combined = pd. concat([matches_mod,fighter1_stats,fighter2_stats],axis=1, sort = False)
all_stats_combined = all_stats_combined[all_stats_combined['Fighter1'].notna()]


#encode categoricals using one hot encoding
df_encoded = pd.get_dummies(all_stats_combined, columns=["Method","f1_STANCE","f2_STANCE"])

df_encoded[['Method_DEC','Method_KO/TKO','Method_SUB','f1_STANCE_Orthodox','f1_STANCE_Southpaw','f1_STANCE_Switch','f1_STANCE_unk','f2_STANCE_Orthodox','f2_STANCE_Southpaw','f2_STANCE_Switch','f2_STANCE_unk']]=df_encoded[['Method_DEC','Method_KO/TKO','Method_SUB','f1_STANCE_Orthodox','f1_STANCE_Southpaw','f1_STANCE_Switch','f1_STANCE_unk','f2_STANCE_Orthodox','f2_STANCE_Southpaw','f2_STANCE_Switch','f2_STANCE_unk']].apply(pd.to_numeric,errors = 'coerce')

mapping = {"Women's Strawweight": 0, "Women's Flyweight":1, "Women's Bantamweight":2, "Women's Featherweight":3, 'Flyweight':4, 'Bantamweight':5, 'Featherweight':6, 'Lightweight':7, 'Welterweight':8, 'Middleweight':9, 'Light Heavyweight':10, 'Heavyweight':11, 'Openweight':12}
df_encoded["Weight class"] = df_encoded["Weight class"].map(mapping)

print(df_encoded.columns)

#drop names as they only really serve as a fight id
df_encoded.drop(df_encoded.columns[[0,1]], axis=1, inplace=True)


#impute height and reach
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_encoded = pd.DataFrame(imputer.fit_transform(df_encoded),columns = df_encoded.columns)

print(df_encoded.isna().sum())

df_encoded = df_encoded.round(decimals=2)

columns = list(df_encoded.columns.values)
columns.pop(columns.index('winner'))
df_encoded = df_encoded[columns+['winner']]
print(df_encoded.columns)

#export file to csv for importing in pandas
df_encoded.to_csv('final_data.csv')


