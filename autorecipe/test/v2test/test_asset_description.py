import pandas as pd
from autorecipe.agentchat.asset_description import get_asset_description

df = pd.read_csv('../autoQ_val_data_input_for_experiments.csv')

cname = list(df['component_short_description'])
dname = list(df['component_boundry_for_short_description'])

for index in range(len(cname)):

    print ('only asset name ----------- ')
    asset_name = cname[index]
    ans = get_asset_description(iteration=2, asset_class=asset_name)
    print (ans)

    print ('only asset + boundry ----------- ')
    asset_name = cname[index] + ', with component boundry - ' + dname[index]
    ans = get_asset_description(iteration=2, asset_class=asset_name)
    print (ans)

    print ('******************* ------------------')