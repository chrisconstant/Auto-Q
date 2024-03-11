from autorecipe.agentchat.RecipeAgent import RecipeAgent
import pandas as pd

gold_df = pd.read_csv("../autoQ_val_data_input_for_experiments.csv")
trial = list(gold_df["component_short_description"])
dname = list(gold_df['component_boundry_for_short_description'])

for index, item in enumerate(trial):
    if 'Brown EPR' not in item:
        continue
    try:
        asset_class = trial[index] + ', with component boundry - ' + dname[index]
    except:
        asset_class = trial[index]
    asset_class = asset_class.replace('<','-')
    asset_class = asset_class.replace('/', ' ').replace(',',' ')
    print ('processing ----' + asset_class)
    obj = RecipeAgent(name="Maximo")
    obj.set_asset_class(
        asset_class=asset_class,
    )
    obj.init_chat(round=0, add_description=True)
