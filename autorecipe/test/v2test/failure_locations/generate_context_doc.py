from autorecipe.agentchat.RecipeAgent import RecipeAgent
import pandas as pd

# for selection, models name : lamma, mystral, Granite
model_ids = [1, 2, 3]

# round : context only = 0, 1 meeting = round 1, ...
round = 0

# asset name and/or component
add_boundry = True

# read the input file
gold_df = pd.read_csv("./autoQ_val_data_input_for_experiments.csv")
trial = list(gold_df["component_short_description"])
dname = list(gold_df['component_boundry_for_short_description'])

# for each entry
for index, item in enumerate(trial):

    if add_boundry:
        # merge the boundry part
        try:
            asset_class = trial[index] + ', with component boundry - ' + dname[index]
        except:
            asset_class = trial[index]
    else:
        asset_class = trial[index]

    # adjust the asset name for file storage
    asset_class = asset_class.replace('<','-')
    asset_class = asset_class.replace('/', ' ').replace(',',' ')
    print ('processing ----' + asset_class)

    # calling the RecipeAgent with 
    obj = RecipeAgent(name="Maximo", model_id=1)
    obj.set_asset_class(
        asset_class=asset_class,
    )
    obj.init_chat(round=0, add_description=True)
