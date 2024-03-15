from autorecipe.agentchat.RecipeAgent import RecipeAgent
import pandas as pd

gold_df = pd.read_csv("../autoQ_val_data_input_for_experiments.csv")
trial = list(gold_df["component_short_description"])

for item in trial:
    print ('processing ----' + item)
    item = item.replace('<','-')
    asset_class = item.replace('/', ' ').replace(',',' ')
    obj = RecipeAgent(name="Maximo", model_id=1)
    obj.set_asset_class(
        asset_class=asset_class,
    )
    obj.init_chat(round=0, add_description=True)