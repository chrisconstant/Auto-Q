from autorecipe.agentchat.RecipeAgent import RecipeAgent

obj = RecipeAgent(name='Maximo')
obj.set_asset_class(asset_class='wind turbine gearbox')
obj.init_chat(round=2)

# anomaly detection model
# failure event prediction
# asset health index model

#obj.init_chat(message='standby generator')
#obj.init_chat(message='air comporessor')
#obj.init_chat(message='hydroelectric power turbine')
#obj.init_chat(message='wind turbine')
#obj.init_chat(message='electrical transformer')
#obj.init_chat(message='induced draft fan')
#obj.init_chat(message='blast furnace')
#obj.init_chat(message='electric battery')
#obj.init_chat(message='industrial robot')

#obj.init_chat(message='The industrial asset class is air compressor')

#client.search_runs(experiment_ids=experiment.experiment_id,order_by=['start_time ASC'])
#ans = client.search_runs(experiment_ids=experiment.experiment_id)
#experiment = client.get_experiment(experiment.experiment_id)
#experiment = client.get_experiment_by_name("MyExperiment_b9540ebf-2570-4b2b-b3da-01e5a5b8b282")
#client = MlflowClient()
#from mlflow import MlflowClient
#obj.init_chat(message='The industrial asset class is Accumulator Hydraulic Bladder Type')
