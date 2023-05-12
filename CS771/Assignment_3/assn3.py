import numpy as np
import pickle as pkl

# Define your prediction method here
# df is a dataframe containing timestamps, weather data and potentials
# def my_predict( df ):
	
# 	# Load your model file
	
# 	# Make two sets of predictions, one for O3 and another for NO2
	
# 	# Return both sets of predictions
# 	return ( pred_o3, pred_no2 )

def my_predict( df ):
		with open( "final.pkl", "rb" ) as file:
			model = pkl.load( file )
	
		pred = model.predict( df.drop( [ "Time" ], axis = "columns" ).to_numpy() )
		return ( pred[ :, 0 ], pred[ :, 1 ] )