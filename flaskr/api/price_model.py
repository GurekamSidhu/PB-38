import pickle as pkl
import runpy

RETRAIN_PATH='script/receipts_model.py'
MODEL_PATH='bin/receipts_model.pkl'

class PriceModel():
	def load_model():
		return pkl.load(open(MODEL_PATH, 'rb'))

	def retrain_model():
		runpy.run_path(RETRAIN_PATH)	
