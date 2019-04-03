import pickle as pkl
import runpy

RETRAIN_PATH='script/receipts_model.py'
MODEL_PATH='bin/receipts_model.pkl'

class PriceModel():
	def load_model():
		obj = pkl.load(open(MODEL_PATH, 'rb'))
		print(obj)
		return (obj['model'], obj['score'],obj['error'])

	def retrain_model():
		runpy.run_path(RETRAIN_PATH)	
