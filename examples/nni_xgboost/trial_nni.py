from lib import *
from data_setup import *
import math
import nni

# function to fit the model and return the performance of the model
def return_model_assessment(params, X_train, y_train, X_test):
    global models, train_scores, test_scores, curr_model_hyper_params
    model = XGBClassifier(random_state=42, seed=42)
    model.set_params(**params)
    fitted_model = model.fit(X_train, y_train, sample_weight=None)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_score = f1_score(train_predictions, y_train)
    test_score = f1_score(test_predictions, y_test)
    return 1 - test_score

params = nni.get_next_parameter()

if type(params['max_depth']) == float:
    params['max_depth'] = math.floor(params['max_depth'])
if type(params['n_estimators']) == float:
    params['n_estimators'] = math.floor(params['n_estimators'])
    
score = return_model_assessment(params=params, X_train=X_train, y_train=y_train, X_test=X_test)
nni.report_final_result(score)