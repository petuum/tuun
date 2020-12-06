from lib import *
# parameters for the data
np.random.seed(42) # important if you want reproducible results
n_samples = 10000 # number of rows
n_features = 30 # number of columns excluding the binary label
n_informative = 10 # number of features that are actually useful
n_classes = 2 # for binary classification
class_weights = 0.6 # fraction of zeros
train_ratio = 0.7 # to split the data in two parts 
# generate the data
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_classes=n_classes, weights=[class_weights], random_state=42)
# split the data into train-test(straitified) and give names to columns
col_names = ['col_' + str(i + 1) for i in range(X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=1 - train_ratio, random_state=42)
target_col = "target"
df_train = pd.DataFrame(X_train, columns=col_names)
df_train.loc[:, target_col] = y_train
df_test = pd.DataFrame(X_test, columns=col_names)
df_test.loc[:, target_col] = y_test
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(y_test[:10])
