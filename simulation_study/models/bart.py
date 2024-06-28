import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/JakeColtman/bartpy.git", "--no-deps"])
from bartpy.sklearnmodel import SklearnModel

from IPython.utils import io
def predict_bart(X_train, Y_train, X_test, n_trees=50, n_chains=4, n_burn=20, n_samples=200):
    with io.capture_output() as captured:
        bart_model = SklearnModel(n_jobs=1, n_chains=n_chains, n_trees=n_trees, n_burn=n_burn, n_samples=n_samples)
        bart_model.fit(X_train, Y_train)
        bart_preds = bart_model.predict(X_test)
    return bart_preds