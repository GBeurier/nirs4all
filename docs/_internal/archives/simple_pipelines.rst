Simple pipeline
===============

You can execute this notebook at:
`Google collab notebook <https://colab.research.google.com/github/GBeurier/nirs4all/blob/main/examples/simple_pipelines.ipynb>`_


.. container:: cell code

   .. code:: python

      !pip install nirs4all
      !pip install scikeras

   .. container:: output stream stdout

      ::

         Collecting nirs4all
           Downloading nirs4all-0.9.5-py3-none-any.whl (38 kB)
         Requirement already satisfied: pandas in c:\workspace\ml\pynirs_env\lib\site-packages (from nirs4all) (1.3.5)
         Requirement already satisfied: scipy in c:\workspace\ml\pynirs_env\lib\site-packages (from nirs4all) (1.7.3)
         Requirement already satisfied: tensorflow in c:\workspace\ml\pynirs_env\lib\site-packages (from nirs4all) (2.8.0)
         Requirement already satisfied: PyWavelets in c:\workspace\ml\pynirs_env\lib\site-packages (from nirs4all) (1.3.0)
         Requirement already satisfied: scikit-learn in c:\workspace\ml\pynirs_env\lib\site-packages (from nirs4all) (1.0.2)
         Requirement already satisfied: numpy in c:\workspace\ml\pynirs_env\lib\site-packages (from nirs4all) (1.21.6)
         Requirement already satisfied: pytz>=2017.3 in c:\workspace\ml\pynirs_env\lib\site-packages (from pandas->nirs4all) (2022.1)
         Requirement already satisfied: python-dateutil>=2.7.3 in c:\workspace\ml\pynirs_env\lib\site-packages (from pandas->nirs4all) (2.8.2)
         Requirement already satisfied: joblib>=0.11 in c:\workspace\ml\pynirs_env\lib\site-packages (from scikit-learn->nirs4all) (1.1.0)
         Requirement already satisfied: threadpoolctl>=2.0.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from scikit-learn->nirs4all) (3.1.0)
         Requirement already satisfied: absl-py>=0.4.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (1.0.0)
         Requirement already satisfied: keras<2.9,>=2.8.0rc0 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (2.8.0)
         Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (0.24.0)
         Requirement already satisfied: gast>=0.2.1 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (0.5.3)
         Requirement already satisfied: termcolor>=1.1.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (1.1.0)
         Requirement already satisfied: tf-estimator-nightly==2.8.0.dev2021122109 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (2.8.0.dev2021122109)
         Requirement already satisfied: h5py>=2.9.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (3.6.0)
         Requirement already satisfied: flatbuffers>=1.12 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (2.0)
         Requirement already satisfied: google-pasta>=0.1.1 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (0.2.0)
         Requirement already satisfied: protobuf>=3.9.2 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (3.20.0)
         Requirement already satisfied: tensorboard<2.9,>=2.8 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (2.8.0)
         Requirement already satisfied: setuptools in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (65.5.0)
         Requirement already satisfied: typing-extensions>=3.6.6 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (4.1.1)
         Requirement already satisfied: libclang>=9.0.1 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (13.0.0)
         Requirement already satisfied: six>=1.12.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (1.16.0)
         Requirement already satisfied: keras-preprocessing>=1.1.1 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (1.1.2)
         Requirement already satisfied: opt-einsum>=2.3.2 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (3.3.0)
         Requirement already satisfied: astunparse>=1.6.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (1.6.3)
         Requirement already satisfied: wrapt>=1.11.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (1.14.0)
         Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorflow->nirs4all) (1.44.0)
         Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from astunparse>=1.6.0->tensorflow->nirs4all) (0.37.1)
         Requirement already satisfied: cached-property in c:\workspace\ml\pynirs_env\lib\site-packages (from h5py>=2.9.0->tensorflow->nirs4all) (1.5.2)
         Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorboard<2.9,>=2.8->tensorflow->nirs4all) (0.4.6)
         Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorboard<2.9,>=2.8->tensorflow->nirs4all) (0.6.1)
         Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorboard<2.9,>=2.8->tensorflow->nirs4all) (1.8.1)
         Requirement already satisfied: werkzeug>=0.11.15 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorboard<2.9,>=2.8->tensorflow->nirs4all) (2.1.1)
         Requirement already satisfied: requests<3,>=2.21.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorboard<2.9,>=2.8->tensorflow->nirs4all) (2.27.1)
         Requirement already satisfied: google-auth<3,>=1.6.3 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorboard<2.9,>=2.8->tensorflow->nirs4all) (2.6.4)
         Requirement already satisfied: markdown>=2.6.8 in c:\workspace\ml\pynirs_env\lib\site-packages (from tensorboard<2.9,>=2.8->tensorflow->nirs4all) (3.3.6)
         Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow->nirs4all) (5.0.0)
         Requirement already satisfied: rsa<5,>=3.1.4 in c:\workspace\ml\pynirs_env\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow->nirs4all) (4.8)
         Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\workspace\ml\pynirs_env\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow->nirs4all) (0.2.8)
         Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow->nirs4all) (1.3.1)
         Requirement already satisfied: importlib-metadata>=4.4 in c:\workspace\ml\pynirs_env\lib\site-packages (from markdown>=2.6.8->tensorboard<2.9,>=2.8->tensorflow->nirs4all) (4.11.3)
         Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\workspace\ml\pynirs_env\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow->nirs4all) (1.26.9)
         Requirement already satisfied: certifi>=2017.4.17 in c:\workspace\ml\pynirs_env\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow->nirs4all) (2021.10.8)
         Requirement already satisfied: idna<4,>=2.5 in c:\workspace\ml\pynirs_env\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow->nirs4all) (3.3)
         Requirement already satisfied: charset-normalizer~=2.0.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow->nirs4all) (2.0.12)
         Requirement already satisfied: zipp>=0.5 in c:\workspace\ml\pynirs_env\lib\site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.9,>=2.8->tensorflow->nirs4all) (3.8.0)
         Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\workspace\ml\pynirs_env\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow->nirs4all) (0.4.8)
         Requirement already satisfied: oauthlib>=3.0.0 in c:\workspace\ml\pynirs_env\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow->nirs4all) (3.2.0)
         Installing collected packages: nirs4all
         Successfully installed nirs4all-0.9.5

   .. container:: output stream stderr

      ::


         [notice] A new release of pip available: 22.3 -> 22.3.1
         [notice] To update, run: python.exe -m pip install --upgrade pip

.. container:: cell markdown

   .. rubric:: Load data
      :name: load-data

   -  initialize random variables
   -  create a data set to train the pipelines

.. container:: cell code

   .. code:: python

      from nirs4all import utils
      from nirs4all.model_selection import train_test_split_idx
      from sklearn.model_selection import train_test_split
      import numpy as np

      # Init basic random
      rd_seed = 42
      np.random.seed(rd_seed)

      xcal_csv = "https://raw.githubusercontent.com/GBeurier/nirs4all/main/examples/Xcal.csv"
      ycal_csv = "https://raw.githubusercontent.com/GBeurier/nirs4all/main/examples/Ycal.csv"

      # Create a set named data
      x, y = utils.load_csv(xcal_csv, ycal_csv, x_hdr=0, y_hdr=0, remove_na=True)
      train_index, test_index = train_test_split_idx(x, y=y, method="random", test_size=0.25, random_state=rd_seed)
      X_train, y_train, X_test, y_test = x[train_index], y[train_index], x[test_index], y[test_index]
      print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

   .. container:: output stream stdout

      ::

         (270, 2151) (270,) (91, 2151) (91,)

.. container:: cell markdown

   .. rubric:: Declare preprocessing operators
      :name: declare-preprocessing-operators

   Here we declare the list of preprocessings that will be applied
   either in FeatureUnion or FeatureAugmentation.

.. container:: cell code

   .. code:: python

      from nirs4all import preprocessing as pp
      from sklearn.pipeline import Pipeline

      ### Declare preprocessing pipeline components
      preprocessing = [   ('id', pp.IdentityTransformer()),
                          ('savgol', pp.SavitzkyGolay()),
                          ('gaussian1', pp.Gaussian(order = 1, sigma = 2)),
                      ]

.. container:: cell markdown

   .. rubric:: Simple PLS regression
      :name: simple-pls-regression

   Here we create a pipeline with a FeatureUnion preprocessing. Then we
   train the pipeline and display results

.. container:: cell code

   .. code:: python

      from sklearn.pipeline import FeatureUnion
      from sklearn.preprocessing import MinMaxScaler
      from sklearn.compose import TransformedTargetRegressor
      from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
      from sklearn.cross_decomposition import PLSRegression

      # Simple PLS pipeline declaration
      pipeline = Pipeline([
          ('scaler', MinMaxScaler()), 
          ('preprocessing', FeatureUnion(preprocessing)), 
          ('pls', PLSRegression(n_components=10))
      ])

      # TransformedTargetRegressor is used to apply scaling to Y within the pipeline
      estimator = TransformedTargetRegressor(regressor = pipeline, transformer = MinMaxScaler())

      # # Train the model
      estimator.fit(X_train, y_train)

      # # Compute metrics on the test set
      Y_preds = estimator.predict(X_test)

      print("MAE", mean_absolute_error(y_test, Y_preds))
      print("MSE", mean_squared_error(y_test, Y_preds))
      print("MAPE", mean_absolute_percentage_error(y_test, Y_preds))
      print("R²", r2_score(y_test, Y_preds))
      # print(estimator.get_params())

   .. container:: output stream stdout

      ::

         MAE 1.1567468539928398
         MSE 2.5117852372966643
         MAPE 0.0254993046121325
         R² 0.7367788435855969

.. container:: cell markdown

   .. rubric:: Same pipeline with a XGBoost estimator
      :name: same-pipeline-with-a-xgboost-estimator

.. container:: cell code

   .. code:: python

      from xgboost import XGBRegressor

      x_pipeline = Pipeline([
          ('scaler', MinMaxScaler()), 
          ('preprocessing', FeatureUnion(preprocessing)), 
          ('XGB', XGBRegressor(n_estimators=50, max_depth=10))
      ])

      x_estimator = TransformedTargetRegressor(regressor = x_pipeline, transformer = MinMaxScaler())

      x_estimator.fit(X_train, y_train)

      Y_preds = x_estimator.predict(X_test)

      print("MAE", mean_absolute_error(y_test, Y_preds))
      print("MSE", mean_squared_error(y_test, Y_preds))
      print("MAPE", mean_absolute_percentage_error(y_test, Y_preds))
      print("R²", r2_score(y_test, Y_preds))

      # print(x_estimator.get_params())

   .. container:: output stream stdout

      ::

         MAE 1.2912766
         MSE 3.8444414
         MAPE 0.028730346
         R² 0.5971238611860624

.. container:: cell markdown

   .. rubric:: Same pipeline with simple KerasRegressor
      :name: same-pipeline-with-simple-kerasregressor

   *A more detailed and complete example is provided in
   keras_regressor.ipynb.*

.. container:: cell code

   .. code:: python

      from nirs4all.sklearn import FeatureAugmentation

      from tensorflow.keras.models import Sequential
      from tensorflow.keras.layers import Dense, Conv1D, SpatialDropout1D,BatchNormalization,Flatten, Dropout, Input

      from scikeras.wrappers import KerasRegressor

      from typing import Dict, Iterable, Any

      # Create the keras model in the scikeras wrapper format (meta arg)
      def keras_model(meta: Dict[str, Any]):
          input_shape = meta["X_shape_"][1:]
          model = Sequential()
          model.add(Input(shape=input_shape))
          model.add(SpatialDropout1D(0.08))
          model.add(Conv1D (filters=8, kernel_size=15, strides=5, activation='selu'))
          model.add(Dropout(0.2))
          model.add(Conv1D (filters=64, kernel_size=21, strides=3, activation='relu'))
          model.add(BatchNormalization())
          model.add(Conv1D (filters=32, kernel_size=5, strides=3, activation='elu'))
          model.add(BatchNormalization())
          model.add(Flatten())
          model.add(Dense(16, activation='sigmoid'))
          model.add(Dense(1, activation='sigmoid'))
          model.compile(loss = 'mean_squared_error', optimizer = 'adam')
          model.summary()
          return model

      # Create the KerasRegressor
      k_regressor = KerasRegressor(model = keras_model,
                                  epochs=400, 
                                  fit__batch_size=50,
                                  fit__validation_split=0.2,
                                  verbose = 2)

      # Declare the pipeline with a FeatureAugmentation (2D)
      k_pipeline = Pipeline([
          ('scaler', MinMaxScaler()), 
          ('preprocessing', FeatureAugmentation(preprocessing)),
          ('KerasNN', k_regressor)
      ])

      # Train and predict same as usual
      k_estimator = TransformedTargetRegressor(regressor = k_pipeline, transformer = MinMaxScaler())
       
      k_estimator.fit(X_train, y_train)
      # print(k_estimator.regressor_[2].history_)

      Y_preds = k_estimator.predict(X_test)

      print("MAE", mean_absolute_error(y_test, Y_preds))
      print("MSE", mean_squared_error(y_test, Y_preds))
      print("MAPE", mean_absolute_percentage_error(y_test, Y_preds))
      print("R²", r2_score(y_test, Y_preds))

   .. container:: output stream stdout

      ::

         Model: "sequential"
         _________________________________________________________________
          Layer (type)                Output Shape              Param #   
         =================================================================
          spatial_dropout1d (SpatialD  (None, 2151, 3)          0         
          ropout1D)                                                       
                                                                          
          conv1d (Conv1D)             (None, 428, 8)            368       
                                                                          
          dropout (Dropout)           (None, 428, 8)            0         
                                                                          
          conv1d_1 (Conv1D)           (None, 136, 64)           10816     
                                                                          
          batch_normalization (BatchN  (None, 136, 64)          256       
          ormalization)                                                   
                                                                          
          conv1d_2 (Conv1D)           (None, 44, 32)            10272     
                                                                          
          batch_normalization_1 (Batc  (None, 44, 32)           128       
          hNormalization)                                                 
                                                                          
          flatten (Flatten)           (None, 1408)              0         
                                                                          
          dense (Dense)               (None, 16)                22544     
                                                                          
          dense_1 (Dense)             (None, 1)                 17        
                                                                          
         =================================================================
         Total params: 44,401
         Trainable params: 44,209
         Non-trainable params: 192
         _________________________________________________________________
         Epoch 1/400
         5/5 - 19s - loss: 0.0294 - val_loss: 0.0220 - 19s/epoch - 4s/step
         Epoch 2/400
         5/5 - 0s - loss: 0.0266 - val_loss: 0.0192 - 197ms/epoch - 39ms/step
         Epoch 3/400
         5/5 - 0s - loss: 0.0218 - val_loss: 0.0190 - 157ms/epoch - 31ms/step
         Epoch 4/400
         5/5 - 0s - loss: 0.0213 - val_loss: 0.0189 - 145ms/epoch - 29ms/step
         Epoch 5/400
         5/5 - 0s - loss: 0.0208 - val_loss: 0.0176 - 147ms/epoch - 29ms/step
         Epoch 6/400
         5/5 - 0s - loss: 0.0210 - val_loss: 0.0193 - 147ms/epoch - 29ms/step
         Epoch 7/400
         5/5 - 0s - loss: 0.0179 - val_loss: 0.0182 - 145ms/epoch - 29ms/step
         Epoch 8/400
         5/5 - 0s - loss: 0.0164 - val_loss: 0.0184 - 147ms/epoch - 29ms/step
         Epoch 9/400
         5/5 - 0s - loss: 0.0174 - val_loss: 0.0204 - 165ms/epoch - 33ms/step
         Epoch 10/400
         5/5 - 0s - loss: 0.0162 - val_loss: 0.0175 - 147ms/epoch - 29ms/step
         Epoch 11/400
         5/5 - 0s - loss: 0.0156 - val_loss: 0.0177 - 161ms/epoch - 32ms/step
         Epoch 12/400
         5/5 - 0s - loss: 0.0139 - val_loss: 0.0185 - 144ms/epoch - 29ms/step
         Epoch 13/400
         5/5 - 0s - loss: 0.0152 - val_loss: 0.0159 - 147ms/epoch - 29ms/step
         Epoch 14/400
         5/5 - 0s - loss: 0.0141 - val_loss: 0.0164 - 163ms/epoch - 33ms/step
         Epoch 15/400
         5/5 - 0s - loss: 0.0137 - val_loss: 0.0162 - 164ms/epoch - 33ms/step
         Epoch 16/400
         5/5 - 0s - loss: 0.0127 - val_loss: 0.0143 - 147ms/epoch - 29ms/step
         Epoch 17/400
         5/5 - 0s - loss: 0.0142 - val_loss: 0.0141 - 146ms/epoch - 29ms/step
         Epoch 18/400
         5/5 - 0s - loss: 0.0136 - val_loss: 0.0150 - 147ms/epoch - 29ms/step
         Epoch 19/400
         5/5 - 0s - loss: 0.0109 - val_loss: 0.0134 - 145ms/epoch - 29ms/step
         Epoch 20/400
         5/5 - 0s - loss: 0.0105 - val_loss: 0.0140 - 161ms/epoch - 32ms/step
         ...
         Epoch 400/400
         5/5 - 0s - loss: 0.0012 - val_loss: 0.0132 - 143ms/epoch - 29ms/step
         3/3 - 0s - 361ms/epoch - 120ms/step
         MAE 1.4564642
         MSE 3.9834785
         MAPE 0.031864382
         R² 0.5825535320021759

.. container:: cell markdown

   .. rubric:: Save and load pipeline
      :name: save-and-load-pipeline

   There is two ways to save a pipeline using either pickle or joblib.
   If a KerasRegressor is used only the pickle method works.

.. container:: cell code

   .. code:: python

      import joblib
      import pickle

      # save xgb estimator
      Y_preds = estimator.predict(X_test)
      print("R²", r2_score(y_test, Y_preds))
      joblib.dump(estimator, 'xgb_estimator.pkl')

      # load xgb estimator
      loaded_estimator = joblib.load('xgb_estimator.pkl')
      Y_preds = loaded_estimator.predict(X_test)
      print("R²", r2_score(y_test, Y_preds))

      # save keras estimator
      Y_preds = k_estimator.predict(X_test)
      print("R²", r2_score(y_test, Y_preds))
      with open('keras_estimator.pickle', 'wb') as handle:
          pickle.dump(k_estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
      # byte_model = pickle.dumps(estimator, 'xgb_estimator.pkl')

      # load keras estimator
      with open('keras_estimator.pickle', 'rb') as handle:
          loaded_estimator = pickle.load(handle)
      # loaded_estimator = pickle.loads(bytes_model)

      Y_preds = loaded_estimator.predict(X_test)
      print("R²", r2_score(y_test, Y_preds))

   .. container:: output stream stdout

      ::

         R² 0.7367788435855969
         R² 0.7367788435855969
         3/3 - 0s - 60ms/epoch - 20ms/step
         R² 0.5825535320021759

   .. container:: output stream stderr

      ::

         WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 3 of 3). These functions will not be directly callable after loading.

   .. container:: output stream stdout

      ::

         INFO:tensorflow:Assets written to: C:\Users\grego\AppData\Local\Temp\tmp53x6_gz7\assets

   .. container:: output stream stderr

      ::

         INFO:tensorflow:Assets written to: C:\Users\grego\AppData\Local\Temp\tmp53x6_gz7\assets

   .. container:: output stream stdout

      ::

         3/3 - 0s - 254ms/epoch - 85ms/step
         R² 0.5825535320021759

.. container:: cell markdown

   .. rubric:: Simple cross validation with sklearn
      :name: simple-cross-validation-with-sklearn

.. container:: cell code

   .. code:: python

      from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate

      print("CV_scores", cross_val_score(estimator, x, y, cv=3))
      print("-- CV predict --")
      Y_preds = cross_val_predict(estimator, x, y, cv=3)
      print("MAE", mean_absolute_error(y, Y_preds))
      print("MSE", mean_squared_error(y, Y_preds))
      print("MAPE", mean_absolute_percentage_error(y, Y_preds))
      print("R²", r2_score(y, Y_preds))

      print("-- Cross Validate --")
      cv_results = cross_validate(estimator, x, y, cv=3, return_train_score=True, n_jobs=3)
      for key in cv_results.keys():
          print(key, cv_results[key])

   .. container:: output error

      ::

         ---------------------------------------------------------------------------
         NameError                                 Traceback (most recent call last)
         ~\AppData\Local\Temp/ipykernel_23356/3318979689.py in <module>
               1 from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
               2 
         ----> 3 print("CV_scores", cross_val_score(estimator, X, y, cv=3))
               4 print("-- CV predict --")
               5 Y_preds = cross_val_predict(estimator, X, y, cv=3)

         NameError: name 'X' is not defined

.. container:: cell markdown

   .. rubric:: Same with Keras Regressor
      :name: same-with-keras-regressor

   -  n_jobs parameter seems to deactivate gpu in tensorflow \* The CV
      do not take the best model but the last model. A better way would
      be to handle cv set by hand and to score the best model per fold.
      (see keras_regressor)

.. container:: cell code

   .. code:: python

      print("-- Cross Validate --")
      cv_results = cross_validate(k_estimator, x, y, cv=3, return_train_score=True)
      for key in cv_results.keys():
          print(key, cv_results[key])

   .. container:: output stream stdout

      ::

         -- Cross Validate --
         Model: "sequential_1"
         _________________________________________________________________
          Layer (type)                Output Shape              Param #   
         =================================================================
          spatial_dropout1d_1 (Spatia  (None, 2151, 9)          0         
          lDropout1D)                                                     
                                                                          
          conv1d_3 (Conv1D)           (None, 428, 8)            1088      
                                                                          
          dropout_1 (Dropout)         (None, 428, 8)            0         
                                                                          
          conv1d_4 (Conv1D)           (None, 136, 64)           10816     
                                                                          
          batch_normalization_2 (Batc  (None, 136, 64)          256       
          hNormalization)                                                 
                                                                          
          conv1d_5 (Conv1D)           (None, 44, 32)            10272     
                                                                          
          batch_normalization_3 (Batc  (None, 44, 32)           128       
          hNormalization)                                                 
                                                                          
          flatten_1 (Flatten)         (None, 1408)              0         
                                                                          
          dense_2 (Dense)             (None, 16)                22544     
                                                                          
          dense_3 (Dense)             (None, 1)                 17        
                                                                          
         =================================================================
         Total params: 45,121
         Trainable params: 44,929
         Non-trainable params: 192
         _________________________________________________________________
         Epoch 1/400
         4/4 - 3s - loss: 0.0364 - val_loss: 0.0542 - 3s/epoch - 679ms/step
         Epoch 2/400
         4/4 - 0s - loss: 0.0246 - val_loss: 0.0490 - 129ms/epoch - 32ms/step
         Epoch 3/400
         4/4 - 0s - loss: 0.0213 - val_loss: 0.0432 - 116ms/epoch - 29ms/step
         Epoch 4/400
         4/4 - 0s - loss: 0.0191 - val_loss: 0.0467 - 126ms/epoch - 31ms/step
         Epoch 5/400
         4/4 - 0s - loss: 0.0208 - val_loss: 0.0308 - 115ms/epoch - 29ms/step
         Epoch 6/400
         4/4 - 0s - loss: 0.0168 - val_loss: 0.0369 - 114ms/epoch - 29ms/step
         Epoch 7/400
         4/4 - 0s - loss: 0.0155 - val_loss: 0.0361 - 122ms/epoch - 31ms/step
         Epoch 8/400
         4/4 - 0s - loss: 0.0155 - val_loss: 0.0292 - 119ms/epoch - 30ms/step
         Epoch 9/400
         4/4 - 0s - loss: 0.0140 - val_loss: 0.0319 - 121ms/epoch - 30ms/step
         Epoch 10/400
         4/4 - 0s - loss: 0.0137 - val_loss: 0.0277 - 127ms/epoch - 32ms/step
         Epoch 11/400
         4/4 - 0s - loss: 0.0134 - val_loss: 0.0313 - 119ms/epoch - 30ms/step
         Epoch 12/400
         4/4 - 0s - loss: 0.0130 - val_loss: 0.0346 - 117ms/epoch - 29ms/step
         Epoch 13/400
         4/4 - 0s - loss: 0.0123 - val_loss: 0.0248 - 122ms/epoch - 31ms/step
         Epoch 14/400
         4/4 - 0s - loss: 0.0123 - val_loss: 0.0289 - 121ms/epoch - 30ms/step
         Epoch 15/400
         4/4 - 0s - loss: 0.0122 - val_loss: 0.0331 - 114ms/epoch - 29ms/step
         Epoch 16/400
         4/4 - 0s - loss: 0.0127 - val_loss: 0.0232 - 130ms/epoch - 33ms/step
         Epoch 17/400
         4/4 - 0s - loss: 0.0119 - val_loss: 0.0296 - 128ms/epoch - 32ms/step
         Epoch 18/400
         4/4 - 0s - loss: 0.0117 - val_loss: 0.0258 - 134ms/epoch - 33ms/step
         Epoch 19/400
         4/4 - 0s - loss: 0.0116 - val_loss: 0.0249 - 129ms/epoch - 32ms/step
         Epoch 20/400
         4/4 - 0s - loss: 0.0106 - val_loss: 0.0288 - 130ms/epoch - 33ms/step
         ...
         Epoch 400/400
         4/4 - 0s - loss: 4.0711e-04 - val_loss: 0.0043 - 142ms/epoch - 36ms/step
         TRANSFORM
         ahahah (121, 2151, 9)
         4/4 - 0s - 401ms/epoch - 100ms/step
         TRANSFORM
         ahahah (240, 2151, 9)
         8/8 - 0s - 156ms/epoch - 20ms/step
         Model: "sequential_2"
         _________________________________________________________________
          Layer (type)                Output Shape              Param #   
         =================================================================
          spatial_dropout1d_2 (Spatia  (None, 2151, 9)          0         
          lDropout1D)                                                     
                                                                          
          conv1d_6 (Conv1D)           (None, 428, 8)            1088      
                                                                          
          dropout_2 (Dropout)         (None, 428, 8)            0         
                                                                          
          conv1d_7 (Conv1D)           (None, 136, 64)           10816     
                                                                          
          batch_normalization_4 (Batc  (None, 136, 64)          256       
          hNormalization)                                                 
                                                                          
          conv1d_8 (Conv1D)           (None, 44, 32)            10272     
                                                                          
          batch_normalization_5 (Batc  (None, 44, 32)           128       
          hNormalization)                                                 
                                                                          
          flatten_2 (Flatten)         (None, 1408)              0         
                                                                          
          dense_4 (Dense)             (None, 16)                22544     
                                                                          
          dense_5 (Dense)             (None, 1)                 17        
                                                                          
         =================================================================
         Total params: 45,121
         Trainable params: 44,929
         Non-trainable params: 192
         _________________________________________________________________
         Epoch 1/400
         4/4 - 2s - loss: 0.0347 - val_loss: 0.0205 - 2s/epoch - 613ms/step
         Epoch 2/400
         4/4 - 0s - loss: 0.0266 - val_loss: 0.0216 - 149ms/epoch - 37ms/step
         Epoch 3/400
         4/4 - 0s - loss: 0.0203 - val_loss: 0.0194 - 133ms/epoch - 33ms/step
         Epoch 4/400
         4/4 - 0s - loss: 0.0189 - val_loss: 0.0173 - 124ms/epoch - 31ms/step
         Epoch 5/400
         4/4 - 0s - loss: 0.0175 - val_loss: 0.0177 - 130ms/epoch - 32ms/step
         Epoch 6/400
         4/4 - 0s - loss: 0.0161 - val_loss: 0.0193 - 123ms/epoch - 31ms/step
         Epoch 7/400
         4/4 - 0s - loss: 0.0169 - val_loss: 0.0174 - 133ms/epoch - 33ms/step
         Epoch 8/400
         4/4 - 0s - loss: 0.0151 - val_loss: 0.0163 - 124ms/epoch - 31ms/step
         Epoch 9/400
         4/4 - 0s - loss: 0.0149 - val_loss: 0.0164 - 136ms/epoch - 34ms/step
         Epoch 10/400
         4/4 - 0s - loss: 0.0132 - val_loss: 0.0167 - 137ms/epoch - 34ms/step
         Epoch 11/400
         4/4 - 0s - loss: 0.0141 - val_loss: 0.0161 - 140ms/epoch - 35ms/step
         Epoch 12/400
         4/4 - 0s - loss: 0.0134 - val_loss: 0.0158 - 120ms/epoch - 30ms/step
         Epoch 13/400
         4/4 - 0s - loss: 0.0129 - val_loss: 0.0164 - 123ms/epoch - 31ms/step
         Epoch 14/400
         4/4 - 0s - loss: 0.0107 - val_loss: 0.0163 - 124ms/epoch - 31ms/step
         Epoch 15/400
         4/4 - 0s - loss: 0.0119 - val_loss: 0.0166 - 128ms/epoch - 32ms/step
         Epoch 16/400
         4/4 - 0s - loss: 0.0123 - val_loss: 0.0167 - 140ms/epoch - 35ms/step
         Epoch 17/400
         4/4 - 0s - loss: 0.0109 - val_loss: 0.0166 - 133ms/epoch - 33ms/step
         Epoch 18/400
         4/4 - 0s - loss: 0.0102 - val_loss: 0.0161 - 136ms/epoch - 34ms/step
         Epoch 19/400
         4/4 - 0s - loss: 0.0105 - val_loss: 0.0158 - 143ms/epoch - 36ms/step
         Epoch 20/400
         4/4 - 0s - loss: 0.0111 - val_loss: 0.0156 - 135ms/epoch - 34ms/step
         ...
         Epoch 400/400
         4/4 - 0s - loss: 4.4512e-04 - val_loss: 0.0034 - 125ms/epoch - 31ms/step
         TRANSFORM
         ahahah (120, 2151, 9)
         4/4 - 0s - 364ms/epoch - 91ms/step
         TRANSFORM
         ahahah (241, 2151, 9)
         8/8 - 0s - 132ms/epoch - 16ms/step
         Model: "sequential_3"
         _________________________________________________________________
          Layer (type)                Output Shape              Param #   
         =================================================================
          spatial_dropout1d_3 (Spatia  (None, 2151, 9)          0         
          lDropout1D)                                                     
                                                                          
          conv1d_9 (Conv1D)           (None, 428, 8)            1088      
                                                                          
          dropout_3 (Dropout)         (None, 428, 8)            0         
                                                                          
          conv1d_10 (Conv1D)          (None, 136, 64)           10816     
                                                                          
          batch_normalization_6 (Batc  (None, 136, 64)          256       
          hNormalization)                                                 
                                                                          
          conv1d_11 (Conv1D)          (None, 44, 32)            10272     
                                                                          
          batch_normalization_7 (Batc  (None, 44, 32)           128       
          hNormalization)                                                 
                                                                          
          flatten_3 (Flatten)         (None, 1408)              0         
                                                                          
          dense_6 (Dense)             (None, 16)                22544     
                                                                          
          dense_7 (Dense)             (None, 1)                 17        
                                                                          
         =================================================================
         Total params: 45,121
         Trainable params: 44,929
         Non-trainable params: 192
         _________________________________________________________________
         Epoch 1/400
         4/4 - 2s - loss: 0.0220 - val_loss: 0.0157 - 2s/epoch - 560ms/step
         Epoch 2/400
         4/4 - 0s - loss: 0.0210 - val_loss: 0.0163 - 174ms/epoch - 43ms/step
         Epoch 3/400
         4/4 - 0s - loss: 0.0188 - val_loss: 0.0164 - 161ms/epoch - 40ms/step
         Epoch 4/400
         4/4 - 0s - loss: 0.0205 - val_loss: 0.0162 - 114ms/epoch - 29ms/step
         Epoch 5/400
         4/4 - 0s - loss: 0.0182 - val_loss: 0.0161 - 128ms/epoch - 32ms/step
         Epoch 6/400
         4/4 - 0s - loss: 0.0161 - val_loss: 0.0156 - 117ms/epoch - 29ms/step
         Epoch 7/400
         4/4 - 0s - loss: 0.0156 - val_loss: 0.0157 - 120ms/epoch - 30ms/step
         Epoch 8/400
         4/4 - 0s - loss: 0.0145 - val_loss: 0.0155 - 128ms/epoch - 32ms/step
         Epoch 9/400
         4/4 - 0s - loss: 0.0142 - val_loss: 0.0149 - 157ms/epoch - 39ms/step
         Epoch 10/400
         4/4 - 0s - loss: 0.0139 - val_loss: 0.0152 - 123ms/epoch - 31ms/step
         Epoch 11/400
         4/4 - 0s - loss: 0.0133 - val_loss: 0.0151 - 137ms/epoch - 34ms/step
         Epoch 12/400
         4/4 - 0s - loss: 0.0125 - val_loss: 0.0144 - 139ms/epoch - 35ms/step
         Epoch 13/400
         4/4 - 0s - loss: 0.0120 - val_loss: 0.0151 - 134ms/epoch - 34ms/step
         Epoch 14/400
         4/4 - 0s - loss: 0.0118 - val_loss: 0.0148 - 127ms/epoch - 32ms/step
         Epoch 15/400
         4/4 - 0s - loss: 0.0122 - val_loss: 0.0151 - 124ms/epoch - 31ms/step
         Epoch 16/400
         4/4 - 0s - loss: 0.0117 - val_loss: 0.0151 - 125ms/epoch - 31ms/step
         Epoch 17/400
         4/4 - 0s - loss: 0.0118 - val_loss: 0.0140 - 123ms/epoch - 31ms/step
         Epoch 18/400
         4/4 - 0s - loss: 0.0113 - val_loss: 0.0150 - 128ms/epoch - 32ms/step
         Epoch 19/400
         4/4 - 0s - loss: 0.0109 - val_loss: 0.0142 - 124ms/epoch - 31ms/step
         Epoch 20/400
         4/4 - 0s - loss: 0.0101 - val_loss: 0.0136 - 133ms/epoch - 33ms/step
         ...
         Epoch 400/400
         4/4 - 0s - loss: 3.2399e-04 - val_loss: 0.0033 - 141ms/epoch - 35ms/step
         TRANSFORM
         ahahah (120, 2151, 9)
         4/4 - 0s - 256ms/epoch - 64ms/step
         TRANSFORM
         ahahah (241, 2151, 9)
         8/8 - 0s - 106ms/epoch - 13ms/step
         fit_time [58.65734673 60.80831957 58.15930986]
         score_time [0.65953374 0.64429283 0.53594303]
         test_score [0.61488104 0.64633656 0.71660909]
         train_score [0.92478667 0.9227233  0.95562093]
