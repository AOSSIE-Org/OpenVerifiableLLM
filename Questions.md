# Questions

* Where are we going to store the trained models?

* Can we make the trained models available via ollama?

* Where are we going to store the data that was used to train the models, so that the training can be reproduced?

* When a new version of the training data (e.g. a new version of Wikipedia is released), how are we going to train a new model?
  From scratch? Or incrementally, based on the changes between the different versions of the training data?
  In case of the latter, how will we ensure that the training remains reproducible?

* For independent verification, in the worst case, the verifier should be able to retrain the whole model from scratch from the whole input data and then check that the model he/she trained is equal to the model that we are providing. How can this equality be checked? Will the model files be equal? Or will a different notion of "equality" be used?

* Can we make the training of the model incremental and provide several checkpoints, so that a verifier could verify only a sample of the training, instead of retraining the whole model from scratch to convince himself/herself that the model has been trained on the data on which we claim it was trained on?

* Are there training aproaches that are compositional? Moreprecisely, let `T` be a deterministic training approach so that `T(D)` is the model resulting from training on data `D`. Can we have a `T` such that `T(D1 + D2) = T(D1) + T(D2)`, for some suitable notion of addition of data and addition of models?
