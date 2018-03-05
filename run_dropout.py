execfile('size.py')
execfile('nn.py')
a = NeuralNetMLP(n_output=10, 
                  n_features=X_train.shape[1], 
                  n_hidden=30, 
                  do_dropout = True,
                  epochs=1000, 
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  minibatches=50, 
                  shuffle=False,
                  random_state=1)

a.fit(X_train, y_train, print_progress=True)

