y_train_pred = a.predict(X_train)

acc = ((np.sum(y_train == y_train_pred, axis=0)).astype('float') /X_train.shape[0])

print('Training accuracy: %.2f%%' % (acc * 100))



y_test_pred = a.predict(X_test)

acc = ((np.sum(y_test == y_test_pred, axis=0)).astype('float') /X_test.shape[0])

print('Test accuracy: %.2f%%' % (acc * 100))
