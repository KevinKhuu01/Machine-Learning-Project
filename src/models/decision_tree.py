from sklearn.tree import DecisionTreeRegressor

def model_decision_tree(X_valid, X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    
    return y_pred