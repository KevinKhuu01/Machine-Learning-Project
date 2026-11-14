from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def model_decision_tree(X_valid, X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': [None, "sqrt", 0.5],
        'ccp_alpha': [0.0, 0.001, 0.01]
    }

    
    search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5,
                           scoring='neg_mean_squared_error', n_jobs=-1)
    search.fit(X_train, y_train)
    
    print(f"Best parameters: {search.best_params_}")
    print(f"CV RMSE: {-search.best_score_}")

    return search.predict(X_valid) 