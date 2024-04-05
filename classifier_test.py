from sklearn.metrics import confusion_matrix
import pandas as pd

def classifier_test(classifier_name, classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    training_score = classifier.score(X_train, y_train)
    testing_score = classifier.score(X_test, y_test)
    
    y_pred = classifier.predict(X_test)
    
    if classifier_name != 'Linear Regression':
        cm = confusion_matrix(y_test, y_pred)
        cm = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
        precision = cm.iloc[1, 1] / (cm.iloc[1, 1] + cm.iloc[0, 1])
        recall = cm.iloc[1, 1] / (cm.iloc[1, 1] + cm.iloc[1, 0])
        specificity = cm.iloc[0, 0] / (cm.iloc[0, 0] + cm.iloc[0, 1])
        sensitivity = cm.iloc[1, 1] / (cm.iloc[1, 1] + cm.iloc[1, 0])
        f1_score = 2 * precision * recall / (precision + recall)
        

    
    print(f"Classifier: {classifier_name}")
    print(f"Training Data Score: {100 * training_score:.2f}%")
    print(f"Testing Data Score: {100 * testing_score:.2f}%")
    
    if classifier_name != 'Linear Regression':
        print('\nConfusion Matrix')
        print(cm)
        
    print('\n---------------------------------------------------\n')

    
    return (training_score, testing_score)