import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
import pickle

def load_data(database_filepath):
    '''
    Load data from database as dataframe
    Input:
        database_filepath: File path of sql database
    Output:
        X: Message data (features)
        Y: Categories (target)
        category_names: Labels for 36 categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    '''
    Tokenize and clean text
    Input:
        text: original message text
    Output:
        lemmed: Tokenized, cleaned, and lemmatized text
    '''
    # Normalize Text
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    # Tokenize
    words = word_tokenize(text)
    # Remove Stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w, pos='n').strip() for w in words]
    lemmed = [lemmatizer.lemmatize(w, pos='v').strip() for w in lemmed]

    return lemmed


def build_model():
    '''
    Build a ML pipeline using ifidf, random forest, and gridsearch
    Input: None
    Output:
        Results of GridSearchCV
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                  }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance using test data
    Input:
        model: Model to be evaluated
        X_test: Test data (features)
        Y_test: True lables for Test data
        category_names: Labels for 36 categories
    Output:
        Print accuracy and classfication report for each category
    '''
    Y_pred = model.predict(X_test)

    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    '''
    Save model as a pickle file
    Input:
        model: Model to be saved
        model_filepath: path of the output pick file
    Output:
        A pickle file of saved model
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()