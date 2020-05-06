import sys, pickle, re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine


def load_data(database_filepath, table_name='disaster_messages'):
    '''
    It loads the database and extract data from the table
    :param: Databased filepath
    Output: Returns the Features X & target y along with target columns names category_names
    '''

    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table(table_name, engine)

    X = df["message"]
    y = df[df.columns[4:]]

    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    :param text: string to tokenize
    :return: text in form as clean token
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens


def build_model():
    """
    :return: Grid Search model with pipeline and parameters
    """
    MOC = MultiOutputClassifier(DecisionTreeClassifier())

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MOC)
    ])

    parameters = {'clf__estimator__max_depth': [10, 50, None],
                  'clf__estimator__min_samples_leaf': [2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    return cv


def performance_results(y_test, y_pred):
    """
    It takes the results of the predictions and compare them with the ground truth
    :param y_test
    :param y_pred
    :return the results of the performance by category in a dataframe
    """
    index = 0
    categories, f_scores, precisions, recalls = [], [], [], []
    for category in y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[category],
                                                                              y_pred[:, index],
                                                                              average='weighted')

        categories.append(category)
        f_scores.append(f_score)
        precisions.append(precision)
        recalls.append(recall)

    results = pd.DataFrame({'Category':categories, 'f_score': f_scores, 'precision': precisions, 'recall':recalls})

    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())
    print('Aggregated f_score:', results['f_score'].mean())
    return results


def evaluate_model(model, X_test, y_test):
    """
    Provide the model results based on the predictions

    :param model: the model estimator object
    :param X_test: columns with features for th test
    :param y_test: column with the ground truth

    """
    # Get results and add them to a dataframe.
    y_pred = model.predict(X_test)
    performance_results(y_test, y_pred)


def save_model(model, model_filepath):
    '''
    Save the model as pickle file in the give filepath

    :param model: model object
    :param model_filepath
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()