from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_data(df, training = True):
    df.drop(columns = ["PassengerId", "Name", "Cabin", "Ticket"], inplace=True)
    if(training):
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
        my_obj = {
            "Age_med": df["Age"].median(),
            "Fare_avg": df["Fare"].mean()
        }
        joblib.dump(my_obj, "stats.pkl")
    else:
        my_obj = joblib.load("stats.pkl")
        df["Age"] = df["Age"].fillna(my_obj["Age_med"])
        df["Fare"] = df["Fare"].fillna(my_obj["Fare_avg"])
    missing_values = df.isnull().sum()
    print(missing_values)
    df.dropna(inplace = True)
    label_encoder = LabelEncoder()
    df["Sex"] = label_encoder.fit_transform(df["Sex"])
    df["Embarked"] = label_encoder.fit_transform(df["Embarked"])
    df["Sex"] = df["Sex"].astype("category")
    df["Pclass"] = df["Pclass"].astype("category")
    df["Embarked"] = df["Embarked"].astype("category")
    
    df["Family_size"] = df["Parch"] + df["SibSp"]
    df["IsAlone"] = df["Family_size"] == 0
    df["NoChildren"] = (df["Parch"] == 0) & (df["Age"] > 20)
    df["NoChildren"] = df["NoChildren"].astype("category")
    df["IsAlone"] = df["IsAlone"].astype("category")
    
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100], labels=["Child", "Teen", "Adult", "Senior", "Elder"])
    df["AgeGroup"] = df["AgeGroup"].astype("category")
    
    # df["FareGroup"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Medium", "High", "Very High"])
    # df["FareGroup"] = df["FareGroup"].astype("category")
def train(df):
    
    
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(learning_rate = .001, 
                          n_estimators=600,      
                          max_depth=6,
                          enable_categorical=True,
                          booster="dart"
                          )
    
    
    # Train the model
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)

    # Calculate training accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "trained_model.pkl")
    
def evaluate(df):
    model = joblib.load("trained_model.pkl")

    passenger_ids = df["PassengerId"].copy()

    prepared_df = df.copy(deep=True)

    prepare_data(prepared_df, training=False)

    predictions = model.predict(prepared_df)

    results = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions
    })
    results.to_csv("predictions.csv", index=False)
    return results
    
def main():
    # df = pd.read_csv("./data/train.csv")   
    # prepare_data(df)
    # train(df)
    test_df = pd.read_csv("./data/test.csv")
    evaluate(test_df)
    
if __name__ == "__main__":
    main()