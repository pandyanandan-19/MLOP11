import joblib
import pandas as pd

MODEL_PATH = "models/titanic_model.pkl"


def predict_passenger(
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked
):
    model = joblib.load(MODEL_PATH)

    family_size = sibsp + parch + 1
    is_alone = int(family_size == 1)

    # Name is needed only because the preprocessing function expects it,
    # but Title is provided directly below.
    passenger = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Title": "Mr" if sex == "male" else "Miss"
    }])

    prediction = int(model.predict(passenger)[0])

    print("\n" + "=" * 50)
    print("NEW / UNSEEN PASSENGER PREDICTION")
    print("=" * 50)
    print(f"Passenger class : {pclass}")
    print(f"Sex             : {sex}")
    print(f"Age             : {age}")
    print(f"Family size     : {family_size}")
    print(f"Fare            : {fare}")
    print(f"Embarked        : {embarked}")

    if prediction == 1:
        print("\nPrediction: SURVIVED")
    else:
        print("\nPrediction: DID NOT SURVIVE")

    return prediction


if __name__ == "__main__":
    # Example unseen passenger
    predict_passenger(
        pclass=3,
        sex="male",
        age=25,
        sibsp=0,
        parch=0,
        fare=10.0,
        embarked="S"
    )
