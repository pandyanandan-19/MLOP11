import os
import pandera.pandas as pa
from pandera import Check, Column


def get_titanic_schema():
    """Defines the strict schema specification for the Titanic dataset."""
    return pa.DataFrameSchema(
        {
            "PassengerId": Column(pa.Int, Check.ge(1)),
            "Survived": Column(pa.Int, Check.isin([0, 1])),
            "Pclass": Column(pa.Int, Check.isin([1, 2, 3])),
            "Name": Column(pa.String),
            "Sex": Column(pa.String, Check.isin(["male", "female"])),
            "Age": Column(pa.Float, Check.ge(0.0), nullable=True, coerce=True),
            "SibSp": Column(pa.Int, Check.ge(0)),
            "Parch": Column(pa.Int, Check.ge(0)),
            "Ticket": Column(pa.String),
            "Fare": Column(pa.Float, Check.ge(0.0), nullable=True, coerce=True),
            "Cabin": Column(pa.String, nullable=True),
            "Embarked": Column(pa.String, Check.isin(["S", "C", "Q"]), nullable=True),
        },
        strict=True,
    )


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validates a Titanic DataFrame against the strict Pandera schema."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_titanic_schema()

    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")
        failures = err.failure_cases[["schema_context", "column", "check", "failure_case", "index"]]
        print(failures.to_string())

        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False
