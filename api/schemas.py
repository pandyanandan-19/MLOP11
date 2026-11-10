from typing import Literal

from pydantic import BaseModel, Field


class PassengerInput(BaseModel):
    PassengerId: int = Field(ge=1)
    Pclass: Literal[1, 2, 3]
    Name: str = Field(min_length=1)
    Sex: Literal["male", "female"]
    Age: float | None = Field(default=None, ge=0.0)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Ticket: str = Field(min_length=1)
    Fare: float | None = Field(default=None, ge=0.0)
    Cabin: str | None = None
    Embarked: Literal["S", "C", "Q"] | None = None


class PredictionResponse(BaseModel):
    prediction: int
    survival_probability: float


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    status: str
    message: str
    details: list[ErrorDetail]


class BatchPredictionRequest(BaseModel):
    passengers: list[PassengerInput] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
