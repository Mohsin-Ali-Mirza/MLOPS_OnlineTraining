from pydantic import BaseModel

class Flower(BaseModel):
    sepalLength: float
    sepalWidth: float
    petalLength: float
    petalWidth: float

class TrainRequest(BaseModel):
    test_size: float = 0.3
    random_state: int = 42