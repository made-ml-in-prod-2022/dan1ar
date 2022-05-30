from pydantic import BaseModel, conlist
from typing import List, Union, Optional
from sklearn.pipeline import Pipeline
import pandas as pd

class TargetModel(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=0, max_items=80)]
    features: List[str]


class TargetResponse(BaseModel):
    condition: int


model: Optional[Pipeline] = None


def make_prediction(
    data: List, features: List[str], mod: Pipeline
) -> List[TargetResponse]:
    data = pd.DataFrame(data, columns=features)

    predicts = mod.predict(data)
    return [
        TargetResponse(condition=int(pred)) for pred in predicts
    ]