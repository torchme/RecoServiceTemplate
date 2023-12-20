import json
import os
import pickle
import sys
from typing import Any, Dict, List, Union

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from service.api.exceptions import BearerAccessTokenError, UserNotFoundError
from service.log import app_logger

LFM_PATH = os.path.join(sys.path[0], "data/lfm_recs.pkl")
ANN_PATH = os.path.join(sys.path[0], "models/ann_lightfm.pkl")
COLD_USERS_PATH = os.path.join(sys.path[0], "models/cold_users.pkl")
POPULAR = os.path.join(sys.path[0], "models/popular.pkl")
MULTIAE = os.path.join(sys.path[0], "data/multiae.json")
DSSM = os.path.join(sys.path[0], "data/dssm_recos.csv")

with open(LFM_PATH, "rb") as file:
    lfm_model = pickle.load(file)

with open(ANN_PATH, "rb") as file:
    ann_model = pickle.load(file)

with open(COLD_USERS_PATH, "rb") as file:
    cold_users = pickle.load(file)

with open(POPULAR, "rb") as file:
    popular_items = pickle.load(file)

with open(MULTIAE, 'r') as file:
    multiae = json.load(file)
multiae_users = multiae.keys()

class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


responses: Dict[Union[int, str], Dict[str, Any]] = {
    404: {"description": "Model not found", "content": {"application/json": {"example": {"detail": "Model not found"}}}}
}

router = APIRouter()

bearer_scheme = HTTPBearer()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses=responses,
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if token.credentials != request.app.state.true_token:
        raise BearerAccessTokenError()
    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs

    if model_name == "ann":
        if user_id in cold_users:
            reco = list(ann_model.get_item_list_for_user(user_id, top_n=k_recs))
            if not reco:
                reco = popular_items
            elif len(reco) < k_recs:
                reco = list(reco + popular_items)[:k_recs]
        else:
            reco = popular_items
    elif model_name == "lfm":
        if user_id in cold_users:
            reco = lfm_model[user_id]
            if not reco:
                reco = popular_items
            elif len(reco) < k_recs:
                reco = list(reco + popular_items)[:k_recs]
        else:
            reco = popular_items
    elif model_name == "mae":
        if user_id in cold_users:
            reco = multiae[str(user_id)]
            if not reco:
                reco = popular_items
            elif len(reco) < k_recs:
                reco = list(reco + popular_items)[:k_recs]
        else:
            reco = popular_items
    elif model_name == "dssm":
        if user_id in cold_users:
            reco = DSSM[DSSM["user_id"] == str(user_id)]["item_id"].tolist()
            if not reco:
                reco = popular_items
            elif len(reco) < k_recs:
                reco = list(reco + popular_items)[:k_recs]
        else:
            reco = popular_items

    else:
        raise HTTPException(status_code=404, detail="Model not found")

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
