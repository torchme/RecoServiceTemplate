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

POPULAR = os.path.join(sys.path[0], "models/general_popular.pkl")
RANKER = os.path.join(sys.path[0], "data/double.json")
# MULTIVAE = os.path.join(sys.path[0], "data/multiae.json")
TOP_POPULAR = os.path.join(sys.path[0], "data/top_popular.json")

with open(POPULAR, "rb") as file:
    popular_items = pickle.load(file)

with open(TOP_POPULAR, 'r') as file:
    top_popular = json.load(file)
top_popular_users = top_popular.keys()

with open(RANKER, 'r') as file:
    ranker = json.load(file)
ranker_users = ranker.keys()

# with open(MULTIVAE, 'r') as file:
#     multivae = json.load(file)
# multivae_users = multivae.keys()


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

    if model_name == "ranker":
        if str(user_id) in ranker_users:
            reco = ranker[str(user_id)]
            if not reco:
                reco = popular_items
            elif len(reco) < k_recs:
                reco = list(reco + popular_items)[:k_recs]
        if str(user_id) in top_popular_users:
            reco = top_popular[str(user_id)]
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
