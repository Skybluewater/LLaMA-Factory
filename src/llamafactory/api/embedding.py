import json

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from ..extras.logging import get_logger
from ..extras.packages import is_fastapi_available, is_pillow_available, is_requests_available
from .common import dictify, jsonify
from .protocol import (
    EmbeddingCompletionResponseUsage,
    EmbeddingDataResponse,
    EmbeddingResponse
)

if is_fastapi_available():
    from fastapi import HTTPException, status

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..chat import EmbeddingModel
    from .protocol import EmbeddingRequest


logger = get_logger(__name__)


def _process_request(
    request: "EmbeddingRequest",
) -> Tuple[Union[List[str], str], Optional[str]]:
    logger.info("==== request ====\n{}".format(json.dumps(dictify(request), indent=2, ensure_ascii=False)))

    if (isinstance(request.input, str) and len(request.input) == 0) or (isinstance(request.input, list) and len(request.input) == 0):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid length")
    
    if isinstance(request.input, list) and isinstance(request.input[0], int):
        request.input = [str(num) for num in request.input]
    
    return request.input, request.model


async def create_embedding_response(
    request: "EmbeddingRequest", embedding_model: "EmbeddingModel"
) -> "EmbeddingResponse":
    input, model = _process_request(request)
    responses = await embedding_model.achat(
        input
    )

    prompt_length, response_length = 0, 0
    data = []
    for i, response in enumerate(responses):
        result = EmbeddingDataResponse(object="embedding", embedding=response, index=i + 1)
        data.append(result)

    usage = EmbeddingCompletionResponseUsage(
        prompt_tokens=prompt_length,
        total_tokens=prompt_length + response_length,
    )

    return EmbeddingResponse(object="list", data=data, model=model, usage=usage)