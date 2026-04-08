from pydantic import BaseModel, Field
from typing import Literal, Dict, List, Optional
from enum import Enum

class NegotiationActionType(str, Enum):
    ACCEPT = "accept"
    COUNTER_OFFER = "counter_offer"
    CLARIFY = "clarify"
    WALK_AWAY = "walk_away"
    CALL_BLUFF = "call_bluff"
    PROPOSE_ALTERNATIVE = "propose_alternative"

class CounterOffer(BaseModel):
    price: float = Field(..., gt=0)
    timeline_days: int = Field(..., gt=0)
    extras: Dict[str, float] = Field(default_factory=dict)  

class Action(BaseModel):
    action_type: NegotiationActionType
    counter_offer: Optional[CounterOffer] = None
    message: str = Field(..., min_length=5, max_length=500)
    target_issue: Optional[str] = None

class OfferState(BaseModel):
    price: float
    timeline_days: int
    extras: Dict[str, float]
    trust_score: float = Field(..., ge=0.0, le=1.0)

class Observation(BaseModel):
    current_counterpart_offer: OfferState
    conversation_history: List[str]
    relationship_score: float
    detected_bluff_probability: float
    remaining_turns: int
    task_difficulty: str  

class Reward(BaseModel):
    value: float = Field(..., ge=-1.0, le=1.0)
    breakdown: Dict[str, float]  

class TaskInfo(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str