import random
from typing import Optional, Dict, List
from models import (
    Action, Observation, Reward, OfferState,
    NegotiationActionType, TaskInfo
)

class NegotiationTrainerEnvironment:
    """
    OpenEnv-compliant environment for Strategic Business Negotiation Training.
    Real-world task: Training AI agents for high-stakes negotiations (salary, contracts, deals).
    """

    def __init__(self):
        self.episode_id: Optional[str] = None
        self.current_step: int = 0
        self.max_steps: int = 15
        self.task: Optional[TaskInfo] = None
        self.current_offer: Optional[OfferState] = None
        self.target_deal: Dict[str, float] = {}
        self.relationship_score: float = 0.5
        self.bluff_probability: float = 0.0
        self.conversation_history: List[str] = []
        self.done: bool = False
        self.tasks = self._define_tasks()

    def _define_tasks(self) -> List[TaskInfo]:
        return [
            TaskInfo(
                task_id="basic_deal",
                difficulty="easy",
                description="Close a simple single-issue price negotiation quickly and professionally."
            ),
            TaskInfo(
                task_id="multi_issue",
                difficulty="medium",
                description="Balance multiple issues (price, timeline, extras) while maximizing mutual value."
            ),
            TaskInfo(
                task_id="bluff_handling",
                difficulty="hard",
                description="Detect and handle bluffs or fake alternatives while maintaining relationship and securing best deal."
            )
        ]

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """Reset the environment for a new episode."""
        self.episode_id = f"neg_{random.randint(10000, 99999)}"
        self.current_step = 0
        self.done = False
        self.conversation_history = []

        if task_id is None:
            self.task = random.choice(self.tasks)
        else:
            self.task = next(t for t in self.tasks if t.task_id == task_id)

        if self.task.difficulty == "easy":
            self.current_offer = OfferState(
                price=85000, timeline_days=30, extras={}, trust_score=0.6
            )
            self.target_deal = {"price": 92000, "timeline": 25}
            self.bluff_probability = 0.1
        elif self.task.difficulty == "medium":
            self.current_offer = OfferState(
                price=82000, timeline_days=45, extras={"warranty": 300}, trust_score=0.5
            )
            self.target_deal = {"price": 90000, "timeline": 30, "extras": 800}
            self.bluff_probability = 0.3
        else:  
            self.current_offer = OfferState(
                price=78000, timeline_days=60, extras={"support": 200}, trust_score=0.4
            )
            self.target_deal = {"price": 88000, "timeline": 28, "extras": 1200}
            self.bluff_probability = 0.6

        self.relationship_score = self.current_offer.trust_score

        initial_obs = Observation(
            current_counterpart_offer=self.current_offer,
            conversation_history=self.conversation_history,
            relationship_score=self.relationship_score,
            detected_bluff_probability=self.bluff_probability,
            remaining_turns=self.max_steps - self.current_step,
            task_difficulty=self.task.difficulty
        )

        return initial_obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """Execute one negotiation step."""
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")

        self.current_step += 1
        self.conversation_history.append(f"Agent: {action.message}")

        reward_value = 0.0
        breakdown = {
            "value_maximization": 0.0,
            "relationship": 0.0,
            "time_efficiency": 0.0,
            "ethics": 0.0,
            "bluff_detection": 0.0
        }

        if action.action_type == NegotiationActionType.ACCEPT:
            deal_value = self._calculate_deal_value(self.current_offer)
            reward_value += deal_value * 0.8
            breakdown["value_maximization"] = deal_value * 0.8
            self.done = True

        elif action.action_type == NegotiationActionType.COUNTER_OFFER and action.counter_offer:
            new_price = action.counter_offer.price
            new_timeline = action.counter_offer.timeline_days

            self.current_offer.price = (self.current_offer.price + new_price) * 0.6
            self.current_offer.timeline_days = int((self.current_offer.timeline_days + new_timeline) * 0.7)
            self.relationship_score = min(1.0, self.relationship_score + 0.08)

            progress = self._calculate_progress()
            reward_value += progress * 0.6
            breakdown["value_maximization"] = progress * 0.6
            breakdown["relationship"] = 0.08

        elif action.action_type == NegotiationActionType.CALL_BLUFF:
            if self.bluff_probability > 0.4 and random.random() < 0.7:
                reward_value += 0.35
                breakdown["bluff_detection"] = 0.35
                self.relationship_score -= 0.05 
            else:
                reward_value -= 0.25
                breakdown["ethics"] = -0.25

        elif action.action_type == NegotiationActionType.WALK_AWAY:
            reward_value -= 0.4
            self.done = True

        time_penalty = -0.02 * (self.current_step / self.max_steps)
        reward_value += time_penalty
        breakdown["time_efficiency"] = time_penalty

        self.relationship_score = max(0.0, min(1.0, self.relationship_score))

        if self.current_step >= self.max_steps:
            self.done = True

        obs = Observation(
            current_counterpart_offer=self.current_offer,
            conversation_history=self.conversation_history[-10:],  # keep recent
            relationship_score=self.relationship_score,
            detected_bluff_probability=self.bluff_probability,
            remaining_turns=self.max_steps - self.current_step,
            task_difficulty=self.task.difficulty
        )

        reward = Reward(value=round(reward_value, 3), breakdown=breakdown)

        info = {
            "task_id": self.task.task_id,
            "step": self.current_step,
            "episode_id": self.episode_id
        }

        return obs, reward, self.done, info

    def state(self) -> dict:
        """Return current episode metadata."""
        return {
            "episode_id": self.episode_id,
            "step_count": self.current_step,
            "task": self.task.dict() if self.task else None,
            "done": self.done
        }

    def _calculate_deal_value(self, offer: OfferState) -> float:
        """Compute how good the final deal is (0.0 - 1.0)."""
        price_score = max(0, min(1, (offer.price - 75000) / 20000))
        timeline_score = max(0, min(1, (60 - offer.timeline_days) / 40))
        return (price_score + timeline_score) / 2

    def _calculate_progress(self) -> float:
        """Partial progress toward target deal."""
        current_value = self._calculate_deal_value(self.current_offer)
        target_value = 0.85  # normalized target
        return min(1.0, current_value / target_value)

    def grade_task(self, total_reward: float, difficulty: str):
        """Grade the task - more realistic scoring"""
        if difficulty == "easy":
            return max(0.0, min(1.0, (total_reward + 0.7) / 1.5))
        elif difficulty == "medium":
            return max(0.0, min(1.0, (total_reward + 0.5) / 1.4))
        else:  # hard
            return max(0.0, min(1.0, (total_reward + 0.3) / 1.3))