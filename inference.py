import os
import json
from openai import OpenAI
from models import Action, NegotiationActionType, CounterOffer
from env import NegotiationTrainerEnvironment

# ==================== SAFE CONFIG ====================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Read API key safely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("[WARNING] OPENAI_API_KEY not found in environment variables.")
    print("[INFO] Running in fallback mode (no real LLM calls).")
    client = None
else:
    client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)

def run_inference(num_episodes=3):
    env = NegotiationTrainerEnvironment()
    
    print("[START] Negotiation Trainer OpenEnv Baseline Inference")
    print(f"[INFO] Team: Royal Ace")
    print(f"[INFO] Project: Negotiation Trainer - Strategic Business Deal Simulator")
    print(f"[INFO] Model: {MODEL_NAME} | Episodes per task: {num_episodes}\n")

    total_score = 0.0
    task_scores = {}

    for task in env.tasks:
        print(f"[STEP] Starting task: {task.task_id} ({task.difficulty})")

        task_rewards = []
        for episode in range(num_episodes):
            obs = env.reset(task_id=task.task_id)
            done = False
            episode_reward = 0.0
            step_count = 0

            while not done and step_count < 20:
                if client is None:
                    # Fallback when no API key
                    action = Action(
                        action_type=NegotiationActionType.COUNTER_OFFER,
                        counter_offer=CounterOffer(price=90000, timeline_days=30),
                        message="Let's find a fair compromise."
                    )
                else:
                    prompt = f"""
You are a professional business negotiator.
Task: {task.description}
Current offer: Price ${obs.current_counterpart_offer.price:.0f}, Timeline {obs.current_counterpart_offer.timeline_days} days
Relationship: {obs.relationship_score:.2f}

What is your next move? Reply with JSON only.
"""

                    try:
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[{"role": "user", "content": prompt.strip()}],
                            temperature=0.7,
                            max_tokens=250,
                            response_format={"type": "json_object"}
                        )
                        llm_output = json.loads(response.choices[0].message.content)

                        action = Action(
                            action_type=NegotiationActionType(llm_output.get("action_type", "counter_offer")),
                            counter_offer=CounterOffer(**llm_output.get("counter_offer", {"price": 90000, "timeline_days": 30}))
                                if llm_output.get("counter_offer") else None,
                            message=llm_output.get("message", "I propose a fair compromise.")
                        )
                    except Exception as e:
                        print(f"[WARNING] LLM call failed: {e}. Using fallback.")
                        action = Action(
                            action_type=NegotiationActionType.COUNTER_OFFER,
                            counter_offer=CounterOffer(price=90000, timeline_days=30),
                            message="Let's find a fair compromise."
                        )

                obs, reward, done, info = env.step(action)
                episode_reward += reward.value
                step_count += 1

            final_grade = env.grade_task(episode_reward, task.difficulty)
            task_rewards.append(final_grade)

        avg_task_score = round(sum(task_rewards) / len(task_rewards), 3)
        task_scores[task.task_id] = avg_task_score
        total_score += avg_task_score

        print(f"[STEP] Task {task.task_id} completed | Average grade: {avg_task_score:.3f}\n")

    final_score = round(total_score / len(env.tasks), 3)

    print("[END] Inference Completed")
    print(f"[RESULT] Overall Baseline Score: {final_score}")
    print(f"[RESULT] Task Breakdown: {task_scores}")

    return final_score


if __name__ == "__main__":
    run_inference(num_episodes=3)