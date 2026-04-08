import os
import json
from openai import OpenAI
from models import Action, NegotiationActionType, CounterOffer
from env import NegotiationTrainerEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

OPENAI_API_KEY = "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"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=OPENAI_API_KEY
)

def run_inference(env: NegotiationTrainerEnvironment, num_episodes: int = 3):
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
                prompt = f"""
You are a professional business negotiator.
Task: {task.description}
Difficulty: {task.difficulty}

Current offer from counterpart:
- Price: ${obs.current_counterpart_offer.price:,.0f}
- Timeline: {obs.current_counterpart_offer.timeline_days} days
- Extras: {obs.current_counterpart_offer.extras}
- Relationship: {obs.relationship_score:.2f}
- Bluff probability: {obs.detected_bluff_probability:.2f}

Choose the best next action. Be strategic, professional, and aim for win-win where possible.
Respond ONLY with valid JSON.
"""

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt.strip()}],
                        temperature=0.7,
                        max_tokens=300,
                        response_format={"type": "json_object"}
                    )

                    llm_output = json.loads(response.choices[0].message.content)

                    action = Action(
                        action_type=NegotiationActionType(llm_output.get("action_type", "counter_offer")),
                        counter_offer=CounterOffer(**llm_output.get("counter_offer", {"price": 90000, "timeline_days": 30}))
                            if llm_output.get("counter_offer") else None,
                        message=llm_output.get("message", "I propose a mutually beneficial deal.")
                    )

                except Exception:
                    action = Action(
                        action_type=NegotiationActionType.COUNTER_OFFER,
                        counter_offer=CounterOffer(price=90000, timeline_days=30),
                        message="Let's find a fair compromise."
                    )

                obs, reward, done, info = env.step(action)
                episode_reward += reward.value
                step_count += 1

                print(f"[STEP] Task:{task.task_id} Ep:{episode+1} Step:{step_count} Reward:{reward.value:.3f}")

            final_grade = env.grade_task(episode_reward, task.difficulty)
            task_rewards.append(final_grade)

        avg_task_score = round(sum(task_rewards) / len(task_rewards), 3)
        task_scores[task.task_id] = avg_task_score
        total_score += avg_task_score

        print(f"[STEP] Task {task.task_id} completed | Avg grade: {avg_task_score:.3f}\n")

    final_score = round(total_score / len(env.tasks), 3)

    print("[END] Inference Completed")
    print(f"[RESULT] Overall Baseline Score: {final_score}")
    print(f"[RESULT] Task Breakdown: {json.dumps(task_scores, indent=2)}")
    print(f"[RESULT] Team Royal Ace - Negotiation Trainer")

    return final_score


if __name__ == "__main__":
    env = NegotiationTrainerEnvironment()
    run_inference(env, num_episodes=3)