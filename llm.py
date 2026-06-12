import time
import boto3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

from utils.logger import logger


MAX_MESSAGES = 20  # 10 user/assistant exchanges

SYSTEM_PROMPT = (
    "You are a knowledge assistant. "
    "Respond briefly and conversationally."
    "do not include any emojis in your response. "
)


def llm_response(message, avatar_session: "BaseAvatar", datainfo: dict = {}):
    try:
        opt = avatar_session.opt
        start = time.perf_counter()

        # Initialize conversation history for this session
        if not hasattr(avatar_session, "history"):
            avatar_session.history = []

        # Add the user's message to history
        avatar_session.history.append(
            {
                "role": "user",
                "content": [
                    {
                        "text": message
                    }
                ]
            }
        )

        # Keep only the most recent messages
        if len(avatar_session.history) > MAX_MESSAGES:
            avatar_session.history = avatar_session.history[-MAX_MESSAGES:]

        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=getattr(opt, "region", "us-east-1")
        )

        end = time.perf_counter()
        logger.info(f"llm Time init: {end - start}s, {message}")

        response = bedrock.converse(
            modelId=getattr(opt, "modelId", "amazon.nova-lite-v1:0"),
            system=[
                {
                    "text": SYSTEM_PROMPT
                }
            ],
            messages=avatar_session.history,
            inferenceConfig={
                "maxTokens": 200,
                "temperature": 0.5
            }
        )

        assistant_text = response["output"]["message"]["content"][0]["text"]

        # Save assistant response to history
        avatar_session.history.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "text": assistant_text
                    }
                ]
            }
        )

        # Trim again after adding assistant response
        if len(avatar_session.history) > MAX_MESSAGES:
            avatar_session.history = avatar_session.history[-MAX_MESSAGES:]

        logger.info(assistant_text)

        # Send response to avatar
        avatar_session.put_msg_txt(assistant_text, datainfo)

        end = time.perf_counter()
        logger.info(f"llm Total Time: {end - start}s")
        return assistant_text

    except Exception:
        logger.exception("llm exception:")
        return