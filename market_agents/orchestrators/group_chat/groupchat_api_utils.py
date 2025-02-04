# api_utils.py

import asyncio
import aiohttp
import logging
from typing import Any, Dict, List, Tuple, Optional

class GroupChatAPIUtils:
    def __init__(self, api_url: str, logger: logging.Logger):
        self.api_url = api_url
        self.logger = logger
        self.logger.info(f"Initializing GroupChat API Utils with URL: {api_url}")

    async def check_api_health(self) -> bool:
        """Check if the GroupChat API is healthy."""
        try:
            self.logger.info(f"Checking GroupChat API health at {self.api_url}/health")
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/health", timeout=5) as resp:
                    if resp.status == 200:
                        self.logger.info("GroupChat API is healthy")
                        return True
                    else:
                        self.logger.error(f"GroupChat API health check failed: {resp.status}")
                        return False
        except aiohttp.ClientError as e:
            self.logger.error(f"Connection error to GroupChat API: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Could not connect to GroupChat API: {e}")
            return False

    async def register_agents(self, agents: List[Any]) -> None:
        """Register multiple agents with the GroupChat API."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, agent in enumerate(agents):
                payload = {
                    "id": agent.id,
                    "index": i
                }
                tasks.append(self._register_agent(session, payload))
            results = await asyncio.gather(*tasks)
            for success, agent_id in results:
                if success:
                    self.logger.info(f"Registered agent {agent_id}")
                else:
                    self.logger.error(f"Failed to register agent {agent_id}")

    async def _register_agent(self, session: aiohttp.ClientSession, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """Helper method to register a single agent."""
        try:
            async with session.post(f"{self.api_url}/register_agent", json=payload) as resp:
                if resp.status == 200:
                    return True, payload["id"]
                else:
                    self.logger.error(f"Failed to register agent {payload['id']}: {resp.status}")
                    return False, payload["id"]
        except Exception as e:
            self.logger.error(f"Exception while registering agent {payload['id']}: {e}")
            return False, payload["id"]

    async def form_cohorts(self, agent_ids: List[str], cohort_size: int) -> List[Dict[str, Any]]:
        """Form cohorts using the GroupChat API."""
        payload = {"agent_ids": agent_ids, "cohort_size": cohort_size}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.api_url}/form_cohorts", json=payload) as resp:
                    if resp.status == 200:
                        cohorts_info = await resp.json()
                        self.logger.info(f"Cohorts formed: {[cohort['cohort_id'] for cohort in cohorts_info]}")
                        return cohorts_info
                    else:
                        error_detail = await resp.text()
                        self.logger.error(f"Failed to form cohorts: {resp.status}, {error_detail}")
                        raise Exception("Failed to form cohorts")
            except Exception as e:
                self.logger.error(f"Exception while forming cohorts: {e}")
                raise

    async def select_proposer(self, cohort_id: str, agent_ids: List[str]) -> Optional[str]:
        """Select a topic proposer for a cohort."""
        payload = {"cohort_id": cohort_id, "agent_ids": agent_ids}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.api_url}/select_proposer", json=payload) as resp:
                    if resp.status == 200:
                        proposer_info = await resp.json()
                        proposer_id = proposer_info.get('proposer_id')
                        self.logger.info(f"Selected proposer {proposer_id} for cohort {cohort_id}")
                        return proposer_id
                    else:
                        error_detail = await resp.text()
                        self.logger.error(f"Failed to select proposer for cohort {cohort_id}: {resp.status}, {error_detail}")
                        return None
            except Exception as e:
                self.logger.error(f"Exception while selecting proposer for cohort {cohort_id}: {e}")
                return None

    async def propose_topic(self, agent_id: str, cohort_id: str, topic: str, round_num: int) -> bool:
        """Submit a topic proposal for a cohort."""
        payload = {
            "agent_id": agent_id,
            "cohort_id": cohort_id,
            "topic": topic,
            "round_num": round_num
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.api_url}/propose_topic", json=payload) as resp:
                    if resp.status == 200:
                        self.logger.info(f"Topic proposed by agent {agent_id} for cohort {cohort_id}")
                        return True
                    else:
                        error_detail = await resp.text()
                        self.logger.error(f"Failed to propose topic for cohort {cohort_id}: {resp.status}, {error_detail}")
                        return False
            except Exception as e:
                self.logger.error(f"Exception while proposing topic for cohort {cohort_id}: {e}")
                return False

    async def get_topic(self, cohort_id: str) -> Optional[str]:
        """Retrieve the current topic for a cohort."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.api_url}/get_topic/{cohort_id}") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        topic = data.get('topic', '')
                        self.logger.debug(f"Retrieved topic for cohort {cohort_id}: {topic}")
                        return topic
                    else:
                        error_detail = await resp.text()
                        self.logger.error(f"Failed to get topic for cohort {cohort_id}: {resp.status}, {error_detail}")
                        return None
            except Exception as e:
                self.logger.error(f"Exception while getting topic for cohort {cohort_id}: {e}")
                return None

    async def get_messages(self, cohort_id: str) -> List[Dict[str, Any]]:
        """Retrieve messages for a cohort."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.api_url}/get_messages/{cohort_id}") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        messages = data.get('messages', [])
                        self.logger.info(f"Retrieved messages for cohort {cohort_id}")
                        return messages
                    else:
                        error_detail = await resp.text()
                        self.logger.error(f"Failed to get messages for cohort {cohort_id}: {resp.status}, {error_detail}")
                        return []
            except Exception as e:
                self.logger.error(f"Exception while getting messages for cohort {cohort_id}: {e}")
                return []

    async def post_message(self, agent_id: str, cohort_id: str, content: str, round_num: int, sub_round_num: int) -> bool:
        """Post a message to a cohort."""
        payload = {
            "agent_id": agent_id,
            "content": content,
            "cohort_id": cohort_id,
            "round_num": round_num,
            "sub_round_num": sub_round_num
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.api_url}/post_message", json=payload) as resp:
                    if resp.status == 200:
                        self.logger.info(f"Message posted by agent {agent_id} in cohort {cohort_id}")
                        return True
                    else:
                        error_detail = await resp.text()
                        self.logger.error(f"Failed to post message for cohort {cohort_id}: {resp.status}, {error_detail}")
                        return False
            except Exception as e:
                self.logger.error(f"Exception while posting message for cohort {cohort_id}: {e}")
                return False
