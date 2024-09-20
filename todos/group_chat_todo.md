# Group Chat Environment TODO

## 1. Create GroupChatMechanism class which will be used as a mechanism for MultiAgentEnvironment group chat environment class
- [ ] Inherit from `Mechanism` class which will be the attribute of MultiAgentEnvironment class
- [ ] Define attributes:
  - [ ] `max_rounds`: int
  - [ ] `current_round`: int
  - [ ] `messages`: List[Dict[str, Any]]
  - [ ] `topics`: List[str]
  - [ ] `current_topic`: str
  - [ ] `speaker_order`: List[str]
  - [ ] `current_speaker_index`: int

## 2. Create GroupChatAction class
- [ ] Inherit from `LocalAction`
- [ ] Define attributes:
  - [ ] `message`: str
  - [ ] `propose_topic`: Optional[str]

## 3. Create GroupChatObservation class
- [ ] Inherit from `LocalObservation`
- [ ] Define attributes:
  - [ ] `messages`: List[Dict[str, Any]]
  - [ ] `current_topic`: str
  - [ ] `current_speaker`: str

## 4. Create GroupChatGlobalObservation class
- [ ] Inherit from `GlobalObservation`
- [ ] Define attributes:
  - [ ] `observations`: Dict[str, GroupChatObservation]
  - [ ] `all_messages`: List[Dict[str, Any]]
  - [ ] `current_topic`: str
  - [ ] `speaker_order`: List[str]

## 5. Create GroupChatActionSpace class
- [ ] Inherit from `ActionSpace`
- [ ] Define `allowed_actions` attribute with `GroupChatAction`

## 6. Create GroupChatObservationSpace class
- [ ] Inherit from `ObservationSpace`
- [ ] Define `allowed_observations` attribute with `GroupChatObservation`

## 7. Implement GroupChatMechanism methods
- [ ] `step(action: GlobalAction) -> EnvironmentStep`:
  - [ ] Update current round
  - [ ] Process new messages and topic proposals
  - [ ] Update speaker order if needed
  - [ ] Create observations for each agent
  - [ ] Return `EnvironmentStep` with global observation and done status
- [ ] `get_global_state() -> Dict[str, Any]`:
  - [ ] Return current state of the group chat
- [ ] `reset() -> None`:
  - [ ] Reset all attributes to initial state

## 8. Create GroupChatEnvironment class
- [ ] Inherit from `MultiAgentEnvironment`
- [ ] Initialize with `GroupChat`, `GroupChatActionSpace`, and `GroupChatObservationSpace`
- [ ] Implement `step`, `reset`, and other necessary methods

## 9. Update Orchestrator
- [ ] Add method to run group chat before auction
- [ ] Integrate group chat results into auction setup

## 10. Extend MarketAgent
- [ ] Add methods for participating in group chat:
  - [ ] `generate_message()`
  - [ ] `propose_topic()`
  - [ ] `process_group_chat_observation()`

## 11. Implement logging for group chat
- [ ] Create a logger as a timeline for group chat messages
- [ ] Log all messages with timestamps

## 12. Create utility functions
- [ ] `select_next_speaker()`
- [ ] `process_topic_proposals()`

## 13. Testing
- [ ] Write unit tests for GroupChat
- [ ] Write integration tests for GroupChatEnvironment
- [ ] Test group chat with multiple agents

## 14. Documentation
- [ ] Add docstrings to all new classes and methods
- [ ] Update README with information about the group chat feature
