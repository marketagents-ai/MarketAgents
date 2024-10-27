# Group Chat Mechanism & Environment TODO

## 1. Create GroupChatMechanism class
- [ ] Inherit from `Mechanism` class
- [ ] Define core attributes:
  - [ ] `max_rounds`: int
  - [ ] `current_round`: int 
  - [ ] `sub_rounds`: int
  - [ ] `messages`: List[GroupChatMessage]
  - [ ] `topics`: List[str]
  - [ ] `current_topic`: str
  - [ ] `cohorts`: Dict[str, List[str]] # cohort_id -> agent_ids
  - [ ] `topic_proposers`: Dict[str, str] # cohort_id -> proposer_id
  - [ ] `parallel`: bool = True

## 2. Create GroupChatMessage class
- [ ] Inherit from `BaseModel`
- [ ] Define message types:
  - [ ] `propose_topic`
  - [ ] `group_message` 
- [ ] Define attributes:
  - [ ] `content`: str
  - [ ] `message_type`: MessageType
  - [ ] `agent_id`: str
  - [ ] `cohort_id`: str
  - [ ] `sub_round`: int

## 3. Create GroupChatAction & Observation classes
- [ ] GroupChatAction (LocalAction):
  - [ ] `action`: GroupChatMessage
  - [ ] `cohort_id`: str
- [ ] GroupChatObservation (LocalObservation):
  - [ ] `messages`: List[GroupChatMessage]
  - [ ] `current_topic`: str
  - [ ] `cohort_id`: str
  - [ ] `is_proposer`: bool
- [ ] GroupChatGlobalObservation:
  - [ ] `observations`: Dict[str, GroupChatObservation]
  - [ ] `cohorts`: Dict[str, List[str]]
  - [ ] `topics`: Dict[str, str] # cohort_id -> topic

## 4. Create CohortManager class
- [ ] Define cohort formation strategies:
  - [ ] Random assignment
  - [ ] Similarity-based (optional)
  - [ ] Interest-based (optional)
- [ ] Methods:
  - [ ] `form_cohorts(agents: List[str], size: int) -> Dict[str, List[str]]`
  - [ ] `select_topic_proposers(cohorts: Dict) -> Dict[str, str]`
  - [ ] `shuffle_cohorts()`

## 5. Implement GroupChatMechanism methods
- [ ] Protocol state handlers:
  - [ ] `handle_cohort_formation()`
  - [ ] `handle_topic_proposal()`
  - [ ] `handle_group_discussion()`
- [ ] Core methods:
  - [ ] `step(action: GlobalAction) -> EnvironmentStep`
  - [ ] `process_parallel_messages(messages: List[GroupChatMessage])`
  - [ ] `get_global_state()`
  - [ ] `reset()`

## 6. Update GroupChatOrchestrator
- [ ] Add parallel execution support:
  - [ ] `run_parallel_sub_rounds()`
  - [ ] `process_batch_messages()`
- [ ] Add cohort management:
  - [ ] `initialize_cohorts()`
  - [ ] `rotate_cohorts_between_rounds()`
- [ ] Add topic management:
  - [ ] `assign_topic_proposers()`
  - [ ] `collect_proposed_topics()`

## 7. Extend MarketAgent
- [ ] Add protocol-aware methods:
  - [ ] `handle_cohort_assignment(cohort_id: str)`
  - [ ] `generate_topic_proposal()`
  - [ ] `generate_group_message()`
  - [ ] `process_group_messages()`

## 8. Implement Metrics & Analytics
- [ ] Track cohort interactions:
  - [ ] Message frequency
  - [ ] Topic engagement
  - [ ] Agent participation
- [ ] Analyze emergent behaviors:
  - [ ] Cooperation patterns
  - [ ] Deception detection
  - [ ] Information cascades

## 9. Testing & Validation
- [ ] Test parallel execution
- [ ] Test cohort formation
- [ ] Test protocol state transitions
- [ ] Test with large agent populations (1000+)
- [ ] Validate emergent behaviors

## 10. Documentation
- [ ] Protocol specification
- [ ] Cohort management strategies
- [ ] Parallel execution model
- [ ] Emergent behavior analysis
