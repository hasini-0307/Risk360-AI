# Risk360-AI
AI-powered multi-agent early warning system for loan default prediction using structured + unstructured data, explainable AI, retrieval, and intelligent recommendations.

# Risk360 AI Engine

## Overview

The AI Engine is the explainability and decision intelligence layer of Risk360 AI. It combines machine learning outputs with multiple specialized AI agents to generate explainable credit risk assessments and a final enterprise lending recommendation.

---

## Modules

### `agents/`

Specialized AI agents responsible for domain-specific risk assessment.

* **behavior_agent.py** → Evaluates repayment behaviour and delinquency risk.
* **income_agent.py** → Assesses income stability and repayment capacity.
* **transaction_agent.py** → Detects transaction-based financial risk patterns.
* **document_agent.py** → Analyzes branch remarks, verification notes, and document-related risks.
* **base_agent.py** → Shared abstraction for all AI agents (LLM communication, retries, logging, parsing).

---

### `cro/`

Chief Risk Officer (CRO) orchestration layer.

* **chief_risk_officer.py** → Coordinates all specialist agents and produces the final credit decision.
* **risk_fusion.py** → Combines specialist assessments using loan-type weighted risk fusion.
* **approval_policy.py** → Enterprise approval rules (APPROVE / MANUAL_REVIEW / REJECT).
* **conflict_detector.py** → Detects disagreements between specialist agents.

---

### `llm/`

* **llm_client.py** → Central LLM interface.
* **prompts.py** → System prompts for all AI agents.
* **config.py** → LLM configuration.
* **types.py** → LLM request/response models.
* **exceptions.py** → LLM-specific exception handling.

---

### `schemas/`

Shared Pydantic models for:

* AI Requests
* Agent Outputs
* CRO Outputs
* Common data structures
* API responses

---

### `utils/`

Reusable utilities including:

* JSON response parsing
* Confidence scoring
* Metadata generation
* Prompt construction
* Response validation
* Risk reason generation
* Logging
* Execution timing

---

## AI Workflow

Borrower Data
→ Specialist AI Agents
→ Behavior Assessment
→ Income Assessment
→ Transaction Assessment
→ Document Assessment
→ Chief Risk Officer
→ Weighted Risk Fusion
→ Approval Policy
→ Final Explainable Decision

---

## Testing

Run the end-to-end AI Engine test:

```bash
python test_cro.py
```
