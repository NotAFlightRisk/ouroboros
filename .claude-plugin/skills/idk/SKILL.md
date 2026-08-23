---
name: idk
description: "Calibrate interview explanations from the user's topic-specific understanding"
aliases: [dont-know, calibrate]
mcp_tool: ouroboros_interview
mcp_args:
  calibration_input: "$1"
---

# /ouroboros:idk

Calibrate the language used by `/ouroboros:interview` without weakening the interview itself.

## Usage

```text
/ouroboros:idk <terms I do not know, or what I understand about them>
```

Examples:

```text
/ouroboros:idk I do not know idempotency or event sourcing. I have built REST APIs.
/ouroboros:idk Kubernetes: deployed a tutorial once; networking and operators are unfamiliar.
/ouroboros:idk OAuth is familiar enough to implement, but I cannot explain PKCE.
```

## Required Skill Capabilities

- `call_mcp` — use Ouroboros MCP tools for calibration processing.
- `maintain_ledger` — keep the inferred calibration level visible in the main session.

## Instructions

1. The user has reported terms or domains they find unfamiliar.
2. Forward the evidence to the `ouroboros_interview` tool using the `calibration_input` argument.
3. This does NOT answer a pending interview question — it adjusts the wording of future questions.
4. Display the calibration result (level, confidence, rephrased question if available).
5. Continue the interview with the next question from the tool response.

## Guardrails

- Do NOT consume calibration input as an answer to a pending question.
- Do NOT persist calibration outside the current conversation session.
- Do NOT lower interview rigor, ambiguity checks, or closure gates.
- Do NOT make global ability judgments — calibration is topic-specific.
