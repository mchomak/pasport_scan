# ADR 0001: Shared adapter/core boundary

## Context

MAX must be added without changing the existing Telegram user flow or duplicating OCR and business logic. Platform SDK objects, downloads, and presentation formats are messenger-specific; recognition, normalization, persistence, and result data are not.

## Decision

Keep Telegram and MAX as thin adapters. Each adapter downloads platform data, normalizes it into the SDK-independent `IncomingImage` contract, invokes `PassportProcessingService`, and renders the shared result in its own presentation format. The core owns OCR/business processing, persistence, and lifecycle; it must not import Telegram or MAX SDK types.

## Why/rejected alternative

This gives one source of truth for OCR behavior and a testable seam between platform integration and domain work. Duplicating a pipeline per messenger was rejected because fixes and behavior would drift; passing SDK objects into the core was rejected because it would couple the shared service to both platforms.

## Consequences

New messengers require an adapter and presentation mapping rather than a second OCR pipeline. Adapter-specific error handling and formatting remain local, while shared processing and persistence changes affect both messengers.
