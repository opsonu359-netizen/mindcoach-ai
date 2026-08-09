# MindCoach AI

An AI-powered habit-tracking and stress management web app that offers personalized, supportive coaching through natural conversation.

## Live Demo
https://mindcoach-ai.tiiny.site/

## Features
- AI chat for stress management and behavioral coaching
- Instant relapse reframing and proactive check-ins
- Mood tracking
- Urge surfing timer
- Breathing exercises
- Voice output (in progress — improvements ongoing)
- Adaptive response handling based on chat context
- Model fallback logic for improved reliability

## Tech Stack
- HTML, CSS, JavaScript
- Cloudflare Workers (Serverless backend)
- OpenRouter AI API (LLM integration, free-tier models)
- Hosted on tiiny.host

## Architecture
Frontend (tiiny.host) → Backend (Cloudflare Workers) → LLM API (OpenRouter)

Designed for a secure, fast, and cost-efficient serverless flow with fallback endpoint handling for improved uptime.
