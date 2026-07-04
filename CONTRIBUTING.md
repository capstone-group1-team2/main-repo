# Contributing Guide

This document describes the team's development workflow and contribution guidelines for the AISPIRE Capstone project.

## Branch Naming

Create a new branch for every task.

Use the following naming convention:

- `feature/<short-description>` for new features.
- `fix/<short-description>` for bug fixes.
- `docs/<short-description>` for documentation updates.

### Examples

```text
feature/setup-readme
feature/rag-retriever
fix/docker-network
docs/update-readme
```
## Commit Message Convention

Write clear and concise commit messages using the following format:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `refactor:` for code improvements
- `test:` for tests

### Examples

```text
feat: implement document retriever
fix: resolve Docker networking issue
docs: update README
refactor: simplify retrieval pipeline
test: add retrieval unit tests
```
## Pull Request Workflow

1. Create a GitHub Issue for the task.
2. Create a new branch from `main`.
3. Implement the changes and commit them.
4. Push the branch to GitHub.
5. Open a Pull Request (PR).
6. Request reviews from teammates.
7. After **2 approvals**, merge the PR into `main`.
8. Close the related GitHub Issue.
> **Note:** Never commit or push directly to the `main` branch. All changes must go through a Pull Request.

## Code Review Guidelines

Before approving a Pull Request, reviewers should check that:

- The code works as expected.
- The code is readable and follows the project's style.
- No unnecessary files are included.
- The changes do not break existing functionality.

Provide constructive feedback when changes are needed.