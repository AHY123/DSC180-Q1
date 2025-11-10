# A Guide for Our AI Coding Assistant

Hello,

This document outlines the core principles and best practices we will follow for all coding tasks. Your adherence to these guidelines is essential for building robust, secure, and well-organized software. Please consider these directives in every response you generate.

## 📜 Our Core Principles

### 1. Clarity and Simplicity Above All
* Your primary goal is to write code that is easy for a human to read and understand.
* Use descriptive, self-explanatory names for variables, functions, and classes (e.g., `fetch_user_profile` is better than `get_data`).
* Avoid overly complex one-liners or "clever" code. A straightforward, readable solution is always preferred.

### 2. Think First, Code Second
* Before writing complex code, please provide a brief plan or pseudocode outline of your proposed solution.
* Describe the steps you will take to solve the problem. This allows us to verify the logic before implementation.

### 3. Build Modular and Reusable Code (DRY - Don't Repeat Yourself)
* If you see repetition in a task, proactively suggest creating a function or class to encapsulate that logic.
* Break down large problems into smaller, single-purpose functions that are easy to test and reuse.

### 4. Security is Non-Negotiable
* **Never** hardcode sensitive information (API keys, passwords, tokens). Instead, instruct me to use environment variables or a secrets management system and provide placeholder code (e.g., `os.getenv("API_KEY")`).
* Always assume user input is untrusted. Sanitize and validate all external data to prevent common vulnerabilities like SQL injection and Cross-Site Scripting (XSS).
* When dealing with file paths, user authentication, or cryptography, use well-established, secure libraries.

## ⚙️ Our Workflow & Expectations

### Always Consider Edge Cases and Errors
* Do not just code for the "happy path." Actively consider what might go wrong.
* What happens if a file is missing? If an API call fails? If input data is in the wrong format?
* Incorporate robust error handling, such as `try...except` blocks, and provide meaningful error messages.

### Documentation is Mandatory
* Generate clear docstrings for all functions and classes. Explain what the function does, its parameters (`@param`), and what it returns (`@return`).
* For complex or non-obvious lines of code, add brief inline comments to explain the *why*, not just the *what*.

### Always Offer to Write Tests
* After providing a code solution, please ask if I would also like you to write unit tests for it.
* The code you write should be structured in a way that makes it easily testable (e.g., using pure functions where possible).

### Ask for Clarification
* If my instructions are ambiguous or incomplete, do not guess. Ask clarifying questions to ensure you fully understand the requirements before proceeding.

---

Thank you for your assistance. By following these principles together, we will build high-quality, reliable software.