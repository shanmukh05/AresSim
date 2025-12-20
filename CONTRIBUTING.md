# Contributing to AresSim 🔴

First off, thank you for considering contributing to AresSim! It's people like you that make AresSim such a great tool for the RL research community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inclusive environment. By participating, you are expected to uphold this standard. Please be respectful, constructive, and supportive of fellow contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/AresSim.git
   cd AresSim
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/AresSim.git
   ```
4. **Install dependencies**:
   ```bash
   npm install
   ```

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples**
- **Describe the observed behavior vs. expected behavior**
- **Include screenshots** if applicable
- **Include your environment details** (OS, Node version, browser)

### 💡 Suggesting Features

Feature suggestions are welcome! Please:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this would be useful** to AresSim users
- **Include mockups or examples** if applicable

### 🔧 Pull Requests

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** following our style guidelines
3. **Test your changes** thoroughly
4. **Commit your changes** with meaningful commit messages
5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request** against the `main` branch

## Development Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Style Guidelines

### TypeScript

- Use TypeScript for all new code
- Define proper types/interfaces (avoid `any`)
- Use functional components with hooks
- Follow existing code patterns

### Code Formatting

- Use 2 spaces for indentation
- Use single quotes for strings
- Add semicolons at the end of statements
- Keep lines under 100 characters when possible

### File Organization

```
AresSim/
├── components/      # React components
├── hooks/           # Custom React hooks
├── assets/          # Static assets
├── types.ts         # TypeScript type definitions
├── constants.ts     # Constants and configuration
└── App.tsx          # Main application component
```

## Commit Messages

Write clear, concise commit messages:

- **Use the present tense** ("Add feature" not "Added feature")
- **Use the imperative mood** ("Move cursor to..." not "Moves cursor to...")
- **Limit the first line to 72 characters**
- **Reference issues and PRs** liberally

### Commit Message Format

```
<type>: <short summary>

<optional body>

<optional footer>
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**
```
feat: add radiation damage system

Implemented radiation damage based on environmental levels:
- Elevated radiation (>5.0 mSv): -0.5 Health/tick
- Dangerous radiation (>8.0 mSv): -2.5 Health/tick

Closes #42
```

## Pull Request Process

1. **Update documentation** if your changes affect it
2. **Add tests** for new functionality
3. **Ensure all tests pass**
4. **Update the README.md** if needed
5. **Request review** from maintainers
6. **Address review feedback** promptly

### PR Title Format

Use the same format as commit messages:
```
feat: add multi-agent support
fix: correct energy depletion calculation
docs: update installation instructions
```

## Questions?

Feel free to open an issue with the `question` label if you have any questions about contributing.

---

Thank you for contributing to AresSim! 🚀🔴
