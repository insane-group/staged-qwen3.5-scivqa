# Vulture whitelist for staged_qwen3_5_scivqa
#
# This file lists intentionally unused code that vulture should not flag.
# Includes SMT grammars, prompts, and other strings that are used
# dynamically (via format() or regex) rather than direct import.

# SMT grammar strings used via CFG() constructor
SMT_LIB_GRAMMAR_PASS1A

# SMT prompt templates used via .format()
PROMPT_TEMPLATE_PASS1A
PROMPT_TEMPLATE_PASS1B
PROMPT_TEMPLATE_PASS2
PROMPT_TEMPLATE_PLANNING
PROMPT_TEMPLATE_REFLECTION
PROMPT_REWRITE
PROMPT_SUMMARY
PROMPT_TABLE

# SMT examples used via dict lookup
EXAMPLES_PASS1A
EXAMPLES_PASS1B
EXAMPLES_PASS2

# Preamble used via string concatenation
PREAMBLE

# pydantic-settings BaseSettings not directly instantiated in this module
BaseSettings

# Config constants used dynamically via dict lookups
TOKEN_BUDGETS

# settings_customise_sources parameters required by pydantic-settings interface
cls
file_secret_settings

# progress_context parameter used in Rich TextColumn format string
description
