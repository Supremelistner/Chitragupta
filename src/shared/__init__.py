"""Shared contracts — canonical types and IDs used across all microservices.

This package defines the stable data contract that all four services
(Document, Model, Validator, Web Search) share.  Services import from
here rather than defining their own ID types or status enums.

Ownership:
    This package is owned by the project as a whole.
    Individual services may extend these types but must not redefine them.
"""
