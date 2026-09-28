"""robots.txt parsing and matching, written to RFC 9309 (the Robots Exclusion Protocol).

Python's `urllib.robotparser` applies rules in file order and ignores `*` and `$` wildcards, so
it disagrees with Google on files like:

    User-agent: *
    Disallow: /
    Allow: /public$

RFC 9309 says the *longest* matching rule wins and `allow` wins a tie. This module implements
that, plus the RFC's handling of an unreachable robots.txt:

    4xx (unavailable)  -> the crawler may fetch anything
    5xx / network error -> assume everything is disallowed (the server may be struggling)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import quote, unquote


@dataclass(frozen=True)
class Rule:
    allow: bool
    pattern: str

    @property
    def specificity(self) -> int:
        return len(self.pattern)

    def matches(self, path: str) -> bool:
        return _compile(self.pattern).match(path) is not None


@lru_cache(maxsize=4096)
def _compile(pattern: str) -> re.Pattern:
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    regex = ".*".join(re.escape(part) for part in body.split("*"))
    return re.compile(regex + ("$" if anchored else ""), re.S)


def _encode(path: str) -> str:
    # Compare paths and patterns in one canonical percent-encoding.
    return quote(unquote(path), safe="/:@!$&'()*+,;=-._~?%")


@dataclass
class Group:
    agents: list[str] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    crawl_delay: float | None = None


@dataclass
class RobotsPolicy:
    groups: list[Group] = field(default_factory=list)
    sitemaps: list[str] = field(default_factory=list)
    mode: str = "rules"  # "rules" | "allow_all" | "disallow_all"

    @classmethod
    def allow_all(cls) -> RobotsPolicy:
        return cls(mode="allow_all")

    @classmethod
    def disallow_all(cls) -> RobotsPolicy:
        return cls(mode="disallow_all")

    @classmethod
    def from_status(cls, status: int, text: str) -> RobotsPolicy:
        if 200 <= status < 300:
            return parse(text)
        if 400 <= status < 500:
            return cls.allow_all()
        return cls.disallow_all()

    def _groups_for(self, agent: str) -> list[Group]:
        token = agent.split("/")[0].strip().lower()
        exact = [g for g in self.groups if token in g.agents]
        return exact or [g for g in self.groups if "*" in g.agents]

    def can_fetch(self, agent: str, path: str) -> bool:
        if self.mode != "rules":
            return self.mode == "allow_all"
        path = _encode(path or "/")
        if path == "/robots.txt":
            return True
        best: Rule | None = None
        for group in self._groups_for(agent):
            for rule in group.rules:
                if not rule.pattern:  # "Disallow:" with no value means allow everything
                    continue
                if rule.matches(path) and (
                    best is None
                    or rule.specificity > best.specificity
                    or (rule.specificity == best.specificity and rule.allow and not best.allow)
                ):
                    best = rule
        return True if best is None else best.allow

    def crawl_delay(self, agent: str) -> float | None:
        delays = [g.crawl_delay for g in self._groups_for(agent) if g.crawl_delay is not None]
        return max(delays) if delays else None


def parse(text: str) -> RobotsPolicy:
    policy = RobotsPolicy()
    current: Group | None = None
    last_was_agent = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = (s.strip() for s in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if current is None or not last_was_agent:
                current = Group()
                policy.groups.append(current)
            current.agents.append(value.lower())
            last_was_agent = True
            continue
        last_was_agent = False
        if key == "sitemap":
            policy.sitemaps.append(value)
        elif current is None:
            continue  # rules before any user-agent line belong to no group
        elif key in ("allow", "disallow"):
            current.rules.append(Rule(allow=key == "allow", pattern=_encode(value) if value else ""))
        elif key == "crawl-delay":  # not in RFC 9309, but widely used
            try:
                current.crawl_delay = float(value)
            except ValueError:
                pass
    return policy
