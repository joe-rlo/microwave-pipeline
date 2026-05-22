"""GitHub REST API client.

Wraps the handful of `api.github.com` endpoints the tool layer needs to
let the LLM "walk your repos and report": list owned repos, fetch a
per-repo summary (README + recent commits + open PRs/issues + languages),
and surface a recent-activity feed.

Read-only by design. The PAT the user provides is fine-grained and
read-scoped (Contents: Read, Metadata: Read, Pull requests: Read); we
never call a write endpoint so even a misuse can't damage anything.

What we deliberately do NOT do here:
- No Agent SDK imports. Pure HTTP; the tool wrapper in
  `src/tools/github.py` adapts these methods to the SDK's `@tool` shape.
- No `gh` CLI shell-out. Goes straight to api.github.com so the user's
  local `gh auth` state stays untouched — the PAT only lives in
  MicrowaveOS env, never in `gh`'s credential store.
- No pagination beyond the first page. Personal-account volume rarely
  needs it; the `limit` argument caps results to keep the LLM's context
  budget honest. If a future use case needs deep pagination, add it
  behind an explicit opt-in flag.
- No retries. GitHub returns crisp 4xx/5xx — surface them and let the
  LLM decide whether to retry (it usually shouldn't).
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any

import aiohttp

log = logging.getLogger(__name__)


GITHUB_API_BASE = "https://api.github.com"
DEFAULT_TIMEOUT_SEC = 20.0
# README excerpt cap. Long enough to convey what the project is, short
# enough that summarizing five repos doesn't blow a turn's context.
README_EXCERPT_CHARS = 2000


class GitHubError(RuntimeError):
    """Raised on any GitHub API failure — auth, rate limit, network, etc.

    Carries the HTTP status (when available) and response body so the
    tool wrapper can surface a useful error to the LLM and the user.
    """

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class RepoSummary:
    """Shape returned for each repo in list / activity calls.

    Stays a flat dict-shaped object (rather than the raw API response)
    so the tool layer can serialize it directly to JSON for the LLM.
    """

    full_name: str
    name: str
    description: str | None
    language: str | None
    default_branch: str
    pushed_at: str | None
    updated_at: str | None
    stargazers_count: int
    open_issues_count: int  # GitHub counts PRs in this; we expose separately when relevant
    private: bool
    html_url: str
    fork: bool
    archived: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "full_name": self.full_name,
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "default_branch": self.default_branch,
            "pushed_at": self.pushed_at,
            "updated_at": self.updated_at,
            "stargazers_count": self.stargazers_count,
            "open_issues_count": self.open_issues_count,
            "private": self.private,
            "html_url": self.html_url,
            "fork": self.fork,
            "archived": self.archived,
        }


@dataclass
class CommitSummary:
    sha: str
    message: str
    author: str | None
    date: str | None
    html_url: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "sha": self.sha[:10],
            "message": self.message,
            "author": self.author,
            "date": self.date,
            "html_url": self.html_url,
        }


@dataclass
class PullOrIssueSummary:
    """Shared shape for open PRs and issues."""

    number: int
    title: str
    state: str
    author: str | None
    created_at: str | None
    updated_at: str | None
    html_url: str
    is_pull_request: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "state": self.state,
            "author": self.author,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "html_url": self.html_url,
            "is_pull_request": self.is_pull_request,
        }


@dataclass
class RepoDeepSummary:
    """What `get_repo_summary` returns — a multi-call composite."""

    repo: RepoSummary
    readme_excerpt: str | None
    languages: dict[str, int] = field(default_factory=dict)
    recent_commits: list[CommitSummary] = field(default_factory=list)
    open_pulls: list[PullOrIssueSummary] = field(default_factory=list)
    open_issues: list[PullOrIssueSummary] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "repo": self.repo.to_payload(),
            "readme_excerpt": self.readme_excerpt,
            "languages": self.languages,
            "recent_commits": [c.to_payload() for c in self.recent_commits],
            "open_pulls": [p.to_payload() for p in self.open_pulls],
            "open_issues": [i.to_payload() for i in self.open_issues],
        }


@dataclass
class ActivityEvent:
    """A flattened event from `/users/{user}/events`.

    GitHub's events feed includes PushEvent, PullRequestEvent, IssuesEvent,
    CreateEvent, etc. We keep the type + repo + a short human label and
    let the LLM make sense of the rest.
    """

    type: str
    repo: str
    created_at: str | None
    summary: str
    html_url: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "repo": self.repo,
            "created_at": self.created_at,
            "summary": self.summary,
            "html_url": self.html_url,
        }


class GitHubClient:
    """Async client for the GitHub REST API."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = GITHUB_API_BASE,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        session: aiohttp.ClientSession | None = None,
    ):
        if not token:
            raise ValueError("GitHub token required")
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._session = session  # tests inject a mock; prod opens per-call

    # --- Low-level request ---

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "MicrowaveOS-github-tool",
        }

        own_session = self._session is None or self._session.closed
        if own_session:
            from src.channels._http import make_session
            session = make_session(
                timeout=aiohttp.ClientTimeout(total=self.timeout_sec)
            )
        else:
            session = self._session

        try:
            async with session.request(
                method, url, params=params, headers=headers
            ) as resp:
                text = await resp.text()
                if resp.status == 404:
                    raise GitHubError(
                        f"GitHub 404: {path} not found",
                        status=404,
                        body=text,
                    )
                if resp.status >= 400:
                    log.warning("GitHub %s %s -> %s: %s", method, path, resp.status, text[:500])
                    raise GitHubError(
                        f"GitHub {resp.status} on {path}",
                        status=resp.status,
                        body=text,
                    )
                try:
                    return await _parse_json(resp, text)
                except ValueError as e:
                    raise GitHubError(
                        f"Malformed GitHub response on {path}: {e}",
                        body=text,
                    ) from e
        except aiohttp.ClientError as e:
            raise GitHubError(f"Network error calling GitHub {path}: {e}") from e
        finally:
            if own_session:
                await session.close()

    # --- High-level methods ---

    async def get_authenticated_user(self) -> dict[str, Any]:
        """`GET /user` — used to discover the username for activity feeds."""
        data = await self._request("GET", "/user")
        if not isinstance(data, dict):
            raise GitHubError("Unexpected /user response shape", body=str(data))
        return data

    async def list_user_repos(
        self,
        *,
        visibility: str = "all",  # "all" | "public" | "private"
        affiliation: str = "owner",  # "owner" | "collaborator" | "organization_member"
        sort: str = "pushed",  # "created" | "updated" | "pushed" | "full_name"
        limit: int = 30,
    ) -> list[RepoSummary]:
        """`GET /user/repos` — repos the PAT can see for the authenticated user.

        Capped at 100 per page (GitHub's max). The `limit` param trims
        further so the LLM doesn't drown in a 200-repo dump.
        """
        per_page = min(max(limit, 1), 100)
        params = {
            "visibility": visibility,
            "affiliation": affiliation,
            "sort": sort,
            "per_page": per_page,
        }
        data = await self._request("GET", "/user/repos", params=params)
        if not isinstance(data, list):
            raise GitHubError("Unexpected /user/repos response shape", body=str(data))
        return [_repo_summary_from_api(r) for r in data[:limit] if isinstance(r, dict)]

    async def get_repo(self, full_name: str) -> RepoSummary:
        """`GET /repos/{owner}/{repo}` — base metadata for one repo."""
        owner, name = _split_full_name(full_name)
        data = await self._request("GET", f"/repos/{owner}/{name}")
        if not isinstance(data, dict):
            raise GitHubError(f"Unexpected /repos response shape for {full_name}", body=str(data))
        return _repo_summary_from_api(data)

    async def get_readme_excerpt(self, full_name: str) -> str | None:
        """`GET /repos/{owner}/{repo}/readme` — decoded + truncated.

        Returns None if the repo has no README. The endpoint returns
        base64-encoded content; we decode and trim to README_EXCERPT_CHARS
        so the LLM gets enough to summarize without paying for the whole
        file.
        """
        owner, name = _split_full_name(full_name)
        try:
            data = await self._request("GET", f"/repos/{owner}/{name}/readme")
        except GitHubError as e:
            if e.status == 404:
                return None
            raise
        if not isinstance(data, dict):
            return None
        encoded = data.get("content") or ""
        if not isinstance(encoded, str):
            return None
        try:
            decoded = base64.b64decode(encoded).decode("utf-8", errors="replace")
        except Exception:
            return None
        if len(decoded) > README_EXCERPT_CHARS:
            return decoded[:README_EXCERPT_CHARS] + "\n…[truncated]"
        return decoded

    async def list_recent_commits(
        self, full_name: str, *, limit: int = 5
    ) -> list[CommitSummary]:
        owner, name = _split_full_name(full_name)
        params = {"per_page": min(max(limit, 1), 30)}
        data = await self._request(
            "GET", f"/repos/{owner}/{name}/commits", params=params
        )
        if not isinstance(data, list):
            return []
        out: list[CommitSummary] = []
        for c in data[:limit]:
            if not isinstance(c, dict):
                continue
            commit = c.get("commit") or {}
            author_info = commit.get("author") or {}
            message = (commit.get("message") or "").splitlines()[0][:200]
            out.append(
                CommitSummary(
                    sha=c.get("sha") or "",
                    message=message,
                    author=author_info.get("name"),
                    date=author_info.get("date"),
                    html_url=c.get("html_url") or "",
                )
            )
        return out

    async def list_open_pulls(
        self, full_name: str, *, limit: int = 10
    ) -> list[PullOrIssueSummary]:
        owner, name = _split_full_name(full_name)
        params = {"state": "open", "per_page": min(max(limit, 1), 30)}
        data = await self._request(
            "GET", f"/repos/{owner}/{name}/pulls", params=params
        )
        if not isinstance(data, list):
            return []
        return [
            _pull_or_issue_from_api(p, is_pull_request=True)
            for p in data[:limit]
            if isinstance(p, dict)
        ]

    async def list_open_issues(
        self, full_name: str, *, limit: int = 10
    ) -> list[PullOrIssueSummary]:
        """`GET /repos/{owner}/{repo}/issues` — open issues, PRs filtered out.

        GitHub's `/issues` endpoint returns PRs too (every PR is an issue
        under the hood). We strip them so the LLM gets a clean issue list
        and can pull PRs separately via list_open_pulls.
        """
        owner, name = _split_full_name(full_name)
        # Fetch a larger page since we'll filter; GitHub's per_page max is 100.
        params = {"state": "open", "per_page": min(max(limit * 3, 10), 100)}
        data = await self._request(
            "GET", f"/repos/{owner}/{name}/issues", params=params
        )
        if not isinstance(data, list):
            return []
        out: list[PullOrIssueSummary] = []
        for i in data:
            if not isinstance(i, dict):
                continue
            if i.get("pull_request"):
                continue  # skip PRs masquerading as issues
            out.append(_pull_or_issue_from_api(i, is_pull_request=False))
            if len(out) >= limit:
                break
        return out

    async def get_languages(self, full_name: str) -> dict[str, int]:
        owner, name = _split_full_name(full_name)
        data = await self._request("GET", f"/repos/{owner}/{name}/languages")
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(v, int)}

    async def get_repo_summary(self, full_name: str) -> RepoDeepSummary:
        """Composite: metadata + README excerpt + commits + open PRs/issues + languages.

        One method on the client side keeps the tool wrapper trivial.
        Each sub-call is best-effort: a 404 on README or a permission
        error on issues shouldn't kill the whole summary.
        """
        repo = await self.get_repo(full_name)
        readme = await self._best_effort(self.get_readme_excerpt(full_name), default=None)
        languages = await self._best_effort(self.get_languages(full_name), default={})
        commits = await self._best_effort(self.list_recent_commits(full_name, limit=5), default=[])
        pulls = await self._best_effort(self.list_open_pulls(full_name, limit=10), default=[])
        issues = await self._best_effort(self.list_open_issues(full_name, limit=10), default=[])
        return RepoDeepSummary(
            repo=repo,
            readme_excerpt=readme,
            languages=languages,
            recent_commits=commits,
            open_pulls=pulls,
            open_issues=issues,
        )

    async def list_recent_activity(
        self, username: str, *, limit: int = 30
    ) -> list[ActivityEvent]:
        """`GET /users/{user}/events` — recent push/PR/issue/etc events.

        GitHub's events feed mixes types; we flatten the common ones into
        a uniform shape so the LLM can scan the timeline easily. Events
        older than 90 days are filtered server-side by GitHub; we don't
        add a `since=` param because the endpoint doesn't accept one.
        """
        params = {"per_page": min(max(limit, 1), 100)}
        data = await self._request(
            "GET", f"/users/{username}/events", params=params
        )
        if not isinstance(data, list):
            return []
        return [
            evt
            for evt in (_event_from_api(e) for e in data[:limit] if isinstance(e, dict))
            if evt is not None
        ]

    @staticmethod
    async def _best_effort(coro, *, default):
        """Await a coroutine and return `default` if it raises GitHubError.

        Keeps `get_repo_summary` resilient — one sub-endpoint failing
        (e.g. empty repo with no commits → 409) shouldn't sink the whole
        summary. We still log so debugging stays sane.
        """
        try:
            return await coro
        except GitHubError as e:
            log.info("github sub-call failed (continuing): %s", e)
            return default


# --- Helpers ---


def _split_full_name(full_name: str) -> tuple[str, str]:
    """Parse 'owner/repo' into ('owner', 'repo'). Raises ValueError otherwise."""
    if not isinstance(full_name, str) or "/" not in full_name:
        raise ValueError(f"Expected 'owner/repo', got {full_name!r}")
    owner, _, name = full_name.partition("/")
    owner = owner.strip()
    name = name.strip()
    if not owner or not name:
        raise ValueError(f"Expected 'owner/repo', got {full_name!r}")
    return owner, name


def _repo_summary_from_api(data: dict[str, Any]) -> RepoSummary:
    return RepoSummary(
        full_name=data.get("full_name") or "",
        name=data.get("name") or "",
        description=data.get("description"),
        language=data.get("language"),
        default_branch=data.get("default_branch") or "main",
        pushed_at=data.get("pushed_at"),
        updated_at=data.get("updated_at"),
        stargazers_count=int(data.get("stargazers_count") or 0),
        open_issues_count=int(data.get("open_issues_count") or 0),
        private=bool(data.get("private")),
        html_url=data.get("html_url") or "",
        fork=bool(data.get("fork")),
        archived=bool(data.get("archived")),
    )


def _pull_or_issue_from_api(
    data: dict[str, Any], *, is_pull_request: bool
) -> PullOrIssueSummary:
    user = data.get("user") or {}
    return PullOrIssueSummary(
        number=int(data.get("number") or 0),
        title=(data.get("title") or "")[:300],
        state=data.get("state") or "open",
        author=user.get("login") if isinstance(user, dict) else None,
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
        html_url=data.get("html_url") or "",
        is_pull_request=is_pull_request,
    )


def _event_from_api(data: dict[str, Any]) -> ActivityEvent | None:
    """Flatten one events-feed row into an ActivityEvent.

    GitHub's events shape varies by `type`; we extract a short human
    summary per common type and skip the rest. Returning None means
    "unrecognized type, drop it" — keeps the LLM's view clean.
    """
    evt_type = data.get("type") or ""
    repo_obj = data.get("repo") or {}
    repo_name = repo_obj.get("name") if isinstance(repo_obj, dict) else ""
    payload = data.get("payload") or {}
    created_at = data.get("created_at")

    summary: str | None = None
    html_url: str | None = None

    if evt_type == "PushEvent":
        commits = payload.get("commits") or []
        commit_count = len(commits) if isinstance(commits, list) else 0
        ref = payload.get("ref") or ""
        branch = ref.rsplit("/", 1)[-1] if ref else "?"
        summary = f"Pushed {commit_count} commit(s) to {branch}"
        if commit_count and isinstance(commits[0], dict):
            first_msg = (commits[0].get("message") or "").splitlines()[0][:120]
            summary += f": {first_msg}"
    elif evt_type == "PullRequestEvent":
        action = payload.get("action") or "updated"
        pr = payload.get("pull_request") or {}
        number = pr.get("number")
        title = (pr.get("title") or "")[:120]
        summary = f"PR #{number} {action}: {title}"
        html_url = pr.get("html_url") if isinstance(pr, dict) else None
    elif evt_type == "IssuesEvent":
        action = payload.get("action") or "updated"
        issue = payload.get("issue") or {}
        number = issue.get("number")
        title = (issue.get("title") or "")[:120]
        summary = f"Issue #{number} {action}: {title}"
        html_url = issue.get("html_url") if isinstance(issue, dict) else None
    elif evt_type == "IssueCommentEvent":
        issue = payload.get("issue") or {}
        number = issue.get("number")
        title = (issue.get("title") or "")[:120]
        summary = f"Commented on #{number}: {title}"
    elif evt_type == "CreateEvent":
        ref_type = payload.get("ref_type") or ""
        ref = payload.get("ref") or ""
        summary = f"Created {ref_type} {ref}".strip()
    elif evt_type == "DeleteEvent":
        ref_type = payload.get("ref_type") or ""
        ref = payload.get("ref") or ""
        summary = f"Deleted {ref_type} {ref}".strip()
    elif evt_type == "ReleaseEvent":
        action = payload.get("action") or "released"
        release = payload.get("release") or {}
        tag = release.get("tag_name") or ""
        summary = f"Release {action}: {tag}"
    elif evt_type == "ForkEvent":
        summary = "Forked the repo"
    elif evt_type == "WatchEvent":
        summary = "Starred the repo"
    else:
        return None

    if not summary:
        return None
    return ActivityEvent(
        type=evt_type,
        repo=repo_name or "",
        created_at=created_at,
        summary=summary,
        html_url=html_url,
    )


async def _parse_json(resp: aiohttp.ClientResponse, text: str) -> Any:
    """Parse JSON without trusting Content-Type.

    Same defensive shape as the Instacart client — some edge gateways
    strip the header, and we've already buffered the text for logging.
    """
    import json

    if not text:
        raise ValueError("empty response body")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from e
