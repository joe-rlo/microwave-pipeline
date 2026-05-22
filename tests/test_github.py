"""Tests for the GitHub integration + tool wrappers.

Covers:
- Client happy paths (list repos, repo summary composite, recent activity)
- Client error paths (4xx, 404, malformed JSON, bad full_name)
- Event flattening (PushEvent, PullRequestEvent, IssuesEvent, unknown)
- Issues-vs-PR filtering (GitHub returns PRs from /issues; we strip them)
- Tool handler MCP shape, error surface
- Registry: empty when token missing, populated when set

Same shape as test_instacart.py — we mock aiohttp session.request and
exercise the handler functions directly without booting the SDK.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.integrations.github import (
    ActivityEvent,
    CommitSummary,
    GitHubClient,
    GitHubError,
    PullOrIssueSummary,
    RepoDeepSummary,
    RepoSummary,
    _event_from_api,
    _split_full_name,
)


# --- helpers ---


def _make_response(status: int, body):
    text_body = body if isinstance(body, str) else json.dumps(body)
    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=text_body)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _stub_session_with_queue(*response_cms) -> MagicMock:
    """Build a session whose successive `.request(...)` calls return the
    given response context managers in order. Lets us script multi-call
    flows (`get_repo_summary` makes 6 calls)."""
    session = MagicMock()
    session.closed = False
    queue = list(response_cms)

    def _request(*args, **kwargs):
        if not queue:
            raise AssertionError("Unexpected extra request to stubbed session")
        return queue.pop(0)

    session.request = MagicMock(side_effect=_request)
    session.close = AsyncMock()
    return session


# --- _split_full_name ---


class TestSplitFullName:
    def test_happy(self):
        assert _split_full_name("octocat/Hello-World") == ("octocat", "Hello-World")

    def test_extra_slashes_partition(self):
        # `partition` keeps everything after the first slash as the name,
        # which matches GitHub's actual API behavior (repo names can't
        # contain slashes but we shouldn't crash if they did).
        assert _split_full_name("o/r/x") == ("o", "r/x")

    def test_missing_slash_raises(self):
        with pytest.raises(ValueError):
            _split_full_name("octocat")

    def test_empty_parts_raise(self):
        with pytest.raises(ValueError):
            _split_full_name("/repo")
        with pytest.raises(ValueError):
            _split_full_name("owner/")


# --- Event flattening ---


class TestEventFromApi:
    def test_push_event(self):
        evt = _event_from_api(
            {
                "type": "PushEvent",
                "repo": {"name": "joe/microwave-os"},
                "created_at": "2026-05-01T12:00:00Z",
                "payload": {
                    "ref": "refs/heads/main",
                    "commits": [{"message": "fix: thinking nudge timing"}],
                },
            }
        )
        assert evt is not None
        assert evt.type == "PushEvent"
        assert evt.repo == "joe/microwave-os"
        assert "main" in evt.summary
        assert "thinking nudge" in evt.summary

    def test_pull_request_event(self):
        evt = _event_from_api(
            {
                "type": "PullRequestEvent",
                "repo": {"name": "joe/microwave-os"},
                "created_at": "2026-05-01T12:00:00Z",
                "payload": {
                    "action": "opened",
                    "pull_request": {
                        "number": 42,
                        "title": "Add GitHub walker tool",
                        "html_url": "https://github.com/joe/microwave-os/pull/42",
                    },
                },
            }
        )
        assert evt is not None
        assert "#42" in evt.summary
        assert "opened" in evt.summary
        assert evt.html_url == "https://github.com/joe/microwave-os/pull/42"

    def test_issues_event(self):
        evt = _event_from_api(
            {
                "type": "IssuesEvent",
                "repo": {"name": "joe/microwave-os"},
                "payload": {
                    "action": "closed",
                    "issue": {"number": 7, "title": "Signal addendum bug"},
                },
            }
        )
        assert evt is not None
        assert "Issue #7" in evt.summary
        assert "closed" in evt.summary

    def test_unknown_event_dropped(self):
        assert _event_from_api({"type": "SponsorshipEvent", "repo": {"name": "x/y"}}) is None

    def test_missing_type_dropped(self):
        assert _event_from_api({}) is None


# --- Client: list_user_repos ---


@pytest.mark.asyncio
class TestListUserRepos:
    async def test_happy_path(self):
        body = [
            {
                "full_name": "joe/microwave-os",
                "name": "microwave-os",
                "description": "cognitive agent runtime",
                "language": "Python",
                "default_branch": "main",
                "pushed_at": "2026-05-10T12:00:00Z",
                "updated_at": "2026-05-10T12:00:00Z",
                "stargazers_count": 12,
                "open_issues_count": 3,
                "private": False,
                "html_url": "https://github.com/joe/microwave-os",
                "fork": False,
                "archived": False,
            },
            {
                "full_name": "joe/old-thing",
                "name": "old-thing",
                "description": None,
                "language": None,
                "default_branch": "master",
                "pushed_at": None,
                "updated_at": None,
                "stargazers_count": 0,
                "open_issues_count": 0,
                "private": True,
                "html_url": "https://github.com/joe/old-thing",
                "fork": False,
                "archived": True,
            },
        ]
        session = _stub_session_with_queue(_make_response(200, body))
        client = GitHubClient(token="fake", session=session)

        repos = await client.list_user_repos(limit=10)

        assert len(repos) == 2
        assert isinstance(repos[0], RepoSummary)
        assert repos[0].full_name == "joe/microwave-os"
        assert repos[0].stargazers_count == 12
        assert repos[1].archived is True
        # Verify request shape
        call = session.request.call_args
        assert call.args[0] == "GET"
        assert call.args[1].endswith("/user/repos")
        assert call.kwargs["headers"]["Authorization"] == "Bearer fake"
        assert call.kwargs["params"]["sort"] == "pushed"

    async def test_limit_trims_oversize_page(self):
        body = [
            {"full_name": f"joe/repo{i}", "name": f"repo{i}", "default_branch": "main"}
            for i in range(50)
        ]
        session = _stub_session_with_queue(_make_response(200, body))
        client = GitHubClient(token="fake", session=session)

        repos = await client.list_user_repos(limit=5)
        assert len(repos) == 5

    async def test_4xx_raises(self):
        session = _stub_session_with_queue(_make_response(401, "bad token"))
        client = GitHubClient(token="bad", session=session)

        with pytest.raises(GitHubError) as exc_info:
            await client.list_user_repos()
        assert exc_info.value.status == 401
        assert "bad token" in (exc_info.value.body or "")

    async def test_404_raises_with_status(self):
        session = _stub_session_with_queue(_make_response(404, "nope"))
        client = GitHubClient(token="fake", session=session)
        with pytest.raises(GitHubError) as exc_info:
            await client.list_user_repos()
        assert exc_info.value.status == 404

    async def test_malformed_json_raises(self):
        session = _stub_session_with_queue(_make_response(200, "not json"))
        client = GitHubClient(token="fake", session=session)
        with pytest.raises(GitHubError) as exc_info:
            await client.list_user_repos()
        assert "Malformed" in str(exc_info.value)

    async def test_unexpected_shape_raises(self):
        session = _stub_session_with_queue(_make_response(200, {"not": "a list"}))
        client = GitHubClient(token="fake", session=session)
        with pytest.raises(GitHubError):
            await client.list_user_repos()


# --- Client: list_open_issues filtering ---


@pytest.mark.asyncio
class TestListOpenIssues:
    async def test_filters_out_pull_requests(self):
        body = [
            {"number": 1, "title": "real issue", "state": "open"},
            {"number": 2, "title": "actually a PR", "state": "open", "pull_request": {"url": "..."}},
            {"number": 3, "title": "another issue", "state": "open"},
        ]
        session = _stub_session_with_queue(_make_response(200, body))
        client = GitHubClient(token="fake", session=session)

        issues = await client.list_open_issues("joe/microwave-os", limit=10)

        numbers = [i.number for i in issues]
        assert numbers == [1, 3]
        assert all(i.is_pull_request is False for i in issues)

    async def test_limit_respected_after_filter(self):
        # 5 issues + 5 PRs interleaved; asking for 2 should give 2 issues.
        body = []
        for n in range(1, 11):
            entry = {"number": n, "title": f"item {n}", "state": "open"}
            if n % 2 == 0:
                entry["pull_request"] = {"url": "..."}
            body.append(entry)
        session = _stub_session_with_queue(_make_response(200, body))
        client = GitHubClient(token="fake", session=session)

        issues = await client.list_open_issues("joe/microwave-os", limit=2)
        assert len(issues) == 2
        assert [i.number for i in issues] == [1, 3]


# --- Client: get_readme_excerpt ---


@pytest.mark.asyncio
class TestReadmeExcerpt:
    async def test_decodes_base64(self):
        import base64

        content = base64.b64encode(b"# Hello\n\nWorld").decode()
        session = _stub_session_with_queue(
            _make_response(200, {"content": content, "encoding": "base64"})
        )
        client = GitHubClient(token="fake", session=session)

        excerpt = await client.get_readme_excerpt("joe/microwave-os")
        assert excerpt == "# Hello\n\nWorld"

    async def test_truncates_long_readme(self):
        import base64
        from src.integrations.github import README_EXCERPT_CHARS

        long = "x" * (README_EXCERPT_CHARS + 500)
        content = base64.b64encode(long.encode()).decode()
        session = _stub_session_with_queue(_make_response(200, {"content": content}))
        client = GitHubClient(token="fake", session=session)

        excerpt = await client.get_readme_excerpt("joe/microwave-os")
        assert excerpt is not None
        assert excerpt.endswith("[truncated]")
        assert len(excerpt) <= README_EXCERPT_CHARS + len("\n…[truncated]")

    async def test_404_returns_none(self):
        session = _stub_session_with_queue(_make_response(404, "no readme"))
        client = GitHubClient(token="fake", session=session)
        assert await client.get_readme_excerpt("joe/empty") is None


# --- Client: get_repo_summary composite ---


@pytest.mark.asyncio
class TestRepoSummaryComposite:
    async def test_combines_subcalls(self):
        import base64

        repo_body = {
            "full_name": "joe/microwave-os",
            "name": "microwave-os",
            "description": "thing",
            "language": "Python",
            "default_branch": "main",
            "pushed_at": "2026-05-10T12:00:00Z",
            "stargazers_count": 5,
            "open_issues_count": 2,
            "private": False,
            "html_url": "https://github.com/joe/microwave-os",
        }
        readme_body = {"content": base64.b64encode(b"# microwave-os").decode()}
        langs_body = {"Python": 12345, "TypeScript": 678}
        commits_body = [
            {
                "sha": "abc123def456",
                "html_url": "https://github.com/x/y/commit/abc",
                "commit": {
                    "message": "feat: ship it\n\nlong body",
                    "author": {"name": "Joe", "date": "2026-05-10T12:00:00Z"},
                },
            }
        ]
        pulls_body = [
            {
                "number": 7,
                "title": "Add github tool",
                "state": "open",
                "user": {"login": "joe"},
                "created_at": "2026-05-09T12:00:00Z",
                "updated_at": "2026-05-10T12:00:00Z",
                "html_url": "https://github.com/joe/microwave-os/pull/7",
            }
        ]
        issues_body = [
            {
                "number": 3,
                "title": "real issue",
                "state": "open",
                "user": {"login": "joe"},
                "html_url": "https://github.com/joe/microwave-os/issues/3",
            },
            # one PR masquerading as an issue — must be filtered
            {
                "number": 7,
                "title": "Add github tool",
                "state": "open",
                "pull_request": {"url": "..."},
            },
        ]

        # Order matters — get_repo_summary calls in this order:
        # get_repo, get_readme_excerpt, get_languages, list_recent_commits,
        # list_open_pulls, list_open_issues
        session = _stub_session_with_queue(
            _make_response(200, repo_body),
            _make_response(200, readme_body),
            _make_response(200, langs_body),
            _make_response(200, commits_body),
            _make_response(200, pulls_body),
            _make_response(200, issues_body),
        )
        client = GitHubClient(token="fake", session=session)

        summary = await client.get_repo_summary("joe/microwave-os")

        assert isinstance(summary, RepoDeepSummary)
        assert summary.repo.full_name == "joe/microwave-os"
        assert summary.readme_excerpt == "# microwave-os"
        assert summary.languages == {"Python": 12345, "TypeScript": 678}
        assert len(summary.recent_commits) == 1
        assert summary.recent_commits[0].message == "feat: ship it"
        assert len(summary.open_pulls) == 1
        assert summary.open_pulls[0].number == 7
        assert len(summary.open_issues) == 1  # PR was filtered out
        assert summary.open_issues[0].number == 3

    async def test_subcall_failure_doesnt_sink_summary(self):
        """A 404 on README or a 409 on commits (empty repo) shouldn't
        prevent the rest of the summary from coming back."""
        repo_body = {
            "full_name": "joe/empty-repo",
            "name": "empty-repo",
            "default_branch": "main",
        }
        session = _stub_session_with_queue(
            _make_response(200, repo_body),         # get_repo
            _make_response(404, "no readme"),        # readme → None
            _make_response(409, "empty repo"),       # languages → {}
            _make_response(409, "empty repo"),       # commits → []
            _make_response(200, []),                  # pulls
            _make_response(200, []),                  # issues
        )
        client = GitHubClient(token="fake", session=session)
        summary = await client.get_repo_summary("joe/empty-repo")
        assert summary.readme_excerpt is None
        assert summary.languages == {}
        assert summary.recent_commits == []
        assert summary.open_pulls == []
        assert summary.open_issues == []


# --- Client: list_recent_activity ---


@pytest.mark.asyncio
class TestRecentActivity:
    async def test_flattens_events(self):
        body = [
            {
                "type": "PushEvent",
                "repo": {"name": "joe/a"},
                "created_at": "2026-05-10T12:00:00Z",
                "payload": {"ref": "refs/heads/main", "commits": [{"message": "x"}]},
            },
            {"type": "SponsorshipEvent", "repo": {"name": "joe/a"}, "payload": {}},  # dropped
            {
                "type": "PullRequestEvent",
                "repo": {"name": "joe/b"},
                "created_at": "2026-05-09T12:00:00Z",
                "payload": {
                    "action": "opened",
                    "pull_request": {"number": 1, "title": "t"},
                },
            },
        ]
        session = _stub_session_with_queue(_make_response(200, body))
        client = GitHubClient(token="fake", session=session)

        events = await client.list_recent_activity("joe", limit=10)
        # SponsorshipEvent dropped → 2 events surface
        assert len(events) == 2
        assert events[0].type == "PushEvent"
        assert events[1].type == "PullRequestEvent"


# --- Tool handlers ---


@pytest.mark.asyncio
class TestToolHandlers:
    async def test_list_repos_returns_mcp_shape(self):
        from src.tools.github import _handle_list_repos

        fake_repos = [
            RepoSummary(
                full_name="joe/microwave-os",
                name="microwave-os",
                description="thing",
                language="Python",
                default_branch="main",
                pushed_at="2026-05-10T12:00:00Z",
                updated_at="2026-05-10T12:00:00Z",
                stargazers_count=5,
                open_issues_count=1,
                private=False,
                html_url="https://github.com/joe/microwave-os",
                fork=False,
                archived=False,
            ),
        ]
        with patch("src.tools.github.GitHubClient") as MockClient:
            instance = MockClient.return_value
            instance.list_user_repos = AsyncMock(return_value=fake_repos)

            result = await _handle_list_repos({"limit": 10}, "tok")

        assert "is_error" not in result
        payload = json.loads(result["content"][0]["text"])
        assert payload["count"] == 1
        assert payload["repos"][0]["full_name"] == "joe/microwave-os"

    async def test_repo_summary_validates_full_name(self):
        from src.tools.github import _handle_repo_summary

        result = await _handle_repo_summary({"repo": "not-a-full-name"}, "tok")
        assert result.get("is_error") is True
        assert "owner/name" in result["content"][0]["text"]

    async def test_repo_summary_surfaces_github_error(self):
        from src.tools.github import _handle_repo_summary

        with patch("src.tools.github.GitHubClient") as MockClient:
            instance = MockClient.return_value
            instance.get_repo_summary = AsyncMock(
                side_effect=GitHubError("rate limited", status=403)
            )

            result = await _handle_repo_summary(
                {"repo": "joe/microwave-os"}, "tok"
            )

        assert result.get("is_error") is True
        assert "GitHub API error" in result["content"][0]["text"]

    async def test_recent_activity_includes_username(self):
        from src.tools.github import _handle_recent_activity

        fake_events = [
            ActivityEvent(
                type="PushEvent",
                repo="joe/a",
                created_at="2026-05-10T12:00:00Z",
                summary="Pushed 1 commit(s) to main",
            ),
        ]
        with patch("src.tools.github.GitHubClient") as MockClient:
            instance = MockClient.return_value
            instance.get_authenticated_user = AsyncMock(
                return_value={"login": "joe"}
            )
            instance.list_recent_activity = AsyncMock(return_value=fake_events)

            result = await _handle_recent_activity({"limit": 30}, "tok")

        payload = json.loads(result["content"][0]["text"])
        assert payload["username"] == "joe"
        assert payload["count"] == 1
        assert payload["events"][0]["type"] == "PushEvent"

    async def test_recent_activity_handles_missing_login(self):
        from src.tools.github import _handle_recent_activity

        with patch("src.tools.github.GitHubClient") as MockClient:
            instance = MockClient.return_value
            instance.get_authenticated_user = AsyncMock(return_value={})

            result = await _handle_recent_activity({}, "tok")

        assert result.get("is_error") is True
        assert "username" in result["content"][0]["text"]


# --- Registry ---


class TestRegistry:
    def test_no_token_means_no_github_tools(self, monkeypatch):
        # Disable always-on web tools so we can assert the registry is
        # genuinely empty for instacart/github when neither key is set.
        monkeypatch.setenv("WEB_TOOLS_DISABLED", "1")
        from src.tools import build_tools

        config = SimpleNamespace(instacart_api_key="", github_token="")
        bundle = build_tools(config)
        assert bundle.is_empty

    def test_token_present_registers_three_tools(self):
        from src.tools import build_tools

        config = SimpleNamespace(
            instacart_api_key="",
            github_token="ghp_fake",
        )
        bundle = build_tools(config)
        assert not bundle.is_empty
        names = " ".join(bundle.allowed_tools)
        assert "github_list_repos" in names
        assert "github_repo_summary" in names
        assert "github_recent_activity" in names
        # Catalog text gets surfaced to the LLM — make sure it documents
        # the tool names so the model can find them in context.
        assert "github_list_repos" in bundle.catalog_text
        assert "github_repo_summary" in bundle.catalog_text
        assert "github_recent_activity" in bundle.catalog_text

    def test_both_keys_set_registers_both_tool_families(self):
        from src.tools import build_tools

        config = SimpleNamespace(
            instacart_api_key="fake",
            instacart_partner_linkback_url="",
            github_token="ghp_fake",
        )
        bundle = build_tools(config)
        names = " ".join(bundle.allowed_tools)
        assert "instacart_create_cart" in names
        assert "github_list_repos" in names
