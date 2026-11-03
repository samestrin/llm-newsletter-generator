import hashlib

import pytest

from newsletter.archive import (
    AUTOMATED_DEMO,
    Approval,
    ApprovalError,
    demo_approval,
    human_approval,
    require_archive,
    require_export,
)

TEXT = "# AI Daily\n\nOne story.\n"


def test_the_demo_approval_is_a_record_marked_automated_demo_with_no_sha256():
    approval = demo_approval()
    assert approval.kind == AUTOMATED_DEMO == "automated-demo"
    assert approval.sha256 is None
    assert approval.automated
    assert approval.to_dict() == {"kind": "automated-demo", "sha256": None, "valid_for": "archive"}


def test_a_human_approval_is_the_sha256_of_the_text():
    approval = human_approval(TEXT)
    assert approval.kind == "human"
    assert approval.sha256 == hashlib.sha256(TEXT.encode()).hexdigest()
    assert not approval.automated


def test_an_export_with_an_automated_demo_approval_is_refused():
    with pytest.raises(ApprovalError, match="automated-demo.*archive pages only"):
        require_export(demo_approval(), TEXT)


def test_an_automated_demo_approval_carrying_a_matching_sha256_is_still_refused():
    smuggled = Approval(kind=AUTOMATED_DEMO, sha256=hashlib.sha256(TEXT.encode()).hexdigest())
    with pytest.raises(ApprovalError, match="archive pages only"):
        require_export(smuggled, TEXT)


def test_an_export_without_an_approval_is_refused():
    with pytest.raises(ApprovalError, match="not approved"):
        require_export(None, TEXT)


def test_an_export_after_an_edit_is_refused():
    approval = human_approval(TEXT)
    with pytest.raises(ApprovalError, match="changed since it was approved"):
        require_export(approval, TEXT + "An edit.\n")


def test_an_export_with_a_human_approval_of_the_current_text_is_allowed():
    require_export(human_approval(TEXT), TEXT)


@pytest.mark.parametrize("kind", ["", "Automated-Demo", "auto", "admin"])
def test_an_unknown_kind_of_approval_is_refused_everywhere(kind):
    approval = Approval(kind=kind, sha256=hashlib.sha256(TEXT.encode()).hexdigest())
    with pytest.raises(ApprovalError, match="unknown approval"):
        require_export(approval, TEXT)
    with pytest.raises(ApprovalError, match="unknown approval"):
        require_archive(approval)


def test_a_human_approval_with_no_sha256_is_refused_for_export():
    with pytest.raises(ApprovalError, match="no sha256"):
        require_export(Approval(kind="human", sha256=None), TEXT)


def test_the_archive_accepts_a_demo_or_a_human_approval_but_not_none():
    require_archive(demo_approval())
    require_archive(human_approval(TEXT))
    with pytest.raises(ApprovalError, match="not approved"):
        require_archive(None)
