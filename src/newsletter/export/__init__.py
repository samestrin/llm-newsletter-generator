"""Export an approved issue to email platforms as a draft, never as a send."""

Beehiiv = Buttondown = Resend = None


def connectors_from_env(env=None):
    raise NotImplementedError


def export_draft(*args, **kwargs):
    raise NotImplementedError


def export_stored_draft(*args, **kwargs):
    raise NotImplementedError
