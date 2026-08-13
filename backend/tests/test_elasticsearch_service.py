import app.services.elasticsearch_service as module


class FakeElasticsearch:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.ping_calls = 0

    def ping(self):
        self.ping_calls += 1
        return True


def test_service_does_not_connect_during_initialization(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "Elasticsearch",
        FakeElasticsearch,
    )

    service = module.ElasticsearchService()

    assert service.client is not None
    assert service.client.ping_calls == 0

    assert service.is_available() is True
    assert service.client.ping_calls == 1