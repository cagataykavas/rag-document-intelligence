"""Opt-in official OpenTelemetry SDK. Only bounded metadata, never source text.

Local SDK exporter is for reproducible evidence, not a hosted collector claim.
Exception messages/stack traces are deliberately excluded from spans.
"""
from contextlib import contextmanager
import json
from pathlib import Path


class Telemetry:
    def __init__(self, exporter):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace.sampling import ALWAYS_ON
        # Explicit opt-in evidence capture ignores ambient sampling and resource metadata.
        self.provider = TracerProvider(resource=Resource({'service.name': 'evidence-agent'}),
                                       sampler=ALWAYS_ON)
        self.provider.add_span_processor(SimpleSpanProcessor(exporter))
        self.tracer = self.provider.get_tracer('evidence-agent', '1')

    @contextmanager
    def span(self, name, **attributes):
        from opentelemetry.trace import Status, StatusCode
        with self.tracer.start_as_current_span(name, attributes=attributes,
                record_exception=False, set_status_on_exception=False) as span:
            try:
                yield span
            except BaseException as exc:
                span.set_attribute('error.type', type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR))
                raise

    @contextmanager
    def task(self, context=None):
        from opentelemetry.trace import Status, StatusCode
        with self.tracer.start_as_current_span('agent.task', context=context,
                record_exception=False, set_status_on_exception=False) as task:
            try:
                yield task
            except BaseException as exc:
                task.set_attribute('error.type', type(exc).__name__)
                task.set_status(Status(StatusCode.ERROR))
                raise

    def shutdown(self):
        self.provider.force_flush()
        self.provider.shutdown()


def file_telemetry(path):
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class LocalExporter(SpanExporter):
        def export(self, spans):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with Path(path).open('a') as stream:
                for span in spans:
                    # SDK ReadableSpan serialization, not synthetic trace construction.
                    stream.write(json.dumps(json.loads(span.to_json())) + '\n')
            return SpanExportResult.SUCCESS

    return Telemetry(LocalExporter())


@contextmanager
def span(telemetry, name, **attributes):
    if telemetry is None:
        yield None
    else:
        with telemetry.span(name, **attributes) as active:
            yield active
