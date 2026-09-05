"""Teacher backends. Resolved lazily through
:func:`data_pipeline.teacher.adapter.get_backend_class` so that importing the
package never pulls in torch/diffusers."""
