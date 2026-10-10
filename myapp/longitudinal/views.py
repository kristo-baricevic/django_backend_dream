"""Authenticated, private API. No body-supplied owner and no provider calls."""
from uuid import UUID
import unicodedata
from django.core.exceptions import ObjectDoesNotExist, PermissionDenied, ValidationError as DjangoValidationError
from django.http import Http404
from rest_framework.authentication import SessionAuthentication
from rest_framework.exceptions import APIException, NotAuthenticated
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from . import core, services
from .ownership import ConfigurationRequired, security


def body(request, allowed, required=()):
    value = request.data
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise core.InvalidEvidence("invalid_request_fields")
    return dict(value)


def revision(value):
    if not isinstance(value, dict) or set(value) != {"source", "privacy", "review"} or any(type(v) is not int or v < 0 for v in value.values()):
        raise core.InvalidEvidence("invalid_revision")
    return value


def ids(value):
    if not isinstance(value, list) or not value or len(value) > 100 or any(not isinstance(v, str) for v in value):
        raise core.InvalidEvidence("invalid_source_selection")
    try:
        return [UUID(v) for v in value]
    except (ValueError, TypeError, AttributeError):
        raise core.InvalidEvidence("invalid_source_selection")


class PrivateView(APIView):
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def owner(self, request):
        owner = security().require_owner(request)
        expected = request.headers.get("X-Dream-Principal")
        # Reject-only account-intent fence: this never chooses an owner.
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not expected:
            raise core.InvalidEvidence("principal_precondition_required")
        if expected and expected != str(owner.domain_user_id):
            raise PermissionDenied()
        request.longitudinal_owner = owner
        return owner

    def handle_exception(self, exc):
        if isinstance(exc, (ObjectDoesNotExist, Http404)):
            return Response({"error": "not_found"}, status=404)
        if isinstance(exc, services.Conflict):
            return Response({"error": "revision_or_request_changed"}, status=409)
        if isinstance(exc, core.InvalidEvidence):
            return Response({"error": str(exc)}, status=400)
        if isinstance(exc, DjangoValidationError):
            return Response({"error": "invalid_request"}, status=400)
        if isinstance(exc, ConfigurationRequired):
            return Response({"error": "owner_integration_unavailable"}, status=503)
        if isinstance(exc, (PermissionDenied, NotAuthenticated)):
            return Response({"error": "authentication_required"}, status=403)
        if isinstance(exc, APIException):
            return Response({"error": "request_not_authorized"}, status=exc.status_code)
        return super().handle_exception(exc)

    def finalize_response(self, request, response, *args, **kwargs):
        response = super().finalize_response(request, response, *args, **kwargs)
        response["Cache-Control"] = "private, no-store"
        response["Vary"] = "Cookie"
        if hasattr(request, "longitudinal_owner"):
            response["X-Dream-Principal"] = str(request.longitudinal_owner.domain_user_id)
        return response


class Preferences(PrivateView):
    def get(self, request):
        return Response(services.preferences(self.owner(request)))

    def patch(self, request):
        data = body(request, {"enabled", "provider_processing", "provider_context", "use_confirmed_context", "expected_revision"}, {"expected_revision"})
        data["expected_revision"] = revision(data["expected_revision"])
        return Response(services.set_preferences(self.owner(request), **data))


class Profile(PrivateView):
    def get(self, request):
        return Response(services.profile(self.owner(request), window=request.query_params.get("window", "all")))


class Context(PrivateView):
    def get(self, request):
        return Response(services.confirmed_context(self.owner(request)))


class ResetPreview(PrivateView):
    def get(self, request):
        return Response(services.source_reset_preview(self.owner(request)))


class Sources(PrivateView):
    def get(self, request):
        try:
            offset = int(request.query_params.get("offset", "0"))
            if not 0 <= offset <= 100000:
                raise ValueError
        except ValueError:
            raise core.InvalidEvidence("invalid_offset")
        return Response(services.sources(self.owner(request), offset=offset))


class SourcePreferences(PrivateView):
    def patch(self, request, entry_id):
        data = body(request, {"included", "origin", "language", "independent_occurrence", "expected_revision"}, {"expected_revision"})
        data["expected_revision"] = revision(data["expected_revision"])
        return Response(services.set_entry_preferences(self.owner(request), entry_id, **data))


class EntryAudit(PrivateView):
    def get(self, request, entry_id):
        return Response(services.entry_audit(self.owner(request), entry_id))


class Preview(PrivateView):
    def post(self, request):
        data = body(request, {"entry_ids", "mode"}, {"entry_ids"})
        return Response(services.preview(self.owner(request), ids(data["entry_ids"]), mode=data.get("mode", "local")))


class Runs(PrivateView):
    def post(self, request):
        data = body(request, {"entry_ids", "idempotency_key", "mode", "expected_revision"}, {"entry_ids", "idempotency_key", "expected_revision"})
        owner = self.owner(request)
        run = services.queue_audit(owner, ids(data["entry_ids"]), data["idempotency_key"], mode=data.get("mode", "local"), expected_revision=revision(data["expected_revision"]))
        return Response(services.run_status(owner, run.pk), status=202)


class Run(PrivateView):
    def get(self, request, id):
        return Response(services.run_status(self.owner(request), id))

    def delete(self, request, id):
        data = body(request, {"expected_revision"}, {"expected_revision"})
        return Response(services.run_status(self.owner(request), id, cancel=True, expected_revision=revision(data["expected_revision"])))


class Review(PrivateView):
    def patch(self, request, id):
        data = body(request, {"status", "label", "expected_revision"}, {"status", "expected_revision"})
        data["expected_revision"] = revision(data["expected_revision"])
        return Response(services.review_theme(self.owner(request), id, **data))


class Reflections(PrivateView):
    def get(self, request):
        try:
            offset = int(request.query_params.get("offset", "0"))
            if not 0 <= offset <= 100000:
                raise ValueError
        except ValueError:
            raise core.InvalidEvidence("invalid_offset")
        return Response(services.list_reflections(self.owner(request), offset=offset))

    def post(self, request):
        data = body(request, {"text", "entry_id", "use_for_profile", "use_for_chat"}, {"text"})
        if data.get("entry_id") is not None:
            try:
                data["entry_id"] = UUID(data["entry_id"]) if isinstance(data["entry_id"], str) else None
                if data["entry_id"] is None:
                    raise ValueError
            except (ValueError, TypeError, AttributeError):
                raise core.InvalidEvidence("invalid_entry_id")
        result = services.create_reflection(self.owner(request), **data)
        return Response(services.reflection_dto(result), status=201)


class Reflection(PrivateView):
    def patch(self, request, id):
        data = body(request, {"text", "expected_revision", "use_for_profile", "use_for_chat"}, {"text", "expected_revision"})
        if type(data["expected_revision"]) is not int or data["expected_revision"] < 1:
            raise core.InvalidEvidence("invalid_revision")
        return Response(services.reflection_dto(services.edit_reflection(self.owner(request), id, **data)))

    def delete(self, request, id):
        data = body(request, {"expected_revision"}, {"expected_revision"})
        if type(data["expected_revision"]) is not int or data["expected_revision"] < 1:
            raise core.InvalidEvidence("invalid_revision")
        services.delete_reflection(self.owner(request), id, **data)
        return Response(status=204)


class Observations(PrivateView):
    def post(self, request, entry_id):
        data = body(request, {"observation", "expected_revision"}, {"observation", "expected_revision"})
        candidate = data["observation"]
        if isinstance(candidate, dict) and candidate.get("kind") == "entity" and candidate.get("key") == "entity:auto" and isinstance(candidate.get("quote"), str):
            candidate["key"] = "entity:" + " ".join(unicodedata.normalize("NFKC", candidate["quote"]).casefold().split())
        result = services.add_observation(self.owner(request), entry_id, data["observation"], expected_revision=revision(data["expected_revision"]))
        return Response({"id": str(result.pk)}, status=201)


class Observation(PrivateView):
    def delete(self, request, entry_id, id):
        data = body(request, {"expected_revision"}, {"expected_revision"})
        return Response(services.delete_observation(self.owner(request), entry_id, id, expected_revision=revision(data["expected_revision"])))


class Merges(PrivateView):
    def post(self, request):
        data = body(request, {"theme_ids", "label", "expected_revision"}, {"theme_ids", "label", "expected_revision"})
        selected = [str(id) for id in ids(data["theme_ids"])]
        return Response(services.merge_themes(self.owner(request), selected, data["label"], expected_revision=revision(data["expected_revision"])), status=201)


class Merge(PrivateView):
    def delete(self, request, id):
        data = body(request, {"expected_revision"}, {"expected_revision"})
        return Response(services.delete_merge(self.owner(request), id, expected_revision=revision(data["expected_revision"])))


class UndoPublicRejection(PrivateView):
    def post(self, request):
        data = body(request, {"key", "expected_revision"}, {"key", "expected_revision"})
        return Response(services.undo_public_rejection(self.owner(request), data["key"], expected_revision=revision(data["expected_revision"])))


class Decisions(PrivateView):
    def get(self, request):
        try:
            offset = int(request.query_params.get("offset", "0"))
            if not 0 <= offset <= 100000:
                raise ValueError
        except ValueError:
            raise core.InvalidEvidence("invalid_offset")
        return Response(services.rejected_decisions(self.owner(request), offset=offset))
