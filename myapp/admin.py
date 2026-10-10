from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.contrib.admin.forms import AdminAuthenticationForm
from django.core.exceptions import ValidationError
from .models import User, JournalEntry, Analysis, CumulativeAnalysis, CustomQuestion, Settings, AnalysisFeedback
from .security.owner import require_owner, _valid_profile


class LinkedAdminAuthenticationForm(AdminAuthenticationForm):
    def confirm_login_allowed(self, user):
        super().confirm_login_allowed(user)
        profile = User.objects.filter(auth_user_id=user.pk).first()
        if profile is None or not _valid_profile(profile, user):
            raise ValidationError('Invalid login.', code='invalid_login')


admin.site.login_form = LinkedAdminAuthenticationForm
for identity_model in (get_user_model(), Group):
    if admin.site.is_registered(identity_model):
        admin.site.unregister(identity_model)


class OwnedReadOnlyAdmin(admin.ModelAdmin):
    actions = None

    def get_queryset(self, request):
        owner = require_owner(request)
        query = super().get_queryset(request)
        return query.filter(pk=owner.domain_user_id) if self.model is User else query.filter(user_id=owner.domain_user_id)

    def get_readonly_fields(self, request, obj=None):
        return tuple(field.name for field in self.model._meta.fields)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def has_view_permission(self, request, obj=None):
        if not super().has_view_permission(request, obj):
            return False
        if obj is None:
            return True
        owner = require_owner(request)
        return (obj.pk if self.model is User else obj.user_id) == owner.domain_user_id


@admin.register(User)
class UserAdmin(OwnedReadOnlyAdmin):
    list_display = ('id', 'email', 'created_at', 'updated_at')
    exclude = ('auth_user', 'active_settings', 'is_local_fixture', 'import_ready')


for domain_model in (JournalEntry, Analysis, CumulativeAnalysis, CustomQuestion, Settings):
    admin.site.register(domain_model, OwnedReadOnlyAdmin)


@admin.register(AnalysisFeedback)
class FeedbackAdmin(OwnedReadOnlyAdmin):
    exclude = ('session_id',)
