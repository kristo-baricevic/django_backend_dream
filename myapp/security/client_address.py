"""Only explicitly trusted peers may attest the edge-observed client address."""
from ipaddress import ip_address, ip_network
from django.conf import settings

def login_address(request):
    try: peer = ip_address(request.META.get('REMOTE_ADDR', ''))
    except ValueError: return None
    trusted = any(peer in ip_network(cidr) for cidr in getattr(settings, 'DREAM_LOGIN_PROXY_CIDRS', []))
    if not trusted: return str(peer)
    if getattr(settings, 'DREAM_ACCEPT_VERIFIED_CLIENT_IP', False):
        try: return str(ip_address(request.META.get('HTTP_X_DREAM_CLIENT_IP', '')))
        except ValueError: pass
    # A shared proxy without attested client IP uses distributed account limits.
    return None
