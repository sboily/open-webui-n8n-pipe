"""User-facing messages (status events and errors) in every supported language.

Open-WebUI functions are distributed as a single file, so the locale catalogue
lives here instead of in external JSON files. ``t`` is the only way to read it.
"""

from .constants import Language

MESSAGES: dict[Language, dict[str, str]] = {
    Language.EN: {
        "status.calling": "Calling n8n workflow...",
        "status.attempt": "Attempt {attempt}/{total}",
        "status.waiting": "Waiting for n8n... ({elapsed}s)",
        "status.retry": "Retrying in {delay}s ({attempt}/{total})",
        "status.streaming": "Streaming response from n8n...",
        "status.complete": "Complete",
        "status.error": "Error while calling n8n: {error}",
        "error.no_messages": "No messages found in the request body",
        "error.last_not_user": "The last message must be from a user",
        "error.empty_question": "Please provide a non-empty question",
        "error.timeout": "n8n did not answer within {seconds}s (increase the 'timeout' valve)",
        "error.transport": "Could not reach n8n: {error}",
        "error.retries_exhausted": "n8n is unavailable after {total} attempts: {error}",
        "error.http_status": "n8n returned HTTP {status}: {body}",
        "error.invalid_json": "n8n response is not valid JSON",
        "error.response_field": "Response field '{field}' not found in the n8n response",
        "error.stream": "n8n stream reported an error: {error}",
        "error.attachment_too_large": "Attachment {name} exceeds {limit} MB",
        "error.attachment_not_image": "Attachment {name} is not an image ({mime})",
        "error.invalid_data_url": "Attachment {name} has an invalid data URL",
        "error.unsupported_url": "Attachment {name} has an unsupported URL scheme",
        "error.remote_images_disabled": (
            "Remote image URLs are disabled (valve 'allow_remote_images')"
        ),
        "error.private_host": "Refusing to fetch {name} from a private network address",
        "error.remote_download": "Could not download {name}: {error}",
        "error.file_unreadable": "File {name} could not be read from Open-WebUI storage",
    },
    Language.FR: {
        "status.calling": "Appel du workflow n8n...",
        "status.attempt": "Tentative {attempt}/{total}",
        "status.waiting": "En attente de n8n... ({elapsed}s)",
        "status.retry": "Nouvel essai dans {delay}s ({attempt}/{total})",
        "status.streaming": "Réception de la réponse n8n en continu...",
        "status.complete": "Terminé",
        "status.error": "Erreur lors de l'appel à n8n : {error}",
        "error.no_messages": "Aucun message trouvé dans la requête",
        "error.last_not_user": "Le dernier message doit provenir d'un utilisateur",
        "error.empty_question": "Veuillez saisir une question non vide",
        "error.timeout": "n8n n'a pas répondu en {seconds}s (augmentez la valve 'timeout')",
        "error.transport": "Impossible de joindre n8n : {error}",
        "error.retries_exhausted": "n8n est indisponible après {total} tentatives : {error}",
        "error.http_status": "n8n a répondu HTTP {status} : {body}",
        "error.invalid_json": "La réponse de n8n n'est pas un JSON valide",
        "error.response_field": "Champ '{field}' absent de la réponse n8n",
        "error.stream": "Le flux n8n a signalé une erreur : {error}",
        "error.attachment_too_large": "La pièce jointe {name} dépasse {limit} Mo",
        "error.attachment_not_image": "La pièce jointe {name} n'est pas une image ({mime})",
        "error.invalid_data_url": "La pièce jointe {name} a une data URL invalide",
        "error.unsupported_url": "La pièce jointe {name} a un schéma d'URL non supporté",
        "error.remote_images_disabled": (
            "Les URL d'images distantes sont désactivées (valve 'allow_remote_images')"
        ),
        "error.private_host": "Refus de télécharger {name} depuis une adresse réseau privée",
        "error.remote_download": "Téléchargement de {name} impossible : {error}",
        "error.file_unreadable": (
            "Le fichier {name} n'a pas pu être lu depuis le stockage Open-WebUI"
        ),
    },
}


def t(language: Language, key: str, **params: object) -> str:
    """Return the message ``key`` in ``language`` with ``params`` interpolated."""
    return MESSAGES[language][key].format(**params)


def describe(error: BaseException) -> str:
    """Describe an exception for users; ``str()`` alone is empty for httpx timeouts."""
    detail = str(error).strip()
    name = type(error).__name__
    return f"{name}: {detail}" if detail else name
