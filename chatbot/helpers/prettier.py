import os


def prettify_source(source):
    document = os.path.basename(source.get("document"))
    score = source.get("score")
    content_preview = source.get("content_preview")
    return f"• **{document}** with score ({round(score,2)}) \n\n **Preview:** \n {content_preview} \n"
