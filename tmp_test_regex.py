import re

def sanitize_tex(text: str) -> str:
    text = text.replace("#", "\\#").replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")
    return text

protected = []
def protect_links(m):
    idx = len(protected)
    label = sanitize_tex(m.group(1))
    url_link = m.group(2)
    protected.append(f"\\href{{{url_link}}}{{{label}}}")
    return f"MARKERPROTECT{idx}X"

regex = r'\[([^\]]+)\]\((https?://[^\s\)]+)\)'
line = "[Download PDF](http://www.cs.cmu.edu/~acarlson/papers/carlson-aaai10.pdf)"
result = re.sub(regex, protect_links, line)
print(f"Result: {result}")
print(f"Protected: {protected}")
