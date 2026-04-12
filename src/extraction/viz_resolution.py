import re
import json
import html
import argparse

from pathlib import Path
from src.config import RAW_TEXT, OUT_DIR
from .utils import print_information, load_text, load_span_index


def main(text: Path, spans_path: Path, unknown: Path, out: Path):
    # Load original text
    text = load_text(text)

    spans = []

    # Load resolved spans
    if spans_path.exists():
        known_spans = load_span_index(spans_path)       
        spans = [(s.get("start_char"), s.get("end_char"), s.get("fullname", "RESOLVED"), True) 
                 for s in known_spans if s.get("start_char") is not None and s.get("end_char") is not None]

    # Load unknown clusters
    if unknown.exists():
        with open(unknown, "r", encoding="utf-8") as f:
            clusters = json.load(f)
            
            for cluster in clusters:
                for data in cluster:
                    start = data.get("start_char")
                    end = data.get("end_char")
                    if start is not None and end is not None:
                        spans.append((start, end, "UNKNOWN", False))

    # Sort spans by start descending length
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Resolve overlapping spans (greedy left-to-right)
    final_spans = []
    last_end = -1
    for start, end, label, is_resolved in spans:
        if start >= last_end:
            final_spans.append((start, end, label, is_resolved))
            last_end = end

    print_information(f"Total non-overlapping spans to highlight: {len(final_spans)}", prefix="     ")

    # Build HTML
    html_parts = []
    current_idx = 0
    
    resolved_count = 0
    for start, end, label, is_resolved in final_spans:
        # Text before span
        html_parts.append(html.escape(text[current_idx:start]))
        
        # Span text
        span_text = html.escape(text[start:end])
        escaped_label = html.escape(label)
        
        if is_resolved:
            color = "#d4edda" # Pale green
            text_color = "#155724"
            resolved_count += 1
        else:
            color = "#f8d7da" # Pale red
            text_color = "#721c24"
            
        html_parts.append(
            f'<mark style="background-color: {color}; color: {text_color}; border-radius: 4px; padding: 2px 4px; border: 1px solid rgba(0,0,0,0.1);">'
            f'{span_text} <sub style="color: #666; font-size: 0.75em; font-weight: 600;">[{escaped_label}]</sub>'
            f'</mark>'
        )
        current_idx = end

    # Remaining text
    html_parts.append(html.escape(text[current_idx:]))

    # Render newlines as <br> inside paragraphs, but keep paragraph structure.
    raw_html_body = "".join(html_parts)
    # Split by double newline to form paragraphs
    paragraphs = raw_html_body.split("\n\n")
    formatted_paragraphs = []
    for p in paragraphs:
        # Check for markdown headings (e.g., # Heading or ## Heading)
        m = re.match(r"^(#+)\s+(.*)$", p.strip(), re.DOTALL)
        if m:
            hashes, content = m.groups()
            h_level = len(hashes)
            if 1 <= h_level <= 6:
                # Still replace internal single newlines with <br> if any exist in the heading
                content_html = content.replace("\n", "<br>\n")
                formatted_paragraphs.append(f"<h{h_level}>{content_html}</h{h_level}>")
                continue

        # replace single newlines inside paragraph with <br>
        p_html = p.replace("\n", "<br>\n")
        formatted_paragraphs.append(f"<p>{p_html}</p>")

    final_html_body = "\n".join(formatted_paragraphs)
    final_html_body = final_html_body.replace(
        '<h1>',
        '</details><details open><summary><span class="chapter-heading">'
    )[len('</details>'):] + '</details>'

    # And replace closing </h1> with </summary>
    final_html_body = final_html_body.replace(
        '</h1>',
        '</span></summary>',
        # only h1s — h2..h6 stay as-is inside the open details
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Coreference Highlight</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.8;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            color: #333;
            background-color: #fcfcfc;
        }}
        h1 {{
            text-align: center;
            color: #444;
        }}
        p {{
            margin-bottom: 1.5em;
            text-align: justify;
        }}
        mark {{
            box-decoration-break: clone;
            -webkit-box-decoration-break: clone;
        }}
        details {{
            margin-bottom: 1em;
            border-top: 1px solid #ddd;
        }}
        details[open] {{
            padding-bottom: 0.5em;
        }}
        summary {{
            list-style: none;
            cursor: pointer;
            padding: 8px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        summary::-webkit-details-marker {{ display: none; }}
        summary::before {{
            content: "▶";
            font-size: 0.65em;
            color: #888;
            transition: transform 0.2s;
            flex-shrink: 0;
        }}
        details[open] > summary::before {{
            transform: rotate(90deg);
        }}
        .chapter-heading {{
            /* Copy your h1 styles here */
            font-size: 2em;
            font-weight: bold;
            color: #444;
        }}
    </style>
</head>
<body>
    <h1>Coreference Resolution Output</h1>
    <p style="text-align: center; font-size: 0.9em; color: #666;">
        <span style="background-color: #d4edda; padding: 2px 6px; border-radius: 4px;">Resolved Entity ({resolved_count})</span> | 
        <span style="background-color: #f8d7da; padding: 2px 6px; border-radius: 4px;">Unknown Cluster ({len(final_spans) - resolved_count})</span>
    </p>
    <hr style="margin: 30px 0; border: 0; border-top: 1px solid #eee;">
    {final_html_body}
</body>
</html>
"""

    out.write_text(html_content, encoding="utf-8")
    # print_information(f"Generated visualization: {out}", prefix="    ")
    # print_information(f"Open this file in your browser: file://{out.absolute()}", prefix="    ")

    # return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Coreference Results")
    parser.add_argument("--text", type=Path, default=RAW_TEXT)
    parser.add_argument("--spans", type=Path, default=OUT_DIR / "coreference" / "span_index.jsonl")
    parser.add_argument("--unknown", type=Path, default=OUT_DIR / "coreference" / "unknown_clusters.json")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "coreference" / "coref_visualization.html")
    args = parser.parse_args()

    main(args.text, args.spans, args.unknown, args.out)
