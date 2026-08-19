import os
import re

svg_dir = r"C:\DriftSenseAi\Drift-Sense-AI\images\readme"

# Map old sizes to new sizes
size_map = {
    '12px': '16px',
    '13px': '16px',
    '14px': '18px',
    '16px': '20px',
    '18px': '22px',
    '20px': '24px',
    '24px': '28px',
    '32px': '36px',
    '36px': '42px',
    '72px': '84px',
}

for filename in os.listdir(svg_dir):
    if filename.endswith(".svg"):
        filepath = os.path.join(svg_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Boost font sizes in style tags
        def replace_font_size(match):
            old_size = match.group(1)
            return f"font-size: {size_map.get(old_size, old_size)};"

        new_content = re.sub(r'font-size:\s*(\d+px);', replace_font_size, content)
        
        # Also fix inline style="font-size: 14px"
        def replace_inline_size(match):
            old_size = match.group(1)
            return f'font-size="{size_map.get(old_size, old_size)}"'
            
        new_content = re.sub(r'font-size="(\d+px)"', replace_inline_size, new_content)

        # Fix infographic-workflow vertical alignment and box size if present
        if filename == 'infographic-workflow.svg':
            new_content = new_content.replace('transform="translate(0, 110)"', 'transform="translate(0, 170)"')

        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated fonts in {filename}")
