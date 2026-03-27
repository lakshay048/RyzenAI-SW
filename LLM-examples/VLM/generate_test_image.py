#!/usr/bin/env python
"""
Generate a synthetic test image for VLM benchmarking.

Creates images with varied visual patterns including:
- Gradients and geometric shapes (stress ViT encoder)
- Text at various sizes (test OCR capabilities)
- Chart-like elements (relevant for ChartQA)
- High and low frequency regions (test multi-scale processing)
"""

import argparse
import math
import os
import random

from PIL import Image, ImageDraw, ImageFont


def draw_bar_chart(draw, x, y, width, height):
    """Draw a simple bar chart."""
    num_bars = 6
    bar_width = width // (num_bars * 2)
    colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF6D01", "#46BDC6"]

    # Draw axes
    draw.line([(x, y + height), (x + width, y + height)], fill="black", width=2)
    draw.line([(x, y), (x, y + height)], fill="black", width=2)

    # Draw bars
    for i in range(num_bars):
        bar_height = random.randint(height // 4, height - 10)
        bx = x + 10 + i * (bar_width * 2)
        draw.rectangle(
            [bx, y + height - bar_height, bx + bar_width, y + height],
            fill=colors[i % len(colors)],
            outline="black"
        )


def draw_line_chart(draw, x, y, width, height):
    """Draw a simple line chart."""
    # Draw axes
    draw.line([(x, y + height), (x + width, y + height)], fill="black", width=2)
    draw.line([(x, y), (x, y + height)], fill="black", width=2)

    # Draw line with points
    num_points = 8
    points = []
    for i in range(num_points):
        px = x + 10 + i * (width - 20) // (num_points - 1)
        py = y + random.randint(10, height - 10)
        points.append((px, py))

    # Draw line
    draw.line(points, fill="#4285F4", width=3)

    # Draw points
    for px, py in points:
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill="#EA4335", outline="white")


def draw_pie_chart(draw, cx, cy, radius):
    """Draw a simple pie chart."""
    colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF6D01"]
    angles = [72, 90, 54, 108, 36]  # Sum = 360

    start_angle = 0
    for i, angle in enumerate(angles):
        draw.pieslice(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            start=start_angle,
            end=start_angle + angle,
            fill=colors[i],
            outline="white",
            width=2
        )
        start_angle += angle


def draw_table(draw, x, y, width, height, font):
    """Draw a simple data table."""
    rows = 4
    cols = 3
    cell_w = width // cols
    cell_h = height // rows

    headers = ["Item", "Value", "Change"]
    data = [
        ["Alpha", "1,234", "+5.2%"],
        ["Beta", "987", "-2.1%"],
        ["Gamma", "2,456", "+8.7%"],
    ]

    # Draw grid
    for i in range(rows + 1):
        draw.line([(x, y + i * cell_h), (x + width, y + i * cell_h)], fill="black", width=1)
    for j in range(cols + 1):
        draw.line([(x + j * cell_w, y), (x + j * cell_w, y + height)], fill="black", width=1)

    # Draw header
    for j, header in enumerate(headers):
        draw.rectangle([x + j * cell_w, y, x + (j + 1) * cell_w, y + cell_h], fill="#E8E8E8")
        draw.text((x + j * cell_w + 5, y + 5), header, fill="black", font=font)

    # Draw data
    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            color = "green" if "+" in cell else ("red" if "-" in cell else "black")
            draw.text((x + j * cell_w + 5, y + (i + 1) * cell_h + 5), cell, fill=color, font=font)


def generate_test_image(size: int = 4032, output_path: str = "test_image.jpg", seed: int = 42):
    """Generate a synthetic test image with visual complexity."""
    random.seed(seed)

    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    # Try to get a font, fall back to default if not available
    try:
        font_size = max(size // 80, 12)
        font_large = ImageFont.truetype("arial.ttf", font_size * 2)
        font_medium = ImageFont.truetype("arial.ttf", font_size)
        font_small = ImageFont.truetype("arial.ttf", font_size // 2)
    except (OSError, IOError):
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large

    cell_size = size // 4  # Divide into 4x4 grid of regions

    # =========================================================================
    # Region (0,0): Gradient background with text
    # =========================================================================
    for x in range(cell_size):
        for y in range(cell_size):
            r = int(255 * x / cell_size)
            g = int(255 * y / cell_size)
            b = 128
            draw.point((x, y), fill=(r, g, b))

    # Add title text
    draw.text((20, 20), "VLM Benchmark Test Image", fill="white", font=font_large)
    draw.text((20, 20 + font_size * 3), f"Resolution: {size}x{size}", fill="white", font=font_medium)
    draw.text((20, 20 + font_size * 5), "This image tests visual understanding", fill="white", font=font_small)

    # =========================================================================
    # Region (1,0): Bar chart
    # =========================================================================
    draw.rectangle([cell_size, 0, cell_size * 2, cell_size], fill="#F5F5F5")
    draw.text((cell_size + 10, 10), "Sales by Quarter", fill="black", font=font_medium)
    draw_bar_chart(draw, cell_size + 20, 50, cell_size - 40, cell_size - 80)

    # =========================================================================
    # Region (2,0): Line chart
    # =========================================================================
    draw.rectangle([cell_size * 2, 0, cell_size * 3, cell_size], fill="#FAFAFA")
    draw.text((cell_size * 2 + 10, 10), "Trend Analysis", fill="black", font=font_medium)
    draw_line_chart(draw, cell_size * 2 + 20, 50, cell_size - 40, cell_size - 80)

    # =========================================================================
    # Region (3,0): Pie chart
    # =========================================================================
    draw.rectangle([cell_size * 3, 0, cell_size * 4, cell_size], fill="#F0F0F0")
    draw.text((cell_size * 3 + 10, 10), "Market Share", fill="black", font=font_medium)
    draw_pie_chart(draw, cell_size * 3 + cell_size // 2, cell_size // 2 + 20, cell_size // 3)

    # =========================================================================
    # Region (0,1): Geometric shapes
    # =========================================================================
    for x in range(cell_size):
        for y in range(cell_size, cell_size * 2):
            draw.point((x, y), fill=(200, 220, 240))

    # Circles, rectangles, triangles
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    for i in range(5):
        cx = 50 + i * (cell_size // 5)
        cy = cell_size + cell_size // 3
        r = cell_size // 12
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colors[i], outline="black", width=2)

    for i in range(4):
        rx = 30 + i * (cell_size // 4)
        ry = cell_size + cell_size * 2 // 3
        rw = cell_size // 10
        draw.rectangle([rx, ry, rx + rw, ry + rw], fill=colors[i], outline="black", width=2)

    # =========================================================================
    # Region (1,1): Data table
    # =========================================================================
    draw.rectangle([cell_size, cell_size, cell_size * 2, cell_size * 2], fill="white")
    draw_table(draw, cell_size + 20, cell_size + 40, cell_size - 40, cell_size - 80, font_small)

    # =========================================================================
    # Region (2,1): Text at multiple sizes
    # =========================================================================
    draw.rectangle([cell_size * 2, cell_size, cell_size * 3, cell_size * 2], fill="#FFFEF0")

    texts = [
        ("Large Text Sample", font_large, cell_size + 20),
        ("Medium text for reading tests", font_medium, cell_size + 80),
        ("Small text: The quick brown fox jumps over the lazy dog", font_small, cell_size + 130),
        ("Numbers: 0123456789", font_medium, cell_size + 170),
        ("Special: @#$%^&*()!?", font_medium, cell_size + 220),
    ]
    for text, font, y_pos in texts:
        draw.text((cell_size * 2 + 20, y_pos), text, fill="black", font=font)

    # =========================================================================
    # Region (3,1): High-frequency patterns (checkerboard, stripes)
    # =========================================================================
    checker_size = max(size // 200, 4)
    for x in range(cell_size * 3, cell_size * 4):
        for y in range(cell_size, cell_size * 2):
            if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                draw.point((x, y), fill=(50, 50, 50))
            else:
                draw.point((x, y), fill=(200, 200, 200))

    # =========================================================================
    # Region (0,2): Concentric circles
    # =========================================================================
    cx, cy = cell_size // 2, cell_size * 2 + cell_size // 2
    for r in range(cell_size // 2, 10, -20):
        color = ((r * 7) % 256, (r * 3) % 256, (r * 11) % 256)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)

    # =========================================================================
    # Region (1,2): Diagonal stripes
    # =========================================================================
    stripe_width = max(size // 100, 8)
    for i in range(-cell_size, cell_size * 2, stripe_width * 2):
        color = (255, (i * 3) % 256, (i * 7) % 256)
        draw.line(
            [(cell_size + i, cell_size * 2), (cell_size * 2 + i, cell_size * 3)],
            fill=color,
            width=stripe_width
        )

    # =========================================================================
    # Region (2,2): Noise/texture pattern
    # =========================================================================
    for x in range(cell_size * 2, cell_size * 3):
        for y in range(cell_size * 2, cell_size * 3):
            noise = random.randint(-30, 30)
            base = 128 + noise
            draw.point((x, y), fill=(base, base, base))

    # =========================================================================
    # Region (3,2): Color gradient wheel
    # =========================================================================
    cx, cy = cell_size * 3 + cell_size // 2, cell_size * 2 + cell_size // 2
    radius = cell_size // 2 - 20
    for angle in range(0, 360, 2):
        rad = math.radians(angle)
        for r in range(radius):
            x = int(cx + r * math.cos(rad))
            y = int(cy + r * math.sin(rad))
            # HSV to RGB approximation
            hue = angle
            sat = r / radius
            val = 1.0

            h = hue / 60
            i = int(h)
            f = h - i
            p = val * (1 - sat)
            q = val * (1 - sat * f)
            t = val * (1 - sat * (1 - f))

            if i == 0:
                rgb = (val, t, p)
            elif i == 1:
                rgb = (q, val, p)
            elif i == 2:
                rgb = (p, val, t)
            elif i == 3:
                rgb = (p, q, val)
            elif i == 4:
                rgb = (t, p, val)
            else:
                rgb = (val, p, q)

            color = tuple(int(c * 255) for c in rgb)
            if 0 <= x < size and 0 <= y < size:
                draw.point((x, y), fill=color)

    # =========================================================================
    # Region (0-3, 3): Bottom row - mixed complexity zones
    # =========================================================================

    # Low complexity: solid colors
    draw.rectangle([0, cell_size * 3, cell_size, cell_size * 4], fill="#E8F4FD")
    draw.text((20, cell_size * 3 + 20), "Low Complexity Zone", fill="black", font=font_medium)

    # Medium complexity: simple shapes
    draw.rectangle([cell_size, cell_size * 3, cell_size * 2, cell_size * 4], fill="#FDF4E8")
    for i in range(5):
        x = cell_size + 30 + i * (cell_size // 5)
        y = cell_size * 3 + cell_size // 2
        draw.rectangle([x, y, x + 40, y + 40], fill=colors[i % len(colors)], outline="black")
    draw.text((cell_size + 20, cell_size * 3 + 20), "Medium Complexity", fill="black", font=font_medium)

    # High complexity: dense patterns
    for x in range(cell_size * 2, cell_size * 3):
        for y in range(cell_size * 3, cell_size * 4):
            pattern = ((x * 7 + y * 13) % 256, (x * 11 + y * 3) % 256, (x * 5 + y * 17) % 256)
            draw.point((x, y), fill=pattern)
    draw.text((cell_size * 2 + 20, cell_size * 3 + 20), "High Complexity", fill="white", font=font_medium)

    # Edge cases: very fine details
    for x in range(cell_size * 3, cell_size * 4):
        for y in range(cell_size * 3, cell_size * 4):
            if (x + y) % 2 == 0:
                draw.point((x, y), fill=(0, 0, 0))
            else:
                draw.point((x, y), fill=(255, 255, 255))
    draw.text((cell_size * 3 + 20, cell_size * 3 + 20), "Fine Detail", fill="gray", font=font_medium)

    # Save image
    img.save(output_path, format="JPEG", quality=95)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Generated: {output_path} ({size}x{size}, {file_size_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic test image for VLM benchmarking")
    parser.add_argument("-s", "--size", type=int, default=4032, help="Image size in pixels (default: 4032)")
    parser.add_argument("-o", "--output", type=str, default="test_image.jpg", help="Output path (default: test_image.jpg)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    generate_test_image(size=args.size, output_path=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
