#!/usr/bin/env python3
"""Generate CGD pipeline schematic using Nano Banana Pro via OpenRouter."""

import os
import sys
import requests
import base64
import json
import time

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print("ERROR: OPENROUTER_API_KEY not set")
    sys.exit(1)

OUTPUT_DIR = "/Users/aayambansal/Desktop/VStudio/#research-repos/iclr-2/figures"

prompt = """Create a scientific pipeline diagram for a method called "Consistency-Guided Decoding (CGD)" used in an AI/NLP research paper at ICLR.

The diagram should show this exact pipeline flowing from LEFT to RIGHT:

SECTION 1 - INPUT (left side):
- A rounded rectangle box labeled "Question Set" with sub-text "Q = {q1, q2, ..., qn}" 
- Above it, a separate box labeled "Shared Premise P" with an arrow feeding down into the Question Set box
- An annotation "for each qi" with a dashed bracket encompassing the main processing loop

SECTION 2 - MAIN PROCESSING LOOP (center):
- Step box 1 (BLUE, color #4A90D9): Labeled "1. Generate" with sub-label "LLM M" and small text "produces initial answer ai"
- Step box 2 (ORANGE, color #E67E22): Labeled "2. Check" with sub-label "NLI Checker C" and small text "compare against history"
- A cylinder/database shape (light gray) labeled "Answer History" with text "{(q1,a1), (q2,a2), ...}" - connected bidirectionally to the Check box (reads from history, writes accepted answers)
- A decision diamond (YELLOW, color #F1C40F): Labeled "Contradiction?" with two outgoing paths:
  - "No" path: arrow going RIGHT toward the output (answer is consistent, stored in history)
  - "Yes" path: RED arrow (color #E74C3C) going DOWN to a repair box
- Step box 3 (GREEN, color #27AE60): Labeled "3. Repair" with sub-label "Revision Prompt → M" and text "produces revised answer ai'" - this box has an arrow looping back to the Check step

SECTION 3 - OUTPUT (right side):
- A rounded rectangle (GREEN border, color #27AE60) labeled "Consistent Answers" with text "{a1, a2, ..., an}"

ARROWS AND FLOW:
- Main flow arrows go left to right: Input → Generate → Check → Diamond → Output
- History cylinder connects to Check box with bidirectional arrows
- "Yes" contradiction arrow goes from diamond down to Repair box (red arrow)
- Repair box loops back to Check (green arrow)
- "No" arrow from diamond goes to Output and also feeds back to History

STYLE REQUIREMENTS:
- Publication-quality for ICLR conference paper
- Clean white background
- Professional sans-serif fonts (like Helvetica or Arial)
- Minimum 10pt equivalent text
- Colorblind-friendly - use the specific hex colors mentioned above
- No unnecessary decorations, shadows, or 3D effects
- Clean thin black arrows with arrowheads
- All boxes have rounded corners except the diamond
- The overall layout should fit a single-column figure in a paper
- Aspect ratio approximately 3:1 (wide landscape)
"""

# Try Nano Banana Pro first
models_to_try = [
    "nanobanana/nano-banana-pro",
    "nanobanana/nano-banana-pro:free",
]

def try_generate(model_name, prompt_text):
    """Try generating with a specific model."""
    print(f"\n--- Trying model: {model_name} ---")
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://cli.syntheticsciences.ai",
                "X-Title": "SynSci CLI",
            },
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],
            },
            timeout=180,
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                
                # Check if content is a list (multimodal response)
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "image_url":
                                img_data = item["image_url"]["url"]
                                if img_data.startswith("data:image"):
                                    b64_data = img_data.split(",")[1]
                                    img_bytes = base64.b64decode(b64_data)
                                    out_path = os.path.join(OUTPUT_DIR, "cgd_pipeline.png")
                                    with open(out_path, "wb") as f:
                                        f.write(img_bytes)
                                    print(f"SUCCESS: Saved image to {out_path}")
                                    return True
                            elif item.get("type") == "text":
                                print(f"Text content: {item['text'][:300]}")
                
                # Check if content is a string with base64
                elif isinstance(content, str):
                    if "base64" in content.lower():
                        # Try to extract base64 image data
                        import re
                        b64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                        matches = re.findall(b64_pattern, content)
                        if matches:
                            img_bytes = base64.b64decode(matches[0])
                            out_path = os.path.join(OUTPUT_DIR, "cgd_pipeline.png")
                            with open(out_path, "wb") as f:
                                f.write(img_bytes)
                            print(f"SUCCESS: Extracted and saved image to {out_path}")
                            return True
                    
                    print(f"Text response (first 500 chars): {content[:500]}")
                    # Save the text response for debugging
                    with open(os.path.join(OUTPUT_DIR, f"response_{model_name.replace('/', '_')}.txt"), "w") as f:
                        f.write(content)
                    
                return False
        else:
            error_text = response.text[:500]
            print(f"Error response: {error_text}")
            return False
            
    except requests.exceptions.Timeout:
        print("Request timed out")
        return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

# Try each model
success = False
for model in models_to_try:
    if try_generate(model, prompt):
        success = True
        break
    time.sleep(2)

if not success:
    print("\n--- Nano Banana Pro models not available. Trying Google Gemini with image gen... ---")
    # Try Gemini with image generation
    gemini_models = [
        "google/gemini-2.0-flash-exp:free",
        "google/gemini-2.0-flash-001",
    ]
    for model in gemini_models:
        if try_generate(model, prompt):
            success = True
            break
        time.sleep(2)

if not success:
    print("\n=== AI image generation did not return images. Will use matplotlib fallback. ===")
    sys.exit(1)
else:
    print("\n=== AI image generation succeeded! ===")
