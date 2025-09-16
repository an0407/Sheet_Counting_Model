import streamlit as st
import base64
import json
import os
from io import BytesIO
from PIL import Image
import requests
from dotenv import load_dotenv

load_dotenv()

def encode_image(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    
    # Convert RGBA to RGB if necessary (for PNG with transparency)
    if image.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        # Paste the image on the background using the alpha channel as mask
        background.paste(image, mask=image.split()[-1])  # -1 is the alpha channel
        image = background
    elif image.mode not in ('RGB', 'L'):
        # Convert other modes to RGB
        image = image.convert('RGB')
    
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_sheets_with_openai(image):
    """Analyze image using OpenAI GPT-4 Vision API to count colored sheets"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OpenAI API key not found. Please check your .env file."}
    
    base64_image = encode_image(image)
    
    system_prompt =  """
You are a specialist in visual sheet analysis. Count sheets in each colored stack using ONLY edge-detection evidence (no height-based thickness estimation, no heuristics that depend on physical scale). Use conservative, repeatable rules to avoid overcounting.

STRICT PRIORITY ORDER (do these in order):
1) Segment stack regions by contiguous color/texture.
2) Within each stack region, use edge maps only to identify sheet boundaries.
3) Apply robust de-noising, clustering and sampling rules (see below).
4) Produce final integer counts per color and a single overall accuracy %.

EDGE COUNTING RULES (concrete, mandatory):
- Count only **sheet-boundary edges** that satisfy ALL these:
  • The edge is oriented roughly parallel to the layer direction (typically near-horizontal along the visible side face, or parallel to the top edge if the side is rotated). Accept ±30° of the dominant layer orientation.
  • The edge is continuous across at least **10–30% of the stack width** (prefer larger continuity). Short isolated speckles are ignored.
  • The edge segment length must be >= max(20 pixels, 10% of stack height) if image resolution is known; otherwise require clear continuity across a significant portion of the visible stack.
- Merge / de-duplicate: collapse edges closer than **2–4 pixels** or within **<2% of stack width** into a single edge to avoid double-counting caused by thin double-lines or noise.
- Reject texture or surface patterns: do not count thin surface patterns, printed lines, wood grain, or table textures. A valid sheet-boundary should be present roughly across the entire side face, not just a tiny local contrast.
- Require alignment consistency: adjacent counted edges should show reasonably consistent spacing. Compute the median spacing between adjacent edges; discard isolated edges whose spacing differs from the median by >60% (treat them as noise).

SAMPLE-BASED ROBUST COUNTING:
- Slice sampling: sample multiple horizontal slices across the visible face of each stack (≥5 slices evenly spaced). For each slice, count valid edge crossings (using above rules). The per-stack count = median of slice counts.
- If a slice shows many fewer edges due to occlusion, ignore that slice for the median (use slices with at least 60% of maximum slice width coverage).
- Final per-stack count = the median-of-slices, then **round down** to the nearest integer (never round up).

NO MULTIPLES-OF-10 POLICY:
- If the median-of-slices yields a small integer (≤20), return that exact integer. Do NOT round to 10s.
- For larger stacks, return the exact integer derived from the median-of-slices (round down), not a rounded multiple of 10.

COLOR SEGMENTATION & ASSIGNMENT:
- Determine stack color by sampling the interior region of the stack (avoid sampling edges). Use simple human color names (red, blue, green, pink, black, white, purple, yellow, orange, brown, grey).
- If color is ambiguous (mixed or faded), choose the closest dominant color name and mark accuracy lower.

CONSISTENCY & SANITY CHECKS (mandatory):
- Per-stack sanity:
  • If median spacing between adjacent edges implies an unrealistic sheet thickness (extremely small or widely varying), lower confidence and, if >50% of edges look noisy, set count conservative (e.g., reduce by 10–20%).
  • If two adjacent stacks of same color are separated by <5% of stack width and edge patterns merge, treat them as a single stack.
- Global sanity:
  • Ensure sum(colour counts) == total.
  • If no reliable edges detected anywhere, set "sheets_present": false and accuracy low.

OUTPUT FORMAT (STRICT — respond with EXACT JSON schema only):
{
  "sheets_present": true/false,
  "colours": {
    "color_name": integer_count,
    ...
  },
  "total": integer,
  "accuracy": "NN%"    // percentage out of 100, rounded to nearest integer percent, reflects overall confidence
}

ACCURACY GUIDELINES (how to compute % roughly):
- High (85–100%): clear, continuous layer lines across most slices, low noise, consistent spacing.
- Medium (65–84%): partial occlusions, some noisy slices but median-of-slices stable.
- Low (40–64%): many noisy edges; inconsistent spacing; partial visibility.
- Very Low (<40%): edges unclear or conflicting — set "sheets_present": false if counting is unreliable.

EXAMPLES (do not copy these answers into output; they are reference only):
- Clear small stacks: image shows exactly 4 blue visible separations and 3 green separations -> {"sheets_present": true, "colours": {"blue": 4, "green": 3}, "total": 7, "accuracy": "95%"}
- Mixed but clear: green stack median-of-slices=47, blue stack median-of-slices=31 -> {"sheets_present": true, "colours": {"green": 47, "blue": 31}, "total": 78, "accuracy": "82%"}
- No reliable edges: -> {"sheets_present": false, "colours": {}, "total": 0, "accuracy": "20%"}

KEY EMPHASIS (do not violate):
- Use EDGE DETECTION ONLY.
- Use slice-based median counts and de-duplication to avoid double-counting.
- Always prefer lower/conservative integer when in doubt.
- Never output rounded tens for visible small counts.
- Respond ONLY with the JSON specified above. Even if you are not able to analyze some type of image for sheet counts, still reply with the JSON format with {"sheets_present": false, "colours": {}, "total": 0, "accuracy": "90%"}
"""
    
    user_prompt = "Count the colored sheets in this image and provide the results in the specified JSON format."
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", 
                               headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        try:
            # Clean the content to handle markdown code blocks
            cleaned_content = content
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.startswith('```'):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
            
            json_result = json.loads(cleaned_content)
            
            required_keys = ['sheets_present', 'colours', 'total', 'accuracy']
            if not all(key in json_result for key in required_keys):
                return {"error": "Invalid response format from OpenAI API"}
            
            if not isinstance(json_result['sheets_present'], bool):
                return {"error": "Invalid sheets_present value in response"}
            
            if not isinstance(json_result['colours'], dict):
                return {"error": "Invalid colours format in response"}
            
            if not isinstance(json_result['total'], int):
                return {"error": "Invalid total value in response"}
            
            colours_sum = sum(json_result['colours'].values()) if json_result['colours'] else 0
            if colours_sum != json_result['total']:
                return {"error": f"Total mismatch: colours sum ({colours_sum}) != total ({json_result['total']})"}
            
            return json_result
            
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON response from OpenAI: {content}"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def display_results(result):
    """Display results in user-friendly format"""
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return
    
    st.success("Analysis Complete!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Summary")
        if result['sheets_present']:
            st.write(f"**Sheets Detected:** Yes")
            st.write(f"**Total Count:** {result['total']}")
            st.write(f"**Accuracy:** {result['accuracy']}")
        else:
            st.write("**Sheets Detected:** No")
    
    with col2:
        st.subheader("Color Breakdown")
        if result['colours']:
            for color, count in result['colours'].items():
                st.write(f"**{color.capitalize()}:** {count}")
        else:
            st.write("No sheets detected")
    

def main():
    st.set_page_config(
        page_title="Sheet Counter API",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🔢 SS-Suite: Sheet Counting Application")
    st.markdown("📷 **Upload an image to detect and count colored sheets using AI vision analysis.**")
    
    st.sidebar.header("⚙️ Configuration")
    
    if not os.getenv("OPENAI_API_KEY"):
        st.sidebar.error("⚠️ OpenAI API key not found!")
        st.sidebar.info("Please add your OpenAI API key to the .env file")
        st.stop()
    else:
        st.sidebar.success("✅ API Connected")
    
    uploaded_file = st.file_uploader(
        "📁 Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image containing stacks of colored sheets to analyze"
    )
    
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Uploaded Image")
                st.image(image, caption="Image to analyze", use_container_width=True)

            with col2:
                st.subheader("ℹ️ Image Information")
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {image.size}")
                st.write(f"**Mode:** {image.mode}")
                st.write(f"**Format:** {image.format}")
            
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("Analyzing image with AI..."):
                    result = analyze_sheets_with_openai(image)
                    display_results(result)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    else:
        st.info("👆 **Please upload an image to get started**")
    
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style="background-color: #e9ecef; padding: 20px; border-radius: 8px; border-left: 4px solid #1f77b4; margin: 10px 0; color: #333;">
                <h4 style="margin-top: 0; color: #1f77b4;">📋 How to use:</h4>
                <ol style="margin-bottom: 0; color: #333;">
                    <li><strong>Upload an image</strong> containing stacks of colored sheets</li>
                    <li><strong>Click 'Analyze Image'</strong> to get AI-powered sheet counting</li>
                    <li><strong>View results</strong> with color breakdown and accuracy estimate</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="background-color: #e9ecef; padding: 20px; border-radius: 8px; border-left: 4px solid #1f77b4; margin: 10px 0; color: #333;">
                <h4 style="margin-top: 0; color: #1f77b4;">📄 Supported formats:</h4>
                <ul style="margin-bottom: 0; color: #333;">
                    <li>PNG, JPG, JPEG images</li>
                    <li>Images containing stacks of sheets (single or multiple colors)</li>
                    <li>Estimates sheet count in each colored stack</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()