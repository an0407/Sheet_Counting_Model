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
    
    system_prompt = """You are an expert document analyst specializing in counting colored paper sheets in stacks. 

COUNTING METHODOLOGY:
1. IDENTIFY STACKS: Look for distinct groups/stacks of colored sheets
2. MEASURE THICKNESS: Estimate sheet count by analyzing:
   - Stack height/thickness relative to single sheet thickness
   - Edge patterns showing individual sheet layers
   - Shadow depth between sheet edges
   - Visible sheet separation lines
3. COLOR ANALYSIS: Distinguish between different colored stacks
4. CROSS-REFERENCE: Verify counts make logical sense

ADVANCED COUNTING TECHNIQUES:
- Each sheet is typically 0.1mm thick - use this to estimate from stack height
- Look for slight color variations within stacks that indicate individual sheets
- Count visible edges on stack sides where individual sheets show
- Use perspective and lighting to judge depth
- Consider that sheets may be slightly offset showing layered edges
- Paper stacks often show subtle horizontal lines between sheets
- Different paper types (copy paper, cardstock, etc.) have different thicknesses

RESPOND ONLY with this JSON format:
{"sheets_present": true/false, "colours": {"color": count}, "total": number, "accuracy": "XX%"}

Examples:
- Thick black stack (50 sheets), thin red stack (15 sheets): {"sheets_present": true, "colours": {"black": 50, "red": 15}, "total": 65, "accuracy": "85%"}
- Single yellow stack (25 sheets): {"sheets_present": true, "colours": {"yellow": 25}, "total": 25, "accuracy": "78%"}
- No sheets visible: {"sheets_present": false, "colours": {}, "total": 0, "accuracy": "95%"}

ACCURACY FACTORS:
- Clear stack edges with visible sheet layers: 85-95%
- Good lighting, distinct colors, measurable thickness: 75-90%
- Partial visibility, some shadows, moderate thickness: 60-80%
- Poor lighting, similar colors, difficult to measure: 45-65%
- Very unclear image, cannot distinguish stacks: 25-50%

CRITICAL RULES:
- Count sheets IN stacks, not individual loose sheets
- Use thickness-to-count ratio for estimation
- Consider standard paper thickness (0.1mm for copy paper)
- Total must equal sum of all color counts
- Use common color names only
- Be conservative with counts if uncertain"""
    
    user_prompt = "Count the colored sheets in this image and provide the results in the specified JSON format."
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
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
    
    st.title("🔢 SS-Suite:Sheet Counting Application")
    st.markdown("Upload an image to detect and count colored sheets using AI vision analysis.")
    
    st.sidebar.header("Configuration")
    
    if not os.getenv("OPENAI_API_KEY"):
        st.sidebar.error("⚠️ OpenAI API key not found!")
        st.sidebar.info("Please add your OpenAI API key to the .env file")
        st.stop()
    else:
        st.sidebar.success("API Running")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image containing stacks of colored sheets to analyze"
    )
    
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, caption="Image to analyze", use_container_width=True)
            
            with col2:
                st.subheader("Image Information")
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
        st.info("👆 Please upload an image to get started")
    
    st.markdown("---")
    st.markdown(
        """
        ### How to use:
        1. **Upload an image** containing stacks of colored sheets
        2. **Click 'Analyze Image'** to get AI-powered sheet counting
        3. **View results** with color breakdown and accuracy estimate
        
        ### Supported formats:
        - PNG, JPG, JPEG images
        - Images containing stacks of sheets (single or multiple colors)
        - Estimates sheet count in each colored stack
        """
    )

if __name__ == "__main__":
    main()