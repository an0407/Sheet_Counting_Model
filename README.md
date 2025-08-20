# Sheet Counting Application

A Streamlit web application that uses OpenAI's GPT-4o-mini API to detect and count colored sheets in uploaded images.

## Features

- **Image Upload**: Support for PNG, JPG, and JPEG formats
- **AI-Powered Analysis**: Uses GPT-4o-mini for accurate sheet detection and counting
- **Color Recognition**: Identifies and counts sheets by color
- **Custom Prompts**: Optional custom prompt input for specialized analysis
- **JSON Response**: Structured output with accuracy estimation
- **User-Friendly Interface**: Clean Streamlit UI with results visualization

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Setup Steps

1. **Clone or download the project files**
   ```bash
   # If you have git, clone the repository
   # Or manually copy the project files to your directory
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy the environment template
   copy .env.template .env
   
   # Edit .env file and add your OpenAI API key
   # Replace 'your_openai_api_key_here' with your actual API key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL shown in your terminal

## Usage

1. **Upload an Image**: Click "Choose an image file" and select an image containing colored sheets
2. **Optional Custom Prompt**: Add specific instructions for the AI analysis
3. **Analyze**: Click "Analyze Image" to process the image
4. **View Results**: See both summary and detailed JSON response

## JSON Response Format

The application returns responses in the following standardized formats:

### Multiple Colored Sheets
```json
{
  "sheets_present": true,
  "colours": {
    "black": 45,
    "pink": 30,
    "purple": 30
  },
  "total": 105,
  "accuracy": "80%"
}
```

### Single Colored Sheets
```json
{
  "sheets_present": true,
  "colours": {
    "yellow": 40
  },
  "total": 40,
  "accuracy": "80%"
}
```

### No Sheets Present
```json
{
  "sheets_present": false,
  "colours": {},
  "total": 0,
  "accuracy": "80%"
}
```

## API Key Security

- Never commit your `.env` file to version control
- Keep your OpenAI API key secure and don't share it
- Consider using environment variables in production deployments

## Troubleshooting

### Common Issues

1. **"OpenAI API key not found" error**
   - Make sure you've created a `.env` file from the template
   - Verify your API key is correctly added to the `.env` file
   - Ensure there are no extra spaces or quotes around the API key

2. **"API request failed" error**
   - Check your internet connection
   - Verify your OpenAI API key is valid and has sufficient credits
   - Make sure you have access to GPT-4o-mini API

3. **Image upload issues**
   - Ensure your image is in PNG, JPG, or JPEG format
   - Try reducing image size if it's very large
   - Check that the image file isn't corrupted

4. **Dependency issues**
   - Make sure you're using Python 3.8 or higher
   - Try upgrading pip: `pip install --upgrade pip`
   - Install dependencies one by one if bulk installation fails

### Getting Help

If you encounter issues not covered here:
1. Check the error message displayed in the app
2. Look at the terminal/command prompt where you ran `streamlit run app.py`
3. Ensure all dependencies are properly installed
4. Verify your OpenAI API key and account status

## File Structure

```
counting_sheets_poc/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.template         # Environment variables template
├── README.md            # This file
└── .env                 # Your actual environment variables (create this)
```

## Dependencies

- **streamlit**: Web app framework
- **openai**: OpenAI API client
- **python-dotenv**: Environment variable management
- **Pillow**: Image processing
- **requests**: HTTP requests handling

## License

This project is provided as-is for educational and development purposes.