# MIrror-ai
Mirror: Your Advanced Skin Analysis Application
Here's a suggested `README` section for your code:

---

# Digital Mirror: Your Advanced Skin Analysis Application

**Created by**: Erfantaghavvi  
**Last Updated**: 2025-02-14 15:14:50 UTC

### Overview

The **Digital Mirror** application is designed to analyze skin types and detect common skin problems through image analysis. By uploading a photo of your face, users can receive a detailed skin analysis, including the detection of skin type, potential problems such as acne, wrinkles, dark spots, and more, along with product recommendations tailored to their skin needs.

### Features

- **Skin Type Analysis**: The app classifies the skin into categories such as Dry, Oily, Combination, Sensitive, or Normal based on various facial attributes.
- **Skin Problem Detection**: The app identifies common skin problems like acne, dark spots, wrinkles, pigmentation, rosacea, and eczema, providing a severity score for each.
- **Product Recommendations**: Based on the skin type and detected problems, the app recommends relevant cosmetic products from a CSV file.
- **Comprehensive Visualization**: The app shows both the original image and an overlay that highlights areas with skin problems.

### Installation

To run this app, you will need:

- Python 3.x
- Streamlit
- OpenCV
- Pillow
- Plotly
- Pandas

Install the required dependencies using:

```bash
pip install streamlit opencv-python pillow plotly pandas
```

### Usage

1. Run the application using the following command:

```bash
streamlit run app.py
```

2. Upload an image of your face when prompted.
3. Wait for the app to process the image and display the skin analysis results, including skin type, skin problem detection, and product recommendations.

### Product Recommendations

The app will display top product suggestions based on the skin type and problems detected. Each product is shown with a description, price, and user ratings if available.

### Disclaimer

The analysis provided by **Digital Mirror** is for informational purposes only and should not be considered as professional medical advice. Always consult a dermatologist for specific skin concerns.

---

Let me know if you'd like to adjust any part!
