"""
Digital Mirror: Your Advanced Skin Analysis Application
Created by: Erfantaghavvi
Last Updated: 2025-02-14 15:14:50 UTC
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import cv2
from PIL import Image
import datetime


# Load product recommendations from a CSV file
def load_product_recommendations():
    """Load and return product recommendations from CSV file"""
    try:
        df = pd.read_csv("sorted_cosmetics_products.csv", encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"Error loading product recommendations: {str(e)}")
        return pd.DataFrame()


def recommend_products(skin_type, skin_problems):
    """
    Recommend products based on skin type and detected problems
    Parameters:
        skin_type (str): Detected skin type
        skin_problems (dict): Dictionary of detected skin problems
    Returns:
        DataFrame: Filtered product recommendations
    """
    try:
        df = load_product_recommendations()
        if df.empty:
            return pd.DataFrame()

        # Determine column names
        skin_type_col = next((col for col in ['Skin Type', 'skin_type'] if col in df.columns), None)
        desc_col = next((col for col in ['Description', 'description'] if col in df.columns), None)

        if not skin_type_col:
            st.error("Skin type column not found in CSV file")
            return pd.DataFrame()

        # Filter for skin type
        recommended_products = df[df[skin_type_col].str.contains(skin_type, case=False, na=False)]

        # Filter for skin problems if description column exists
        if skin_problems and desc_col:
            problem_keywords = [p.lower() for p in skin_problems.keys()]
            problem_mask = recommended_products[desc_col].str.lower().apply(
                lambda x: any(k in str(x) for k in problem_keywords)
            )
            recommended_products = recommended_products[problem_mask]

        return recommended_products.head(10)  # Return top 10 recommendations

    except Exception as e:
        st.error(f"Error in product recommendations: {str(e)}")
        return pd.DataFrame()


def preprocess_image(image):
    """Basic image preprocessing function"""
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))
    img_normalized = img_resized / 255.0
    return img_normalized


def analyze_skin_type(img_array):
    """Determine skin type using advanced image analysis"""
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)

    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    y, cr, cb = cv2.split(ycrcb)

    sebum_index = np.mean(s) / 255.0
    moisture_index = np.mean(v) / 255.0
    texture_index = cv2.Laplacian(l, cv2.CV_64F).var()
    redness_index = np.mean(a) / 255.0

    scores = {
        "Dry": 0,
        "Oily": 0,
        "Normal": 0,
        "Combination": 0,
        "Sensitive": 0
    }

    # Scoring logic
    if moisture_index < 0.4:
        scores["Dry"] += 2
    if sebum_index < 0.3:
        scores["Dry"] += 2
    if texture_index > 500:
        scores["Dry"] += 1

    if sebum_index > 0.6:
        scores["Oily"] += 2
    if texture_index < 300:
        scores["Oily"] += 1

    if 0.4 <= moisture_index <= 0.6:
        scores["Normal"] += 2
    if 0.3 <= sebum_index <= 0.6:
        scores["Normal"] += 2
    if 300 <= texture_index <= 500:
        scores["Normal"] += 1

    if abs(sebum_index - moisture_index) > 0.3:
        scores["Combination"] += 2

    if redness_index > 0.55:
        scores["Sensitive"] += 2
    if texture_index > 400:
        scores["Sensitive"] += 1

    total_score = sum(scores.values())
    percentages = {k: (v / total_score * 100) for k, v in scores.items()}

    # Determine skin type with refined logic
    if percentages["Combination"] > 24 and percentages["Dry"] < 13 and percentages["Oily"] > 24:
        skin_type = "Oily"
    elif percentages["Combination"] > 24 and percentages["Dry"] < 13:
        skin_type = "Combination"
    elif percentages["Dry"] > 24 and percentages["Oily"] < 13:
        skin_type = "Dry"
    elif percentages["Normal"] > 40:
        skin_type = "Normal"
    elif percentages["Sensitive"] > 25:
        skin_type = "Sensitive"
    else:
        skin_type = max(scores.items(), key=lambda x: x[1])[0]

    return skin_type, percentages


def detect_skin_problems(img_array, skin_type):
    """Advanced skin problem detection"""
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    min_hsv = np.array([0, 48, 80])
    max_hsv = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, min_hsv, max_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    problems = {}
    problem_regions = np.zeros_like(img_array)

    # Acne detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    acne_mask = cv2.dilate(edges, kernel, iterations=1)
    acne_mask = acne_mask & (skin_mask > 0)

    if np.sum(acne_mask) > 100:
        severity = min(100, (np.sum(acne_mask) / acne_mask.size) * 1000)
        problems['Acne'] = {
            'severity': round(severity, 2),
            'regions': acne_mask,
            'areas': 'Forehead, Cheeks, Chin'
        }
        problem_regions[acne_mask] = [255, 0, 0]  # Red for acne

    # Dark spots detection
    l_channel = lab[:, :, 0]
    dark_spots_mask = cv2.threshold(l_channel, 85, 255, cv2.THRESH_BINARY_INV)[1] > 0
    dark_spots_mask = dark_spots_mask & (skin_mask > 0)

    if np.sum(dark_spots_mask) > 100:
        severity = min(100, (np.sum(dark_spots_mask) / dark_spots_mask.size) * 1000)
        problems['Dark Spots'] = {
            'severity': round(severity, 2),
            'regions': dark_spots_mask,
            'areas': 'Cheeks, Forehead, Nose'
        }
        problem_regions[dark_spots_mask] = [0, 0, 255]  # Blue for dark spots

    # Wrinkles detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    wrinkles_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    wrinkles_mask = wrinkles_mask & (skin_mask > 0)

    if laplacian_var < 200:
        severity = min(100, (np.sum(wrinkles_mask) / wrinkles_mask.size) * 1000)
        problems['Wrinkles'] = {
            'severity': round(severity, 2),
            'regions': wrinkles_mask,
            'areas': 'Forehead, Eyes, Mouth'
        }
        problem_regions[wrinkles_mask] = [0, 255, 0]  # Green for wrinkles

    # Pigmentation detection
    pigmentation_mask = cv2.equalizeHist(l_channel) > 150
    pigmentation_mask = pigmentation_mask & (skin_mask > 0)

    if np.sum(pigmentation_mask) > 100:
        severity = min(100, (np.sum(pigmentation_mask) / pigmentation_mask.size) * 1000)
        problems['Pigmentation'] = {
            'severity': round(severity, 2),
            'regions': pigmentation_mask,
            'areas': 'Cheeks, Forehead, Nose'
        }
        problem_regions[pigmentation_mask] = [255, 255, 0]  # Yellow for pigmentation

    # Rosacea detection
    rosacea_mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 150, 255]))
    rosacea_mask = rosacea_mask & (skin_mask > 0)

    if np.sum(rosacea_mask) > 100:
        severity = min(100, (np.sum(rosacea_mask) / rosacea_mask.size) * 1000)
        problems['Rosacea'] = {
            'severity': round(severity, 2),
            'regions': rosacea_mask,
            'areas': 'Cheeks, Nose, Forehead'
        }
        problem_regions[rosacea_mask] = [255, 105, 180]  # Pink for rosacea

    # Eczema detection only for sensitive skin or combination skin
    if skin_type in {"Sensitive", "Combination"}:
        eczema_mask = cv2.inRange(hsv, np.array([0, 48, 80]), np.array([20, 255, 255]))
        eczema_mask = eczema_mask & (skin_mask > 0)

        if np.sum(eczema_mask) > 100:
            severity = min(100, (np.sum(eczema_mask) / eczema_mask.size) * 1000)
            eczema_severity = severity if skin_type == "Sensitive" else severity * 0.5  # Reduce severity for combination skin
            problems['Eczema'] = {
                'severity': round(eczema_severity, 2),
                'regions': eczema_mask,
                'areas': 'Cheeks, Forehead, Nose'
            }
            problem_regions[eczema_mask] = [255, 105, 180]  # Pink for eczema

    # Large pores detection for oily skin
    if skin_type == "Oily":
        pores_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        pores_mask = pores_mask & (skin_mask > 0)

        if np.sum(pores_mask) > 100:
            severity = min(100, (np.sum(pores_mask) / pores_mask.size) * 1000)
            problems['Large Pores'] = {
                'severity': round(severity, 2),
                'regions': pores_mask,
                'areas': 'Nose, Cheeks, Forehead'
            }
            problem_regions[pores_mask] = [0, 255, 255]  # Cyan for large pores

    return problems, problem_regions


def analyze_skin(image):
    """Comprehensive skin analysis function"""
    try:
        img_array = np.array(image)
        preprocessed_img = preprocess_image(image)
        skin_type, skin_type_percentages = analyze_skin_type(img_array)
        skin_problems, problem_regions = detect_skin_problems(img_array, skin_type)

        results = {
            "skin_type": skin_type,
            "skin_type_percentages": skin_type_percentages,
            "skin_problems": skin_problems
        }

        return results, problem_regions

    except Exception as e:
        st.error(f"Error in skin analysis: {str(e)}")
        return None, None


def display_results(image, results, problem_regions):
    """Display comprehensive skin analysis results"""
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="📸 تصویر اصلی", use_column_width=True)
        overlay = np.array(image).astype(float)
        overlay = overlay * 0.7 + problem_regions.astype(float) * 0.3
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        st.image(overlay, caption="🔍 نمایش نواحی مشکل‌دار", use_column_width=True)

    with col2:
        st.subheader("🎯 نتایج تحلیل")

        # Skin Type Results
        st.write(f"### نوع پوست: {results['skin_type']}")

        fig = px.pie(
            values=list(results['skin_type_percentages'].values()),
            names=list(results['skin_type_percentages'].keys()),
            title="توزیع انواع پوست"
        )
        st.plotly_chart(fig)

        # Skin Problems
        if results['skin_problems']:
            st.markdown("### 🔍 مشکلات پوستی شناسایی شده")
            sorted_problems = sorted(
                results['skin_problems'].items(),
                key=lambda x: x[1]['severity'],
                reverse=True
            )[:5]

            for problem_type, details in sorted_problems:
                with st.expander(f"**{problem_type}** - شدت: {details['severity']}%"):
                    st.write(f"نواحی تحت تأثیر: {details['areas']}")
                    st.write(f"توضیحات: {get_problem_description(problem_type)}")

            # Product Recommendations
            st.markdown("### 💡 محصولات پیشنهادی")
            recommended_products = recommend_products(results['skin_type'], results['skin_problems'])

            if not recommended_products.empty:
                for _, product in recommended_products.iterrows():
                    # Get column names dynamically
                    name_col = next((col for col in ['Product Name', 'product_name'] if col in product.index), None)
                    desc_col = next((col for col in ['Description', 'description'] if col in product.index), None)
                    price_col = next((col for col in ['Price', 'price'] if col in product.index), None)

                    if name_col:
                        with st.expander(f"**{product[name_col]}**"):
                            if desc_col:
                                st.write(f"توضیحات: {product[desc_col]}")
                            if price_col:
                                st.write(f"قیمت: {product[price_col]}")
                            if 'rating' in product.index:
                                st.write(f"امتیاز: {'⭐' * int(product['rating'])}")
            else:
                st.info("محصول پیشنهادی برای شرایط پوستی شما یافت نشد.")

            st.info("""
                💡 **نکته مهم**: این تحلیل‌ها صرفاً جنبه راهنمایی دارند. 
                همیشه برای درمان تخصصی با متخصص پوست مشورت کنید.
            """)
        else:
            st.success("هیچ مشکل پوستی قابل توجهی یافت نشد!")


def get_problem_description(problem_type):
    """Return description for skin problem types"""
    descriptions = {
        'Acne': 'التهاب پوست که با جوش، جوش سر سفید یا جوش سر سیاه مشخص می‌شود.',
        'Dark Spots': 'لکه‌های تیره یا نقاط تیره‌تر از رنگ پوست اطراف.',
        'Wrinkles': 'خطوط و چین و چروک‌های پوست ناشی از پیری، نور خورشید و حالت‌های تکراری صورت.',
        'Pigmentation': 'رنگ پوست ناهموار و لکه‌های تیره ناشی از تولید بیش از حد ملانین.',
        'Rosacea': 'قرمزی و التهاب مزمن، اغلب با رگ‌های خونی قابل مشاهده و جوش‌های قرمز کوچک.',
        'Eczema': 'التهاب پوست که باعث قرمزی، خارش و پوسته‌پوسته شدن می‌شود.',
        'Large Pores': 'منافذ پوستی قابل توجه، اغلب به دلیل تولید بیش از حد چربی یا پیری.'
    }
    return descriptions.get(problem_type, "توضیحات در دسترس نیست.")


def main():
    st.set_page_config(
        page_title="آینه دیجیتالی پوست شما",
        page_icon="🌟",
        layout="wide"
    )

    st.title("🌟 آینه دیجیتالی پوست شما")
    st.write("تصویر خود را برای تحلیل جامع پوست آپلود کنید")

    uploaded_file = st.file_uploader("📸 آپلود تصویر", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)

            with st.spinner('در حال تحلیل پوست شما...'):
                results, problem_regions = analyze_skin(image)

                if results is not None and problem_regions is not None:
                    display_results(image, results, problem_regions)
                else:
                    st.error("تحلیل پوست کامل نشد. لطفاً تصویر دیگری را امتحان کنید.")

        except Exception as e:
            st.error(f"خطا در پردازش تصویر: {str(e)}")

    # Add footer with timestamp
    st.markdown("---")
    st.markdown(f"آخرین به‌روزرسانی: 2025-02-14 15:20:49 UTC")


if __name__ == "__main__":
    main()